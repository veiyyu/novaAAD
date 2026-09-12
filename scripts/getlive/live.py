"""Live acceptance test: run the streaming package against the real EEG cap.

Run from the repository root, with the amplifier publishing LSL:

    .venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30

What it does, in order:

1. selects the amplifier outlet (the single EEG-like one, or the identity given
   with ``--stream-name`` / ``--source-id`` / ``--stream-type``);
2. connects, reads what the outlet declares, and resolves the **cap profile**:
   ``--cap auto`` (default) uses the CA-208 datasheet contract when the declared
   labels cover it and otherwise builds the contract from the declared montage,
   so any cap the amplifier drives can be brought up without editing a table;
3. runs the package pre-flight check (``prepare``), which validates identity,
   rate, declared units, channel labels and channel types **before** a sample
   is used;
4. builds the same preprocessing chain as ``scripts/streaming_demo.py``:
   repair -> source units to uV -> quality observation -> notch -> band-pass ->
   resample to the model rate, then 2 s windows every 0.5 s;
5. streams for ``--duration`` seconds and prints one line per window, naming the
   electrodes the quality monitor found faulty;
6. scores the run against ``checks.evaluate`` and prints a verdict plus a
   per-electrode summary, optionally writing both as JSON with ``--out``.

There is no synthetic source here on purpose: this script is the hardware test.
Use ``scripts.getlive.probe`` first to read what the outlet declares, because
``--sfreq`` and ``--source-units`` are operator assertions that must match it.

Known-dead electrodes (dry caps) are supported explicitly: pass
``--exclude-channels`` to keep them out of the fault verdict while still
recording them, ``--max-bad-channels`` to tolerate a census, or
``--no-channel-check`` to record without ever letting a channel stop the run.

The run's mutable state lives in one :class:`_Run` context, so every helper
below takes a single argument and the live loop stays readable.

Exit codes: 0 accepted, 1 rejected or failed, 130 stopped with Ctrl+C (the
report is still printed and written).
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import numpy as np
from mne_lsl.lsl import local_clock

from nova2026.config import SAMPLE_RATE as MODEL_SAMPLE_RATE
from nova2026.config import WINDOW_SIZE as MODEL_WINDOW_SIZE_MS
from nova2026.streaming import (
    Recovery,
    StreamStats,
    UnrepairableError,
    prepare,
    processing_contract,
)
from nova2026.streaming.bootstrap import StreamSession, make_parser
from nova2026.streaming.preprocess import (
    QualityMonitor,
    Repair,
    Resampler,
    SosFilter,
    design_bandpass,
    design_notch,
    unit_scaler,
)

from .cap import (
    CAP_MODES,
    CA208,
    CHANNEL_MODES,
    EOG_MODES,
    MODEL_EEG_CHANNELS,
    CapProfile,
    select_profile,
    resolve_channels,
)
from .checks import RunFacts, evaluate
from .electrodes import ChannelStats
from .outlets import channel_facts, open_inlet, wait_for_outlet
from .report import (
    build_report,
    format_channels,
    format_run,
    format_setup,
    format_window,
    write_report,
)

# Declared source units, as the exponent the package reasons in.
SOURCE_UNITS = {"V": 0, "mV": -3, "uV": -6, "nV": -9}

# Window length the offline model was trained with, in seconds.
MODEL_WINDOW_SECONDS = MODEL_WINDOW_SIZE_MS / 1000.0


def channel_list(text: str) -> tuple[str, ...]:
    """Split a comma-separated channel option into labels, dropping blanks."""

    if not isinstance(text, str):
        raise TypeError("A channel list option must be a string.")
    return tuple(name.strip() for name in text.split(",") if name.strip())


def abbreviate(names: tuple[str, ...], limit: int = 12) -> str:
    """Render a channel list for an error message, shortened when long."""

    shown = ", ".join(names[:limit])
    return shown + (f", ... (+{len(names) - limit} more)" if len(names) > limit else "")


@dataclass
class _Run:
    """One live acceptance run: the decisions in, the measurement out.

    Every helper below takes this single object rather than a dozen arguments.
    The first eight fields are settled before streaming starts; the rest are
    filled in by :func:`_prepare` and the live loop.
    """

    args: argparse.Namespace
    stream: object
    source: dict
    contract: object
    profile: CapProfile
    eeg: tuple[str, ...]
    eog: tuple[str, ...]
    excluded: tuple[str, ...]

    repair: Repair | None = None
    quality: QualityMonitor | None = None
    resampler: Resampler | None = None
    stages: tuple = ()
    resettable: tuple = ()
    session: object | None = None
    recovery: object | None = None
    stats: StreamStats | None = None
    electrodes: ChannelStats | None = None
    reasons: Counter = field(default_factory=Counter)
    bad_channel_windows: Counter = field(default_factory=Counter)
    warmup_windows: int = 0
    interrupted: bool = False
    failure: Exception | None = None
    started: float = 0.0
    elapsed: float = 0.0

    @property
    def channels(self) -> tuple[str, ...]:
        """The contracted channels: EEG first, then auxiliary."""

        return self.eeg + self.eog


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Register how to find and reach the amplifier outlet."""

    source = parser.add_argument_group("amplifier outlet")
    source.add_argument("--stream-name", help="LSL outlet name, as probe.py prints it")
    source.add_argument("--source-id", help="LSL source id, as probe.py prints it")
    source.add_argument(
        "--stream-type",
        help="LSL stream type, exact spelling as probe.py prints it",
    )
    source.add_argument(
        "--resolve-timeout",
        type=float,
        default=10.0,
        help="seconds to keep looking for the outlet",
    )
    source.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="seconds allowed for the inlet connection",
    )
    source.add_argument(
        "--expect-channels",
        type=int,
        help="with no declared stream type, treat an outlet with at least this "
        "many channels as the amplifier (--cap ca-208 implies 64)",
    )


def _add_contract_args(parser: argparse.ArgumentParser) -> None:
    """Register the cap and channel-contract options."""

    contract = parser.add_argument_group("channel contract")
    contract.add_argument(
        "--cap",
        choices=CAP_MODES,
        default="auto",
        help="auto (default): the CA-208 datasheet contract when the declared "
        "labels cover it, otherwise the declared montage; ca-208: demand the "
        "datasheet; declared: always trust the outlet",
    )
    contract.add_argument(
        "--source-units",
        choices=tuple(SOURCE_UNITS),
        default=None,
        help="units the outlet declares; must be stated (probe.py prints them)",
    )
    contract.add_argument(
        "--channels",
        choices=CHANNEL_MODES,
        default="cap",
        help="cap: every EEG electrode of the resolved cap; model: the "
        "electrodes the offline classifier uses that this cap also has",
    )
    contract.add_argument(
        "--eog",
        choices=EOG_MODES,
        default="eog",
        help="eog: keep the cap's EOG electrode as auxiliary; drop: exclude it "
        "from the contract",
    )
    contract.add_argument(
        "--reference",
        default=None,
        help="reference already applied by the amplifier, recorded as "
        "provenance; defaults to the profile's own reference",
    )
    contract.add_argument(
        "--ground",
        default=None,
        help="ground of the cap, recorded as provenance; defaults to the "
        "profile's own ground",
    )
    contract.add_argument(
        "--upstream",
        default="eego software LSL streaming; gain and filters as configured there",
        help="upstream processing, recorded as provenance",
    )


def _add_quality_args(parser: argparse.ArgumentParser) -> None:
    """Register the channel-quality policy (dead electrodes on a dry cap)."""

    quality = parser.add_argument_group("channel quality policy")
    quality.add_argument(
        "--exclude-channels",
        default="",
        help="comma-separated EEG labels known to be dead (a dry cap): they are "
        "still recorded, but never reject a window",
    )
    quality.add_argument(
        "--max-bad-channels",
        type=int,
        default=0,
        help="faulting EEG channels tolerated per fault type and window before "
        "quality rejects it (default 0: any faulty channel rejects)",
    )
    quality.add_argument(
        "--no-channel-check",
        action="store_true",
        help="record channel faults but never let them reject a window",
    )


def _add_scoring_args(parser: argparse.ArgumentParser) -> None:
    """Register the acceptance thresholds and the output paths."""

    scoring = parser.add_argument_group("acceptance")
    scoring.add_argument(
        "--min-valid-windows",
        type=int,
        default=3,
        help="valid windows required for acceptance",
    )
    scoring.add_argument(
        "--flat-uv",
        type=float,
        default=1.0,
        help="mean peak-to-peak below this counts as a flat electrode",
    )
    scoring.add_argument(
        "--noisy-uv",
        type=float,
        default=200.0,
        help="mean peak-to-peak above this counts as a noisy electrode",
    )
    scoring.add_argument("--quiet", action="store_true", help="no per-window lines")
    scoring.add_argument("--out", type=Path, help="write the acceptance report as JSON")
    scoring.add_argument("--role", default="bringup", help="run role stored when recording")


def _add_args(parser: argparse.ArgumentParser) -> None:
    """Register the hardware-specific options on top of the shared ones."""

    _add_source_args(parser)
    _add_contract_args(parser)
    _add_quality_args(parser)
    _add_scoring_args(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the shared live-stream parser plus the hardware options."""

    parser = make_parser(description=__doc__, extra=_add_args)
    # Source facts are never defaulted: a wrong amplifier setting must fail the
    # run loudly instead of passing against a guessed default.
    parser.set_defaults(sfreq=None, source_units=None, duration=30.0)
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject impossible flag combinations before touching the network."""

    missing = [
        flag
        for flag, value in (("--sfreq", args.sfreq), ("--source-units", args.source_units))
        if value is None
    ]
    if missing:
        parser.error(
            "required: "
            + ", ".join(missing)
            + " - read the declared values with: python -m scripts.getlive.probe"
        )
    for flag, value in (
        ("--sfreq", args.sfreq),
        ("--duration", args.duration),
        ("--out-sfreq", args.out_sfreq),
        ("--window", args.window),
        ("--hop", args.hop),
        ("--capacity", args.capacity),
        ("--resolve-timeout", args.resolve_timeout),
        ("--connect-timeout", args.connect_timeout),
    ):
        if value <= 0:
            parser.error(f"{flag} must be positive.")
    if args.block < 1:
        parser.error("--block must be a positive number of samples.")
    if args.min_valid_windows < 1:
        parser.error("--min-valid-windows must be at least 1.")
    if args.noisy_uv < args.flat_uv:
        parser.error("--noisy-uv must not be below --flat-uv.")
    if args.max_bad_channels < 0:
        parser.error("--max-bad-channels must not be negative.")
    if args.expect_channels is not None and args.expect_channels < 1:
        parser.error("--expect-channels must be a positive channel count.")


def _diagnose(message: str, args: argparse.Namespace, source: dict, profile) -> str:
    """Turn a pre-flight rejection into the flag that probably fixes it."""

    hints = []
    if "channel types" in message:
        declared = dict(zip(source["channels"], source["types"]))
        auxiliary = tuple(profile.eog_channels)
        kinds = {name: declared.get(name) for name in auxiliary}
        wrong = {name: kind for name, kind in kinds.items() if kind not in (None, "eog")}
        if wrong:
            names = ", ".join(f"{name} as {kind!r}" for name, kind in wrong.items())
            hints.append(
                f"The outlet declares {names}, while the package requires EEG "
                "channels declared 'eeg' and auxiliary channels 'eog'. Rerun "
                "with --eog drop to test the EEG electrodes on their own."
            )
        else:
            hints.append(
                "The declared channel types do not match the contract (EEG "
                "'eeg', auxiliary 'eog'); check the montage the control "
                "software publishes, or rerun with --eog drop."
            )
    if "units" in message:
        hints.append(
            "Compare the declared ch_units from probe.py with --source-units "
            f"(currently {args.source_units})."
        )
    if "sampling rate" in message:
        hints.append(
            f"Set the amplifier sampling rate to {args.sfreq:g} Hz, or pass the "
            "rate probe.py reported."
        )
    if "missing required channels" in message:
        hints.append(
            "The montage the control software publishes does not carry every "
            f"expected electrode of the {profile.name} contract; compare the "
            "probe output with the montage, or use --cap declared to run "
            "against exactly what the outlet declares."
        )
    if "does not match the configuration" in message:
        hints.append("Use the exact identity probe.py prints.")
    return " ".join(hints)


def _build_chain(run: _Run) -> None:
    """Build the preprocessing chain, in the order it runs.

    Repair works in source units, quality observes raw uV, filters and
    resampling come last. Both stages that could stop the run for a single
    electrode take the same dead-channel list, so no path is left unguarded.
    """

    args = run.args
    channels = run.channels
    unit_exponent = SOURCE_UNITS[args.source_units]
    run.repair = Repair(
        args.sfreq,
        # Keep the grid tolerance below half a sample at any source rate.
        tolerance_seconds=min(2e-4, 0.4 / args.sfreq),
        source_unit_exponent=unit_exponent,
        n_eeg=len(run.eeg),
        channel_names=run.eeg,
        exclude_channels=run.excluded,
    )
    scaler = unit_scaler(unit_exponent, desired_exponent=-6)
    run.quality = QualityMonitor(
        n_eeg=len(run.eeg),
        sfreq=args.sfreq,
        warmup_seconds=0.0,
        channel_names=run.eeg,
        check_channels=not args.no_channel_check,
        max_bad_channels=args.max_bad_channels,
        exclude_channels=run.excluded,
    )
    quality = run.quality

    def observe(data, timestamps):
        # Watch the signal in uV, before any filter can hide a fault.
        quality.feed(data, timestamps)
        return data, timestamps

    notch = SosFilter(design_notch(args.notch, args.notch_q, args.sfreq), len(channels))
    bandpass = SosFilter(
        design_bandpass(args.lpass, args.hpass, args.order, args.sfreq), len(channels)
    )
    run.resampler = Resampler(
        args.sfreq, args.out_sfreq, len(channels), quality=args.resample_quality
    )
    run.stages = (run.repair, scaler, observe, notch, bandpass, run.resampler)
    # The session's ring buffer joins these when a recovery resets the chain.
    run.resettable = (run.repair, quality, notch, bandpass, run.resampler)


def _source_provenance(source: dict) -> dict:
    """The part of the run record that describes what the outlet declared."""

    return {
        "name": source["name"],
        "type": source["stype"],
        "source_id": source["source_id"],
        "sfreq": source["sfreq"],
        "channels": list(source["channels"]),
        "types": list(source["types"]),
        "units": None if source["units"] is None else list(source["units"]),
    }


def _chain_provenance(run: _Run) -> dict:
    """Provenance for the run record: what the windows mean, and who asserted what."""

    args = run.args
    resampler = run.resampler
    config = processing_contract(
        eeg_channels=run.eeg,
        eog_channels=run.eog,
        out_sfreq=resampler.out_sfreq,
        stamp=(
            f"notch{args.notch:g}-q{args.notch_q:g}/"
            f"band{args.lpass:g}-{args.hpass:g}-o{args.order}/"
            f"soxr-{resampler.quality}"
        ),
    )
    config.update(
        {
            "channel_mode": args.channels,
            "window_seconds": args.window,
            "step_seconds": args.hop,
            "warmup_seconds": args.warmup,
            "cap": {
                "profile": run.profile.name,
                "source": run.profile.source,
                "reference": run.profile.reference,
                "ground": run.profile.ground,
                "eog_channels": list(run.eog),
                "other_auxiliary": list(run.profile.other_auxiliary),
            },
            "channel_policy": {
                "check_channels": not args.no_channel_check,
                "max_bad_channels": args.max_bad_channels,
                "excluded_channels": list(run.excluded),
            },
            "assertions": {
                "reference": args.reference or run.profile.reference or "not asserted",
                "ground": args.ground or run.profile.ground or "not asserted",
                "upstream_processing": args.upstream,
            },
            "source": _source_provenance(run.source),
        }
    )
    return config


def _prepare(run: _Run) -> None:
    """Build the chain, the session and the counters for one run."""

    args = run.args
    _build_chain(run)
    # The script owns the chain; the session owns the contract, the optional
    # recorder, acquisition and the window gate.
    run.session = StreamSession(
        run.stream,
        args,
        run.channels,
        judges=(run.quality, run.repair),
        out_sfreq=run.resampler.out_sfreq,
        source_unit_exponent=SOURCE_UNITS[args.source_units],
        role=args.role,
        n_eeg=len(run.eeg),
        ch_types=("eeg",) * len(run.eeg) + ("eog",) * len(run.eog),
        contract=run.contract,
        recorder_config=_chain_provenance(run),
        recorder_track_windows=True,
    )
    run.recovery = Recovery(
        resettable=run.resettable + (run.session.buffer,),
        recorder=run.session.recorder,
    )
    run.stats = StreamStats()
    run.electrodes = ChannelStats(
        run.channels, flat_uv=args.flat_uv, noisy_uv=args.noisy_uv
    )


def _handle_window(run: _Run, window, window_times, start) -> None:
    """Package, judge and count one finished window."""

    stats = run.stats
    session = run.session
    stats.windows += 1
    eeg_window = session.wrap(window, window_times, start)
    eeg_window.segment = run.recovery.segment
    run.recovery.watch(eeg_window)
    if session.recorder is not None:
        session.recorder.log_window(
            float(window_times[-1]),
            eeg_window.valid,
            eeg_window.reasons,
            eeg_window.segment,
            eeg_window.artifact_id,
        )
    for reason in eeg_window.reasons:
        run.reasons[reason] += 1
    # Faulting channels are recorded whether or not they rejected the window:
    # the names decide the next session's exclusion list.
    for name in eeg_window.bad_channels:
        run.bad_channel_windows[name] += 1
    if eeg_window.valid:
        stats.valid += 1
        run.electrodes.update(eeg_window.data, eeg_window.eog)
    else:
        stats.rejected += 1
        if not eeg_window.reasons:
            # No judge complained: the window is warm-up.
            run.warmup_windows += 1
    if not run.args.quiet:
        print(
            format_window(
                stats.windows,
                monotonic() - run.started,
                local_clock() - float(eeg_window.timestamps[-1]),
                eeg_window,
            )
        )


def _loop(run: _Run) -> None:
    """Stream one acquired block per iteration until the time or a stop."""

    args = run.args
    stats = run.stats
    session = run.session
    repair = run.repair
    run.started = monotonic()
    print(format_setup(args, session, run.resampler))

    try:
        while monotonic() - run.started < args.duration:
            data, timestamps = session.acquire.read()
            stats.blocks += 1
            stats.samples += len(data)
            data, timestamps = session.ingest(data, timestamps)

            try:
                for stage in run.stages:
                    data, timestamps = stage(data, timestamps)
            except UnrepairableError as error:
                # Bounded recovery: reset the whole chain and keep going.
                run.recovery.handle(error)
                stats.recoveries = run.recovery.recoveries
                print(
                    f"[{monotonic() - run.started:7.2f}s] source damage "
                    f"({error.kind}); chain restarted as segment "
                    f"{run.recovery.segment}"
                )
                continue

            stats.repairs = repair.repaired_samples
            stats.dropped = repair.dropped_rows

            for window, window_times, start in session.buffer.push(data, timestamps):
                _handle_window(run, window, window_times, start)
    except KeyboardInterrupt:
        # Ctrl+C is how a hardware test normally ends: finalize, keep the data.
        run.interrupted = True
        print("\nstopped by Ctrl+C; finalizing the run")
    except Exception as error:  # noqa: BLE001 - reported, then scored by checks
        run.failure = error
        print(f"\nrun stopped: {error}")
    finally:
        run.elapsed = monotonic() - run.started
        # Samples buffered toward the next block never became windows: record
        # them honestly before the acquire handle drops them.
        tail = session.acquire.pending_samples
        session.acquire.close()
        if session.recorder is not None and tail:
            session.recorder.mark_not_processed(tail)
        stats.max_lag = session.acquire.max_lag
        stats.gaps = session.acquire.gaps
        # Authoritative count: a recovery that raised still happened, and the
        # report must not claim the chain never restarted.
        stats.recoveries = run.recovery.recoveries
        session.close(
            status="completed" if run.failure is None else "failed",
            error=None if run.failure is None else repr(run.failure),
            stats=stats.to_dict(),
        )


def _channel_report(run: _Run) -> dict:
    """The channel section of the JSON report: contract, policy and evidence."""

    args = run.args
    return {
        "profile": run.profile.name,
        "profile_source": run.profile.source,
        "profile_detail": run.profile.detail,
        "cap_mode": args.cap,
        "channel_mode": args.channels,
        "eeg_channels": list(run.eeg),
        "eog_channels": list(run.eog),
        "other_auxiliary": list(run.profile.other_auxiliary),
        "reference": args.reference or run.profile.reference,
        "ground": args.ground or run.profile.ground,
        "check_channels": not args.no_channel_check,
        "max_bad_channels": args.max_bad_channels,
        "excluded_channels": list(run.excluded),
        "bad_channel_windows": dict(sorted(run.bad_channel_windows.items())),
        "held_rows": int(run.repair.held_rows),
        "model_channels": len(MODEL_EEG_CHANNELS),
        "model_missing": [
            name for name in MODEL_EEG_CHANNELS if name not in set(run.channels)
        ],
    }


def _print_verdict(run: _Run, acceptance, electrode_summary: dict) -> None:
    """Print the run counters, the per-electrode table and the acceptance block."""

    print(format_run(run.stats, run.elapsed, run.resampler))
    print(f"\nper-electrode ({run.electrodes.windows} valid window(s)):")
    print(run.electrodes.format_worst())
    print(f"\nacceptance:\n{acceptance.format()}")
    if run.session.recorder is not None:
        print(
            f"\nrecorded {run.session.recorder.samples} raw samples -> "
            f"{run.session.recorder.path}"
        )


def _facts(run: _Run, electrode_summary: dict, model_missing: tuple[str, ...]) -> RunFacts:
    """Map the measured run onto the acceptance rules' input."""

    args = run.args
    return RunFacts(
        expected_sfreq=args.sfreq,
        source_sfreq=run.source["sfreq"],
        expected_channels=len(run.channels),
        source_channels=run.source["n_channels"],
        dropped_channels=tuple(
            name for name in run.source["channels"] if name not in set(run.channels)
        ),
        window_reasons=dict(run.reasons),
        warmup_windows=run.warmup_windows,
        stats=run.stats.to_dict(),
        resampler_quality=run.resampler.quality,
        resampler_delay=run.resampler.startup_delay_seconds,
        out_sfreq=run.resampler.out_sfreq,
        window_seconds=args.window,
        model_sfreq=float(MODEL_SAMPLE_RATE),
        model_window_seconds=float(MODEL_WINDOW_SECONDS),
        flat_channels=electrode_summary["flat"],
        noisy_channels=electrode_summary["noisy"],
        channel_count=len(run.channels),
        flat_uv=args.flat_uv,
        noisy_uv=args.noisy_uv,
        duration=args.duration,
        elapsed=run.elapsed,
        interrupted=run.interrupted,
        failure=None if run.failure is None else repr(run.failure),
        min_valid_windows=args.min_valid_windows,
        cap_profile=run.profile.name,
        cap_mode=args.cap,
        excluded_channels=run.excluded,
        check_channels=not args.no_channel_check,
        max_bad_channels=args.max_bad_channels,
        bad_channel_windows=dict(run.bad_channel_windows),
        held_rows=int(run.repair.held_rows),
        model_channels=len(MODEL_EEG_CHANNELS),
        model_missing=model_missing,
    )


def _finish(run: _Run) -> int:
    """Score the run, print the verdict and write the report."""

    args = run.args
    session = run.session
    stats = run.stats
    electrode_summary = run.electrodes.summary()
    model_missing = tuple(
        name for name in MODEL_EEG_CHANNELS if name not in set(run.channels)
    )
    acceptance = evaluate(_facts(run, electrode_summary, model_missing))
    _print_verdict(run, acceptance, electrode_summary)

    if args.out is not None:
        write_report(
            args.out,
            build_report(
                args=args,
                source=run.source,
                acceptance=acceptance,
                stats=stats,
                electrodes=electrode_summary,
                resampler=run.resampler,
                recording=None if session.recorder is None else session.recorder.path,
                channels=_channel_report(run),
            ),
        )
        print(f"report -> {args.out}")

    if run.failure is not None:
        return 1
    if run.interrupted:
        return 130
    return 0 if acceptance.ok else 1


def _stream(run: _Run) -> int:
    """Run the preprocessing chain on the connected amplifier."""

    _prepare(run)
    _loop(run)

    return _finish(run)


def _resolve_cap(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    source: dict,
) -> tuple[CapProfile, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    """Resolve the cap profile, the channel contract and the policy.

    Returns ``(profile, eeg, eog, excluded)``, or ``None`` after reporting why
    the contract cannot be built.
    """

    try:
        profile = select_profile(
            args.cap,
            source["channels"],
            source["types"],
            reference=args.reference,
            ground=args.ground,
        )
    except ValueError as error:
        print(f"\ncap profile rejected: {error}")
        return None
    if args.cap == "auto" and profile is not CA208:
        print(
            "note      : the declared montage does not cover the CA-208 "
            f"datasheet; using the declared profile ({profile.n_eeg} EEG "
            "electrodes). Order and completeness cannot be checked."
        )

    try:
        eeg, eog = resolve_channels(profile, args.channels, args.eog)
    except ValueError as error:
        print(f"\nchannel contract rejected: {error}")
        return None

    excluded = channel_list(args.exclude_channels)
    unknown = [name for name in excluded if name not in set(eeg)]
    if unknown:
        parser.error(
            "--exclude-channels names channel(s) that are not EEG electrodes "
            f"of this contract: {', '.join(unknown)}\nEEG electrodes are: "
            f"{abbreviate(tuple(eeg))}"
        )
    if args.max_bad_channels >= len(eeg):
        parser.error(
            f"--max-bad-channels must be below the {len(eeg)} EEG electrodes; "
            "use --no-channel-check to stop judging channels on purpose."
        )

    return profile, tuple(eeg), tuple(eog), excluded


def _report_outlet(args: argparse.Namespace, row: dict) -> None:
    """Print the selected outlet and warn when its rate disagrees with --sfreq."""

    print(
        f"\noutlet    : name={row['name']!r} type={row['stype']!r} "
        f"channels={row['n_channels']} sfreq={row['sfreq']:g} Hz "
        f"source_id={row['source_id']!r}"
    )
    if not np.isclose(row["sfreq"], args.sfreq, rtol=0, atol=1e-9):
        print(
            f"warning   : the outlet publishes {row['sfreq']:g} Hz but --sfreq is "
            f"{args.sfreq:g}; the pre-flight check will reject this run"
        )


def _connect(args: argparse.Namespace, row: dict):
    """Open an inlet for the selected outlet, or report why it failed."""

    bufsize = max(args.capacity + 1.0, 2.0)
    try:
        return open_inlet(row, bufsize=bufsize, connect_timeout=args.connect_timeout)
    except Exception as error:  # noqa: BLE001 - reported to the operator
        print(f"\nerror: could not connect to the outlet: {error}")
        return None


def _announce_contract(
    source: dict,
    profile: CapProfile,
    eeg: tuple[str, ...],
    eog: tuple[str, ...],
    excluded: tuple[str, ...],
) -> None:
    """Print what was declared and which contract this run resolved to."""

    print(
        f"declared  : {source['n_channels']} channels, {source['sfreq']:g} Hz, "
        f"types {sorted(set(source['types']))}, "
        f"units {sorted(set(source['units'] or []))}"
    )
    print(format_channels(profile, eeg, eog, excluded))
    if source["n_channels"] != len(eeg) + len(eog):
        dropped = [name for name in source["channels"] if name not in set(eeg + eog)]
        print(
            f"contract  : {len(eeg) + len(eog)} of {source['n_channels']} channels "
            "kept"
            + (f"; dropped {abbreviate(tuple(dropped))}" if dropped else "")
        )


def main(argv: list[str] | None = None) -> int:
    """Validate the arguments, connect the amplifier and stream."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    # A datasheet cap knows its own size, so outlet selection can use it; an
    # auto or declared profile is sized by whatever the outlet publishes.
    hint_channels = len(CA208.channels) if args.cap == "ca-208" else args.expect_channels
    print(
        f"looking for: {args.cap} cap profile, {args.sfreq:g} Hz, "
        f"units {args.source_units}"
        + (f", at least {hint_channels} channels" if hint_channels else "")
    )

    try:
        row = wait_for_outlet(
            name=args.stream_name,
            source_id=args.source_id,
            stream_type=args.stream_type,
            timeout=args.resolve_timeout,
            expected_channels=hint_channels,
        )
    except RuntimeError as error:
        print(f"\nerror: {error}")
        return 1
    _report_outlet(args, row)

    stream = _connect(args, row)
    if stream is None:
        return 1

    try:
        source = channel_facts(stream)
        resolved = _resolve_cap(parser, args, source)
        if resolved is None:
            return 1
        profile, eeg, eog, excluded = resolved
        args.excluded_channels = excluded
        _announce_contract(source, profile, eeg, eog, excluded)

        try:
            contract = prepare(
                stream,
                sfreq=args.sfreq,
                channels=eeg + eog,
                source_unit_exponent=SOURCE_UNITS[args.source_units],
                n_eeg=len(eeg),
                stream_name=args.stream_name,
                source_id=args.source_id,
                stream_type=args.stream_type,
            )
        except RuntimeError as error:
            print(f"\npre-flight rejected the source: {error}")
            hint = _diagnose(str(error), args, source, profile)
            if hint:
                print(f"hint: {hint}")
            return 1

        return _stream(
            _Run(
                args=args,
                stream=stream,
                source=source,
                contract=contract,
                profile=profile,
                eeg=eeg,
                eog=eog,
                excluded=excluded,
            )
        )
    finally:
        if stream.connected:
            stream.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
