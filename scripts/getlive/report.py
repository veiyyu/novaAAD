"""Text and JSON reporting for one bring-up run.

Keeping this out of ``live`` serves two purposes: the live script stays
about connecting and streaming, and the wording of every line can be tested
with plain values, without an amplifier.
"""

import json
from pathlib import Path

import numpy as np


def format_window(index: int, elapsed: float, age: float, window) -> str:
    """Render one window as a single console line.

    Args:
        index: Window number since the run started.
        elapsed: Seconds since the run started.
        age: Age of the window's newest sample against the local LSL clock.
        window: The ``EEGWindow`` (or any object with ``data``, ``eog``,
            ``valid``, ``reasons``, ``segment``, ``timestamps`` and optionally
            ``bad_channels``).

    Notes:
        ``bad_channels`` is evidence and is printed whether or not it rejected
        the window: a tolerated dead electrode must stay visible in the log.
    """

    bad = getattr(window, "bad_channels", ())
    evidence = f" bad={','.join(bad)}" if bad else ""

    if not window.valid:
        reasons = ",".join(window.reasons) if window.reasons else "warmup"
        return (
            f"[{elapsed:7.2f}s] win{index:4d} seg{window.segment} "
            f"REJECT  reasons={reasons}{evidence} lag={age:6.3f}s"
        )

    peak_to_peak = float(np.max(np.ptp(window.data, axis=0)))
    rms = float(np.median(np.sqrt(np.mean(window.data**2, axis=0))))
    extra = ""
    if window.eog.size:
        extra = f" eog_ptp={float(np.max(np.ptp(window.eog, axis=0))):7.1f} uV"
    return (
        f"[{elapsed:7.2f}s] win{index:4d} seg{window.segment} valid   "
        f"ptp_max={peak_to_peak:7.1f} uV rms_med={rms:6.2f} uV{extra}{evidence} "
        f"lag={age:6.3f}s"
    )


def format_run(stats, elapsed: float, resampler) -> str:
    """Render the transport, window and chain counters of a finished run."""

    return (
        f"\nrun    : {stats.blocks} blocks, {stats.samples} samples, "
        f"{elapsed:.1f}s wall clock, max_lag {stats.max_lag:.3f}s, "
        f"gaps {stats.gaps}\n"
        f"windows: {stats.windows} total, {stats.valid} valid, "
        f"{stats.rejected} rejected\n"
        f"recovery: {stats.recoveries} restart(s), {stats.repairs} repaired row(s), "
        f"{stats.dropped} dropped row(s)\n"
        f"output : {resampler.output_samples} samples @ {resampler.out_sfreq:g} Hz"
    )


def format_setup(args, session, resampler) -> str:
    """Render what this run is about to do, before any sample arrives."""

    return (
        f"\nchain : {args.source_units}->uV, notch {args.notch:g} Hz, "
        f"band {args.lpass:g}-{args.hpass:g} Hz, "
        f"resample {args.sfreq:g}->{resampler.out_sfreq:g} Hz "
        f"({resampler.quality}, startup delay "
        f"{resampler.startup_delay_seconds:.2f}s)\n"
        f"buffer: window={session.window_samples} hop={session.hop_samples} "
        f"capacity={session.capacity_samples} samples @ {session.out_sfreq:g} Hz, "
        f"warm-up {session.warmup_samples} samples\n"
        f"run   : {args.duration:g}s, block={args.block} samples"
        + (f", recording under {args.record}" if args.record else "")
    )


def format_channels(profile, eeg, eog, exclusions: tuple[str, ...] = ()) -> str:
    """Render the cap contract this run will use, before any sample arrives."""

    return (
        f"\ncap   : {profile.name} ({profile.source}) - {profile.detail}\n"
        f"contract: {len(eeg)} EEG"
        + (f" + {len(eog)} EOG ({', '.join(eog)})" if eog else " + no auxiliary")
        + (f", known dead (never fatal): {', '.join(exclusions)}" if exclusions else "")
        + f"\nassertions: reference {profile.reference or 'not asserted by the profile'}"
        f", ground {profile.ground or 'not asserted by the profile'}"
    )


def build_report(
    *,
    args,
    source: dict,
    acceptance,
    stats,
    electrodes: dict,
    resampler,
    recording: Path | None,
    channels: dict | None = None,
) -> dict:
    """Assemble the JSON acceptance report."""

    report = {
        "accepted": acceptance.ok,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": source,
        "acceptance": acceptance.to_dict(),
        "stats": stats.to_dict(),
        "electrodes": electrodes,
        "resampler": {
            "quality": resampler.quality,
            "startup_delay_seconds": resampler.startup_delay_seconds,
            "out_sfreq": resampler.out_sfreq,
            "output_samples": resampler.output_samples,
        },
        "recording": None if recording is None else str(recording),
    }
    if channels is not None:
        report["channels"] = channels

    return report


def write_report(path: Path, report: dict) -> None:
    """Write the report, creating its parent directory when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
