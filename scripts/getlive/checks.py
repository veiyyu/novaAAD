"""Acceptance rules for one live bring-up run, free of LSL and of hardware.

``live`` collects what happened during a run (per-electrode numbers come from
``electrodes.ChannelStats``, channel faults from the package's own
``QualityMonitor``) and ``evaluate`` turns those facts into a verdict.

Three statuses are used, and only ``fail`` blocks acceptance:

* ``pass`` - the expected behaviour was observed.
* ``warn``  - something an operator should read, e.g. one dead electrode, a
  rejected window, a resampler that buffers for two seconds, or a run stopped
  by Ctrl+C.
* ``fail``  - the streaming path itself did not work.

The numeric limits below are prototype values chosen for this bring-up script,
not amplifier specifications; the datasheet and the recording settings are the
authority for gain, range and impedance.
"""

import math
from dataclasses import dataclass
from typing import Mapping

PASS = "pass"
WARN = "warn"
FAIL = "fail"

# A run that acquires less than this fraction of the requested duration lost
# samples instead of merely stopping early.
FLOW_PASS_RATIO = 0.9
FLOW_WARN_RATIO = 0.5
# Age of the newest consumed sample. The package's own acquire guard stops the
# run at 3 s, so this check reports the margin: below half a second the chain is
# comfortably live, up to the guard it works with little headroom, and at the
# guard it cannot be sustained at all.
LAG_PASS_SECONDS = 0.5
LAG_WARN_SECONDS = 3.0
# Resampler startup delay. The preset name itself is not a problem: what
# matters is how long the chain buffers before its first output. 2 s is the
# package's own default delay budget; beyond the acquire guard's 3 s the run
# cannot stay timely at all.
RESAMPLER_PASS_SECONDS = 2.0
RESAMPLER_FAIL_SECONDS = 3.0


@dataclass(frozen=True)
class Check:
    """One acceptance rule and how the run scored on it."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class RunFacts:
    """Everything ``evaluate`` needs, as plain values.

    Args:
        expected_sfreq: Source rate the operator declared.
        source_sfreq: Source rate the outlet declared and the run connected to.
        expected_channels: Channels the run's contract required.
        source_channels: Channels the outlet published (extras are dropped).
        dropped_channels: Published channels outside the contract.
        window_reasons: Rejection reasons and how many windows carried each.
        warmup_windows: Windows rejected because the chain was still settling;
            they carry no reason and are the normal first windows of a run.
        stats: ``StreamStats.to_dict()`` snapshot of the run.
        resampler_quality: SoXR preset actually used.
        resampler_delay: Measured startup delay of that preset, in seconds.
        out_sfreq: Output rate of the preprocessing chain.
        window_seconds: Window length at the output rate.
        model_sfreq: Sample rate the offline model expects.
        model_window_seconds: Window length the offline model expects.
        flat_channels: Electrodes whose mean peak-to-peak stayed below the
            flatness threshold.
        noisy_channels: Electrodes whose mean peak-to-peak exceeded the noise
            threshold.
        channel_count: Electrodes the statistics covered.
        flat_uv: Mean peak-to-peak below this counted as flat.
        noisy_uv: Mean peak-to-peak above this counted as noisy.
        duration: Seconds the run was asked to stream.
        elapsed: Seconds the run actually streamed.
        interrupted: Whether the operator stopped the run with Ctrl+C.
        failure: Error text when the run raised, else ``None``.
        min_valid_windows: Valid windows required for acceptance.
        cap_profile: Name of the cap profile the run resolved.
        cap_mode: How that profile was chosen (``auto``/``ca-208``/``declared``).
        excluded_channels: Known-dead channels kept out of the fault verdict.
        check_channels: Whether channel faults could reject a window at all.
        max_bad_channels: Faulting channels tolerated per fault type and window.
        bad_channel_windows: Faulting channel -> windows it appeared in.
        held_rows: Rows where an excluded column was held instead of repaired.
        model_channels: Electrodes the offline model contract expects.
        model_missing: Model electrodes this run's contract does not carry.
    """

    expected_sfreq: float
    source_sfreq: float
    expected_channels: int
    source_channels: int
    dropped_channels: tuple[str, ...]
    window_reasons: Mapping[str, int]
    warmup_windows: int
    stats: Mapping[str, float]
    resampler_quality: str
    resampler_delay: float
    out_sfreq: float
    window_seconds: float
    model_sfreq: float
    model_window_seconds: float
    flat_channels: tuple[str, ...]
    noisy_channels: tuple[str, ...]
    channel_count: int
    flat_uv: float
    noisy_uv: float
    duration: float
    elapsed: float
    interrupted: bool
    failure: str | None
    min_valid_windows: int
    cap_profile: str
    cap_mode: str
    excluded_channels: tuple[str, ...]
    check_channels: bool
    max_bad_channels: int
    bad_channel_windows: Mapping[str, int]
    held_rows: int
    model_channels: int
    model_missing: tuple[str, ...]


@dataclass(frozen=True)
class Acceptance:
    """The verdict of one run: every check, plus the yes/no answer."""

    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """Whether no check failed."""

        return all(check.status != FAIL for check in self.checks)

    def summary(self) -> dict:
        """Count the checks per status."""

        return {
            status: sum(1 for check in self.checks if check.status == status)
            for status in (PASS, WARN, FAIL)
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable report."""

        return {
            "ok": self.ok,
            "counts": self.summary(),
            "checks": [
                {"name": check.name, "status": check.status, "detail": check.detail}
                for check in self.checks
            ],
        }

    def format(self) -> str:
        """Render the report as a fixed-width table."""

        width = max(len(check.name) for check in self.checks)
        lines = [
            f"{check.status.upper():<4}  {check.name:<{width}}  {check.detail}"
            for check in self.checks
        ]
        counts = self.summary()
        lines.append(
            f"\nverdict: {'USABLE' if self.ok else 'NOT USABLE'} "
            f"({counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail)"
        )
        return "\n".join(lines)


def _number(value, default: float = 0.0) -> float:
    """Read a numeric counter defensively."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate(facts: RunFacts) -> Acceptance:
    """Score one run against the acceptance rules.

    The rules deliberately separate "the streaming path works" (``fail`` when
    broken) from "the signal on the cap needs attention" (``warn``), because a
    single dead electrode is a gel problem, not a software problem.
    """

    checks: list[Check] = []
    _check_connection(checks, facts)
    _check_transport(checks, facts)
    _check_windows(checks, facts)
    _check_chain(checks, facts)
    _check_channel_scope(checks, facts)
    _check_bad_channels(checks, facts)
    _check_cap(checks, facts)
    _check_geometry(checks, facts)
    _check_model_channels(checks, facts)
    _check_duration(checks, facts)
    return Acceptance(tuple(checks))


def _check_connection(checks: list[Check], facts: RunFacts) -> None:
    """Rules 1-3: the run finished, at the declared rate, with the contract."""

    # 1. Did anything fail at all?
    if facts.failure is None:
        checks.append(Check("connection", PASS, "the run completed without an error"))
    else:
        checks.append(Check("connection", FAIL, f"the run stopped: {facts.failure}"))

    # 2. Source rate, as declared by the outlet.
    if math.isclose(facts.source_sfreq, facts.expected_sfreq, abs_tol=1e-9):
        checks.append(
            Check("source_rate", PASS, f"{facts.source_sfreq:g} Hz as requested")
        )
    else:
        checks.append(
            Check(
                "source_rate",
                FAIL,
                f"outlet publishes {facts.source_sfreq:g} Hz, "
                f"run was configured for {facts.expected_sfreq:g} Hz",
            )
        )

    # 3. Channel contract: every expected electrode present, extras dropped.
    if facts.source_channels < facts.expected_channels:
        checks.append(
            Check(
                "channel_contract",
                FAIL,
                f"outlet publishes {facts.source_channels} channels, "
                f"contract needs {facts.expected_channels}",
            )
        )
    elif facts.dropped_channels:
        checks.append(
            Check(
                "channel_contract",
                WARN,
                f"contract satisfied; dropped {', '.join(facts.dropped_channels)}",
            )
        )
    else:
        checks.append(
            Check(
                "channel_contract",
                PASS,
                f"all {facts.expected_channels} expected channels from a "
                f"{facts.source_channels}-channel outlet",
            )
        )


def _check_transport(checks: list[Check], facts: RunFacts) -> None:
    """Rules 4-6: samples arrived, on time, and not too late to be used."""

    stats = facts.stats
    samples = _number(stats.get("samples"))
    gaps = _number(stats.get("gaps"))
    max_lag = _number(stats.get("max_lag"))

    # 4. Did data actually flow for the requested time?
    wanted = facts.duration * facts.expected_sfreq
    ratio = samples / wanted if wanted > 0 else 0.0
    if ratio >= FLOW_PASS_RATIO:
        checks.append(
            Check(
                "data_flow",
                PASS,
                f"{samples:.0f} samples ({ratio:.0%} of {facts.duration:g}s)",
            )
        )
    elif ratio >= FLOW_WARN_RATIO:
        checks.append(
            Check(
                "data_flow",
                WARN,
                f"only {samples:.0f} samples ({ratio:.0%} of {facts.duration:g}s)",
            )
        )
    else:
        checks.append(
            Check(
                "data_flow",
                FAIL,
                f"{samples:.0f} samples ({ratio:.0%} of {facts.duration:g}s)",
            )
        )

    # 5. Transport jitter: timestamp spacings above one and a half samples.
    if gaps == 0:
        checks.append(Check("timing_gaps", PASS, "no timestamp gap above 1.5 samples"))
    else:
        checks.append(
            Check(
                "timing_gaps",
                WARN,
                f"{gaps:.0f} gap(s), largest {_number(stats.get('max_gap')):.4f}s",
            )
        )

    # 6. Age of consumed data; the package's own guard stops the run at 3 s.
    if max_lag < LAG_PASS_SECONDS:
        checks.append(Check("input_lag", PASS, f"oldest consumed block {max_lag:.3f}s"))
    elif max_lag < LAG_WARN_SECONDS:
        checks.append(Check("input_lag", WARN, f"oldest consumed block {max_lag:.3f}s"))
    else:
        checks.append(Check("input_lag", FAIL, f"oldest consumed block {max_lag:.3f}s"))


def _check_windows(checks: list[Check], facts: RunFacts) -> None:
    """Rules 7-8: usable windows came out, and why the others did not."""

    stats = facts.stats
    windows = _number(stats.get("windows"))
    valid = _number(stats.get("valid"))
    rejected = _number(stats.get("rejected"))

    # 7. Did the chain produce usable windows?
    if valid >= facts.min_valid_windows:
        checks.append(
            Check(
                "windows",
                PASS,
                f"{valid:.0f} valid of {windows:.0f} windows ({rejected:.0f} rejected)",
            )
        )
    elif valid > 0:
        checks.append(
            Check(
                "windows",
                WARN,
                f"only {valid:.0f} valid windows, {facts.min_valid_windows} wanted "
                f"of {windows:.0f}",
            )
        )
    else:
        checks.append(Check("windows", FAIL, f"no valid window out of {windows:.0f}"))

    # 8. Why windows were rejected. A first cap test normally rejects the
    #    warm-up windows and possibly a few noisy ones, so rejections are a
    #    warning to read, not a failure; the counts are what matters.
    if not facts.window_reasons:
        detail = "no window was rejected by a judge"
        if facts.warmup_windows:
            detail += (
                f" ({facts.warmup_windows} warm-up window(s) before the chain "
                "settled)"
            )
        checks.append(Check("quality_reasons", PASS, detail))
    else:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(facts.window_reasons.items())
        )
        extra = (
            f"; {facts.warmup_windows} warm-up window(s)"
            if facts.warmup_windows
            else ""
        )
        checks.append(Check("quality_reasons", WARN, f"rejections: {reasons}{extra}"))


def _check_chain(checks: list[Check], facts: RunFacts) -> None:
    """Rules 9-11: the chain recovered, repaired, and still buffers in time."""

    stats = facts.stats
    recoveries = _number(stats.get("recoveries"))
    repairs = _number(stats.get("repairs"))
    dropped = _number(stats.get("dropped"))

    # 9. The chain restarting itself means the source delivered damage.
    if recoveries == 0:
        checks.append(Check("recovery", PASS, "the chain never had to restart"))
    else:
        checks.append(
            Check("recovery", WARN, f"{recoveries:.0f} bounded recovery restart(s)")
        )

    # 10. Repaired or dropped source rows.
    if repairs == 0 and dropped == 0:
        checks.append(Check("repair", PASS, "no damaged source row"))
    else:
        checks.append(
            Check(
                "repair",
                WARN,
                f"{repairs:.0f} repaired row(s), {dropped:.0f} dropped row(s)",
            )
        )

    # 11. Resampler startup delay: how long the chain buffers before its first
    #     output, whatever preset produced it.
    delay = facts.resampler_delay
    if delay <= RESAMPLER_PASS_SECONDS:
        checks.append(
            Check(
                "resampler",
                PASS,
                f"{facts.resampler_quality} preset, {delay:.2f}s startup delay",
            )
        )
    elif delay <= RESAMPLER_FAIL_SECONDS:
        checks.append(
            Check(
                "resampler",
                WARN,
                f"{facts.resampler_quality} preset, {delay:.2f}s startup delay "
                "before the first window",
            )
        )
    else:
        checks.append(
            Check(
                "resampler",
                FAIL,
                f"{facts.resampler_quality} preset needs {delay:.2f}s before its "
                "first output; the run cannot stay timely",
            )
        )


def _check_channel_scope(checks: list[Check], facts: RunFacts) -> None:
    """Rule 12: how channel faults are judged, before what was seen.

    This is run configuration, so it reports rather than scores: a dry cap that
    excludes its known-dead electrodes is a deliberate choice, and an operator
    who turned the guard off entirely should still read it in the verdict.
    """

    tolerance = (
        f", up to {facts.max_bad_channels} faulting channel(s) tolerated"
        if facts.max_bad_channels
        else ""
    )
    excluded = ", ".join(facts.excluded_channels)
    if not facts.check_channels and facts.excluded_channels:
        checks.append(
            Check(
                "channel_scope",
                WARN,
                f"channel faults never reject a window; excluded: {excluded}",
            )
        )
    elif not facts.check_channels:
        checks.append(
            Check(
                "channel_scope",
                WARN,
                "channel faults never reject a window (--no-channel-check)",
            )
        )
    elif facts.excluded_channels:
        checks.append(
            Check(
                "channel_scope",
                PASS,
                f"excluded from the verdict: {excluded} (still recorded)"
                + tolerance,
            )
        )
    else:
        checks.append(
            Check(
                "channel_scope",
                PASS,
                f"every EEG electrode is judged; no exclusion{tolerance}",
            )
        )


def _check_bad_channels(checks: list[Check], facts: RunFacts) -> None:
    """Rule 13: which electrodes the quality monitor actually caught.

    Evidence, not a verdict: a run that tolerates a dead electrode is still
    usable, but the names belong in the report because they decide the next
    session's exclusion list.
    """

    if not facts.bad_channel_windows and not facts.held_rows:
        checks.append(Check("bad_channels", PASS, "no channel fault during the run"))
        return

    detail = []
    if facts.bad_channel_windows:
        detail.append(
            "faulting: "
            + ", ".join(
                f"{name} x{count} window(s)"
                for name, count in sorted(facts.bad_channel_windows.items())
            )
        )
    if facts.held_rows:
        detail.append(f"{facts.held_rows} row(s) held on excluded channels")
    checks.append(Check("bad_channels", WARN, "; ".join(detail)))


def _check_cap(checks: list[Check], facts: RunFacts) -> None:
    """Rule 14: the electrodes themselves, which is what the operator acts on."""

    # 14. A flat electrode is a contact problem, not a streaming problem,
    #     unless every electrode is flat.
    if not facts.flat_channels and not facts.noisy_channels:
        checks.append(
            Check(
                "electrodes",
                PASS,
                f"all {facts.channel_count} electrodes between "
                f"{facts.flat_uv:g} and {facts.noisy_uv:g} uV mean peak-to-peak",
            )
        )
    elif facts.channel_count and len(facts.flat_channels) == facts.channel_count:
        checks.append(
            Check(
                "electrodes",
                FAIL,
                f"every one of {facts.channel_count} electrodes is flat",
            )
        )
    else:
        detail = []
        if facts.flat_channels:
            detail.append(f"flat: {', '.join(facts.flat_channels)}")
        if facts.noisy_channels:
            detail.append(f"noisy: {', '.join(facts.noisy_channels)}")
        checks.append(Check("electrodes", WARN, "; ".join(detail)))


def _check_geometry(checks: list[Check], facts: RunFacts) -> None:
    """Rule 15: the windows this chain produces match the offline model."""

    if (
        math.isclose(facts.out_sfreq, facts.model_sfreq, rel_tol=0, abs_tol=1e-9)
        and math.isclose(
            facts.window_seconds, facts.model_window_seconds, rel_tol=0, abs_tol=1e-9
        )
    ):
        checks.append(
            Check(
                "model_geometry",
                PASS,
                f"{facts.out_sfreq:g} Hz windows of {facts.window_seconds:g}s match "
                "the offline model",
            )
        )
    else:
        checks.append(
            Check(
                "model_geometry",
                WARN,
                f"run produces {facts.out_sfreq:g} Hz windows of "
                f"{facts.window_seconds:g}s, the offline model expects "
                f"{facts.model_sfreq:g} Hz / {facts.model_window_seconds:g}s",
            )
        )


def _check_model_channels(checks: list[Check], facts: RunFacts) -> None:
    """Rule 16: the contract still feeds every electrode the model expects.

    For the CA-208 profile in ``cap`` mode this is automatic. For another cap,
    or for ``--channels model`` on a montage that shares only some electrode
    names, the model would refuse the windows - so say it here rather than let
    a decoder fail later.
    """

    if not facts.model_missing:
        checks.append(
            Check(
                "model_channels",
                PASS,
                f"all {facts.model_channels} offline model electrodes are in the "
                "contract",
            )
        )
        return

    covered = facts.model_channels - len(facts.model_missing)
    missing = ", ".join(facts.model_missing[:8])
    suffix = "..." if len(facts.model_missing) > 8 else ""
    checks.append(
        Check(
            "model_channels",
            WARN,
            f"{covered}/{facts.model_channels} offline model electrodes in the "
            f"contract; missing {missing}{suffix}",
        )
    )


def _check_duration(checks: list[Check], facts: RunFacts) -> None:
    """Rule 17: the run streamed for as long as it was asked to."""

    # A run stopped by Ctrl+C is the normal way to end a hardware test.
    if facts.interrupted:
        checks.append(
            Check("duration", WARN, f"stopped by Ctrl+C after {facts.elapsed:.1f}s")
        )
    elif facts.elapsed + 1.0 < facts.duration:
        checks.append(
            Check(
                "duration",
                WARN,
                f"streamed {facts.elapsed:.1f}s of the requested {facts.duration:g}s",
            )
        )
    else:
        checks.append(
            Check(
                "duration",
                PASS,
                f"streamed {facts.elapsed:.1f}s of {facts.duration:g}s",
            )
        )
