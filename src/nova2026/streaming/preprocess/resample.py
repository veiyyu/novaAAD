"""Streaming rate conversion stage backed by SoXR.

The module imports cleanly without ``soxr``; only constructing the stage
requires it. If it is missing, construction raises ``ImportError`` with the
install command.

The quality preset is a trade-off between how cleanly the signal is converted
and how long the resampler buffers before its first output. That delay is an
internal *sample count*, so the same preset means very different delays at
different rates (about 1.9 s for 500 -> 128 Hz but about 8 s for 128 -> 64 Hz
on the build measured here). ``Resampler`` therefore does not trust a
hard-coded table: when no preset is given it measures each candidate and picks
the cleanest one that fits the caller's delay budget.
"""

import math
import warnings

import numpy as np

# Quality/speed trade-off presets understood by python-soxr.
# "QQ" is deliberately absent: it performs essentially no anti-aliasing. This
# path band-limits with a third-order 1-45 Hz band-pass and then resamples
# 500 -> 128 Hz, so the output Nyquist (64 Hz) sits close to the filter's
# stopband edge and folded energy would not be attenuated enough. The other
# presets keep a proper anti-aliasing filter; LQ has the least latency of them.
SOXR_QUALITIES = ("LQ", "MQ", "HQ", "VHQ")
# Anti-aliased presets ordered from strongest filtering to weakest.
QUALITY_ORDER = ("VHQ", "HQ", "MQ", "LQ")
# Delay budget used when the caller states no age requirement.
DEFAULT_DELAY_BUDGET_SECONDS = 2.0


class ResamplerQualityWarning(UserWarning):
    """The selected preset is low quality, or cannot meet the delay budget."""


def _measure_startup_delay(
    in_sfreq: float,
    out_sfreq: float,
    quality: str,
    *,
    max_samples: int = 100_000,
) -> float:
    """Measure how long a preset buffers before its first output, in seconds.

    Feeds one zero sample at a time until the first output row appears. The
    result depends on the installed SoXR build, which is exactly why the
    selection below measures instead of reading a table.
    """

    import soxr  # lazy: keeps this module importable without soxr

    stream = soxr.ResampleStream(
        in_rate=float(in_sfreq),
        out_rate=float(out_sfreq),
        num_channels=1,
        dtype="float64",
        quality=quality,
    )
    sample = np.zeros((1, 1), dtype=np.float64)
    fed = 0
    while fed < max_samples:
        output = stream.resample_chunk(sample, last=False)
        fed += 1
        if len(output):
            return fed / float(in_sfreq)
    raise RuntimeError(
        f"The {quality!r} preset produced no output within {max_samples} samples."
    )


def select_quality(
    in_sfreq: float,
    out_sfreq: float,
    *,
    max_delay_seconds: float = DEFAULT_DELAY_BUDGET_SECONDS,
    allow_qq: bool = False,
) -> tuple[str, float]:
    """Return ``(quality, startup_delay_seconds)`` for the given delay budget.

    Walks the anti-aliased presets from strongest to weakest and returns the
    first whose *measured* startup delay fits the budget, so the result is the
    cleanest conversion that still arrives in time. When nothing fits, the
    fastest clean preset (LQ) is returned with its delay, or ``QQ`` when it
    fits and ``allow_qq`` is set.

    Raises:
        ValueError: If the budget is not finite and positive.
    """

    if not math.isfinite(max_delay_seconds) or max_delay_seconds <= 0:
        raise ValueError("max_delay_seconds must be finite and positive.")

    fastest: tuple[str, float] | None = None
    for quality in QUALITY_ORDER:
        delay = _measure_startup_delay(in_sfreq, out_sfreq, quality)
        if fastest is None or delay < fastest[1]:
            fastest = (quality, delay)
        if delay <= max_delay_seconds:
            return quality, delay

    if allow_qq:
        qq_delay = _measure_startup_delay(in_sfreq, out_sfreq, "QQ")
        if qq_delay <= max_delay_seconds:
            return "QQ", qq_delay

    assert fastest is not None  # QUALITY_ORDER is never empty
    return fastest


class Resampler:
    """Resample consecutive chunks from ``in_sfreq`` to ``out_sfreq``.

    Args:
        in_sfreq: Source sample rate in Hz.
        out_sfreq: Target sample rate in Hz.
        n_channels: Expected number of data columns.
        quality: SoXR preset, ``None``/``"auto"`` to choose one automatically,
            or one of ``SOXR_QUALITIES``. ``"QQ"`` is accepted only together
            with ``allow_qq=True``.
        max_age_seconds: How old a sample may be when the consumer uses it
            (e.g. a controller's expiry). The delay budget is derived from it.
        reserve_seconds: Part of ``max_age_seconds`` reserved for everything
            else that eats time (filling the window, analysis, queueing,
            safety margin).
        max_delay_seconds: Explicit delay budget that overrides the derivation
            above.
        allow_qq: Permit the ``QQ`` preset when nothing anti-aliased fits.
            Only safe when an earlier stage already band-limits the signal.
        strict: Raise instead of warning when the budget cannot be met.

    Notes:
        Budget derivation::

            budget = max_delay_seconds
            budget = max_age_seconds - reserve_seconds   # when no override
            budget = DEFAULT_DELAY_BUDGET_SECONDS        # when neither is set

        With ``quality=None`` the class measures each anti-aliased preset and
        picks the strongest one that fits the budget; it warns when the pick
        is the weakest preset (LQ) or when no preset can meet the budget at
        all. Delays are measured on the installed SoXR build, because the same
        preset buffers a different number of samples across builds.

        The output row count per call varies (SoXR buffers internally), which
        is normal for a streaming resampler. Output timestamps form a uniform
        grid at ``out_sfreq`` anchored to the first finite input timestamp;
        SoXR's internal processing latency is not compensated, matching the
        dataproc prototype. Call ``reset()`` before a new segment.

    Attributes:
        quality: The preset actually used.
        startup_delay_seconds: Measured delay before the first output.
        max_delay_seconds: The delay budget the selection had to respect.

    Raises:
        ImportError: If ``soxr`` is not installed.
    """

    def __init__(
        self,
        in_sfreq: float,
        out_sfreq: float,
        n_channels: int,
        quality: str | None = None,
        *,
        max_age_seconds: float | None = None,
        reserve_seconds: float = 0.0,
        max_delay_seconds: float | None = None,
        allow_qq: bool = False,
        strict: bool = False,
    ) -> None:
        """Validate geometry, derive the budget and create the SoXR stream."""

        if not math.isfinite(in_sfreq) or in_sfreq <= 0:
            raise ValueError("in_sfreq must be finite and positive.")
        if not math.isfinite(out_sfreq) or out_sfreq <= 0:
            raise ValueError("out_sfreq must be finite and positive.")
        if math.isclose(in_sfreq, out_sfreq):
            raise ValueError("in_sfreq and out_sfreq must differ.")
        if (
            isinstance(n_channels, bool)
            or not isinstance(n_channels, int)
            or n_channels < 1
        ):
            raise ValueError("n_channels must be a positive integer.")

        # Option validation before any measurement work.
        if not isinstance(allow_qq, bool):
            raise TypeError("allow_qq must be a boolean.")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("reserve_seconds must be finite and non-negative.")
        for name, value in (
            ("max_age_seconds", max_age_seconds),
            ("max_delay_seconds", max_delay_seconds),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when given.")

        if quality == "auto":
            quality = None
        if quality is not None:
            if quality == "QQ":
                if not allow_qq:
                    raise ValueError(
                        "The QQ preset is disabled here: it performs essentially "
                        "no anti-aliasing, and this path's anti-alias margin is "
                        "thin. Pass allow_qq=True only when an earlier stage "
                        "already band-limits the signal."
                    )
            elif quality not in SOXR_QUALITIES:
                raise ValueError(
                    f"quality must be one of {SOXR_QUALITIES}, 'QQ' (with "
                    "allow_qq=True), None or 'auto'."
                )

        # Optional dependency: import lazily so the rest of the package works
        # without soxr installed.
        try:
            import soxr
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "The Resampler requires 'soxr'. Install it with: "
                "uv add soxr   (or: pip install soxr)"
            ) from error

        # Delay budget: explicit override, else derived from the allowed age.
        if max_delay_seconds is not None:
            budget = float(max_delay_seconds)
        elif max_age_seconds is not None:
            budget = float(max_age_seconds) - float(reserve_seconds)
            if budget <= 0:
                raise ValueError(
                    "max_age_seconds minus reserve_seconds must stay positive: "
                    "the reserved time already exceeds the allowed age, so no "
                    "resampler preset can help."
                )
        else:
            budget = DEFAULT_DELAY_BUDGET_SECONDS

        if quality is None:
            chosen, delay = select_quality(
                in_sfreq, out_sfreq, max_delay_seconds=budget, allow_qq=allow_qq
            )
            self._report_selection(
                chosen, delay, budget, allow_qq, in_sfreq, out_sfreq, strict
            )
        else:
            chosen = quality
            delay = _measure_startup_delay(in_sfreq, out_sfreq, chosen)
            if delay > budget:
                message = (
                    f"The explicitly selected {chosen!r} preset needs "
                    f"{delay:.3f}s of startup delay, more than the "
                    f"{budget:.3f}s budget at {in_sfreq:g}->{out_sfreq:g} Hz."
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, ResamplerQualityWarning, stacklevel=2)

        self._in_sfreq = float(in_sfreq)
        self._out_sfreq = float(out_sfreq)
        self._channels = n_channels
        # Public rates so a session can derive its output geometry.
        self.in_sfreq = self._in_sfreq
        self.out_sfreq = self._out_sfreq
        # Selection results, so scripts and logs can see what was chosen.
        self.quality = chosen
        self.startup_delay_seconds = delay
        self.max_delay_seconds = budget
        # ResampleStream keeps its own history between resample_chunk calls.
        self._resampler = soxr.ResampleStream(
            in_rate=self._in_sfreq,
            out_rate=self._out_sfreq,
            num_channels=n_channels,
            dtype="float64",
            quality=chosen,
        )
        self.reset()

    @staticmethod
    def _report_selection(
        chosen: str,
        delay: float,
        budget: float,
        allow_qq: bool,
        in_sfreq: float,
        out_sfreq: float,
        strict: bool,
    ) -> None:
        """Warn (or raise) about an automatic selection that had to compromise."""

        if delay > budget:
            hint = ""
            if not allow_qq:
                try:
                    qq_delay = _measure_startup_delay(in_sfreq, out_sfreq, "QQ")
                except Exception:  # pragma: no cover - build without QQ
                    qq_delay = None
                if qq_delay is not None and qq_delay <= budget:
                    hint = (
                        f" QQ would meet it (~{qq_delay:.3f}s) but is disabled; "
                        "pass allow_qq=True only if an earlier stage already "
                        "band-limits the signal."
                    )
            message = (
                f"No SoXR preset meets the {budget:.3f}s delay budget at "
                f"{in_sfreq:g}->{out_sfreq:g} Hz; the fastest clean preset "
                f"({chosen}) needs {delay:.3f}s.{hint}"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, ResamplerQualityWarning, stacklevel=3)
        elif chosen == "QQ":
            warnings.warn(
                "The QQ preset performs essentially no anti-aliasing; it is only "
                "safe when an earlier stage already band-limits the signal.",
                ResamplerQualityWarning,
                stacklevel=3,
            )
        elif chosen == "LQ":
            warnings.warn(
                f"The delay budget only allows LQ, the lowest-quality "
                f"anti-aliased preset (startup delay {delay:.3f}s). Raise the "
                f"budget for a cleaner conversion.",
                ResamplerQualityWarning,
                stacklevel=3,
            )

    def __call__(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample one chunk and return it with a new output-rate time grid."""

        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self._channels:
            raise ValueError(
                f"Expected samples by {self._channels} channels, "
                f"received shape {data.shape}."
            )
        if timestamps.ndim != 1 or len(timestamps) != len(data):
            raise ValueError("Each sample must have one timestamp.")
        if data.shape[0] == 0:
            return (
                np.empty((0, self._channels), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        # Remember the first real time so output samples get a time grid.
        if self._anchor is None:
            finite = timestamps[np.isfinite(timestamps)]
            if finite.size:
                self._anchor = float(finite[0])

        # SoXR needs contiguous float64 input; may return 0 rows on early calls.
        output = self._resampler.resample_chunk(
            np.ascontiguousarray(data, dtype=np.float64),
            last=False,
        )
        output = np.asarray(output, dtype=np.float64)

        # Timestamps follow a uniform grid at the output rate.
        if self._anchor is None:
            output_times = np.full(len(output), np.nan)
        else:
            output_times = self._anchor + (
                self._output_index + np.arange(len(output))
            ) / self._out_sfreq
        self._output_index += len(output)
        self.output_samples += len(output)

        return output, output_times

    def reset(self) -> None:
        """Clear internal history without flushing an end-of-stream tail."""

        self._resampler.clear()
        self._anchor: float | None = None
        self._output_index = 0
        self.output_samples = 0
