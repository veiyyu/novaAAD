"""Linear repair of short non-finite runs before causal processing.

A stream occasionally delivers a few broken samples: an isolated ``NaN`` or
``Inf``, or a short missing run. Left alone, one broken sample poisons the
state of every causal filter that follows for its whole settling time. This
module repairs such short runs between two finite endpoints, channel by
channel, before any filter sees them.

``Repair`` is both a chain stage (``stage(data, timestamps) ->
(data, timestamps)``) and a window judge (``reasons(start, end)``): finished
repairs are remembered as time intervals, and windows overlapping one are
flagged ``"interpolated"`` so consumers can reject them. Damage that cannot be
repaired raises :class:`UnrepairableError`; a bounded-recovery guard (see
``nova2026.streaming.recovery``) catches it and decides between restarting the
chain and stopping the run.
"""

import math

import numpy as np


class UnrepairableError(RuntimeError):
    """A chunk cannot be repaired; bounded recovery decides what happens next.

    Args:
        kind: Machine-readable fault name, e.g. ``"nonfinite_run"``,
            ``"irregular_timestamps"``, ``"large_gap"``,
            ``"nonfinite_timestamp"`` or ``"unsafe_endpoints"``.
        gap_seconds: Size of a timestamp gap in seconds, when the fault is a
            gap. Recovery uses the size to tell recoverable gaps apart from
            fatal ones.

    Notes:
        A subclass of ``RuntimeError``, so code written before bounded
        recovery existed still treats it as a run-stopping error.
    """

    def __init__(self, kind: str, *, gap_seconds: float | None = None) -> None:
        """Build the message and keep the structured fault details."""

        message = f"Cannot repair source damage ({kind})."
        if gap_seconds is not None:
            message += f" Timestamp gap of {gap_seconds:.3f}s."
        super().__init__(message)
        self.kind = kind
        self.gap_seconds = gap_seconds


class Repair:
    """Repair short NaN/Inf runs and report which windows they touch.

    Args:
        sfreq: Source sampling rate in Hz.
        max_seconds: Longest repairable damage; longer runs stop the run.
            Defaults to 0.02 s (10 samples at 500 Hz).
        tolerance_seconds: Permitted timestamp jitter around the nominal grid.
        settle_seconds: Extra interval after a repair that still flags
            windows (filter history stays suspect).
        source_unit_exponent: Power of ten of the source unit, used only to
            convert endpoints to uV for the safety check; ``None`` disables
            that check.
        n_eeg: Number of leading EEG columns the safety check inspects;
            defaults to all columns when ``None``.
        amplitude_limit_uv: Largest finite-to-finite jump still safe to bridge.
        saturation_limit_uv: Absolute endpoint level that makes a repair unsafe.

    Notes:
        A row is damaged when any of its values is non-finite. Short missing
        runs are detected from timestamp gaps, synthesised as damaged rows and
        repaired the same way. Repair is linear, per channel, and never edits
        healthy values. Damage that cannot be repaired (a run longer than
        ``max_seconds``, unsafe endpoints, irregular or non-finite timestamps,
        a gap beyond the repair limit) raises :class:`UnrepairableError` with
        a structured kind; a bounded-recovery guard decides whether the run
        restarts or stops.

    Attributes:
        repaired_samples: Rows repaired so far (dropped rows are excluded).
        dropped_rows: Damaged rows before the first finite sample, dropped.
    """

    def __init__(
        self,
        sfreq: float,
        *,
        max_seconds: float = 0.02,
        tolerance_seconds: float = 2e-4,
        settle_seconds: float = 0.0,
        source_unit_exponent: int | None = None,
        n_eeg: int | None = None,
        amplitude_limit_uv: float = 500.0,
        saturation_limit_uv: float = 75000.0,
    ) -> None:
        """Validate limits and start empty."""

        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be finite and positive.")
        for name, value in (
            ("max_seconds", max_seconds),
            ("tolerance_seconds", tolerance_seconds),
            ("settle_seconds", settle_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if max_seconds > 0.1:
            raise ValueError("max_seconds is capped at 0.1 s.")
        if tolerance_seconds >= 0.5 / sfreq:
            raise ValueError("tolerance_seconds must be less than half a sample.")
        if source_unit_exponent is not None and source_unit_exponent not in (0, -3, -6, -9):
            raise ValueError("Supported source exponents are 0, -3, -6, -9.")
        if not math.isfinite(amplitude_limit_uv) or amplitude_limit_uv < 0:
            raise ValueError("amplitude_limit_uv must be finite and non-negative.")
        if not math.isfinite(saturation_limit_uv) or saturation_limit_uv < amplitude_limit_uv:
            raise ValueError("saturation_limit_uv must not be below the amplitude limit.")

        self._sfreq = float(sfreq)
        self._interval = 1.0 / self._sfreq
        self._limit = int(math.floor(max_seconds * self._sfreq))
        self._tol_steps = tolerance_seconds * self._sfreq
        self._settle = float(settle_seconds)
        # Raw units -> uV for the endpoint safety check, when units are known.
        if source_unit_exponent is None:
            self._to_uv = None
            self._guard_columns = None
        else:
            self._to_uv = 10.0 ** (source_unit_exponent + 6)
            self._guard_columns = None if n_eeg is None else int(n_eeg)
            self._amplitude_uv = float(amplitude_limit_uv)
            self._saturation_uv = float(saturation_limit_uv)
        self.reset()

    def reset(self) -> None:
        """Discard pending rows, anchors and repair history."""

        self._left: np.ndarray | None = None   # last finite row seen
        self._left_time: float | None = None
        self._previous_time: float | None = None
        self._pending: list[tuple[np.ndarray, float]] = []
        self._intervals: list[tuple[float, float]] = []
        self.repaired_samples = 0
        self.dropped_rows = 0

    def __call__(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Repair this chunk and return everything that could be settled.

        A damaged tail without a finite right endpoint is retained and
        returned with the next call, so the output may be shorter than the
        input (or empty) while damage is pending.
        """

        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")
        timestamps = np.asarray(timestamps)
        if timestamps.ndim != 1 or len(timestamps) != len(data):
            raise ValueError("Each sample must have one timestamp.")
        if len(data) == 0:
            return data, timestamps
        channels = data.shape[1]

        emitted_rows: list[np.ndarray] = []
        emitted_times: list[float] = []

        for row, stamp in zip(data, timestamps):
            stamp = float(stamp)
            # Time is never repaired: a broken clock means a broken grid.
            if not math.isfinite(stamp):
                raise UnrepairableError("nonfinite_timestamp")
            self._check_grid(stamp, channels)

            if not np.all(np.isfinite(row)):
                if self._left is None:
                    # Damage before the first finite sample: the run simply
                    # starts later, so drop the rows instead of repairing.
                    self.dropped_rows += 1
                    continue
                self._pending.append((np.array(row, copy=True), stamp))
                if len(self._pending) > self._limit:
                    raise UnrepairableError("nonfinite_run")
                continue

            # A finite row is the right endpoint of any pending damage.
            if self._pending:
                self._resolve(
                    row, stamp, emitted_rows, emitted_times
                )
            self._left = np.array(row, copy=True)
            self._left_time = stamp
            emitted_rows.append(row)
            emitted_times.append(stamp)

        if emitted_rows:
            return np.stack(emitted_rows), np.asarray(emitted_times)
        return np.empty((0, channels)), np.empty((0,))

    def _check_grid(self, stamp: float, channels: int) -> None:
        """Verify the sample is on the nominal grid; synthesise small gaps."""

        if self._previous_time is None:
            self._previous_time = stamp
            return
        previous = self._previous_time
        steps = (stamp - previous) * self._sfreq
        self._previous_time = stamp

        if abs(steps - 1.0) <= self._tol_steps:
            return
        if steps <= 1.0:
            raise UnrepairableError("irregular_timestamps")
        count = int(round(steps))
        if abs(steps - count) > self._tol_steps:
            raise UnrepairableError("irregular_timestamps")
        if count - 1 > self._limit:
            raise UnrepairableError(
                "large_gap", gap_seconds=float(stamp - previous)
            )
        # Missing samples between the previous row and this one.
        for index in range(1, count):
            time = previous + index * self._interval
            if self._left is None:
                # No anchor yet: nothing before the first finite row matters.
                self.dropped_rows += 1
                continue
            self._pending.append((np.full(channels, np.nan), time))

    def _resolve(
        self,
        right: np.ndarray,
        right_time: float,
        emitted_rows: list,
        emitted_times: list,
    ) -> None:
        """Interpolate the pending damage using the finite left/right rows."""

        pending = self._pending
        if len(pending) > self._limit:
            raise UnrepairableError("nonfinite_run")
        if self._left is None or self._left_time is None:
            raise RuntimeError("No finite left endpoint for the repair.")
        if self._unsafe(self._left, right):
            raise UnrepairableError("unsafe_endpoints")

        span = right_time - self._left_time
        first_time = pending[0][1]
        last_time = pending[-1][1]
        for damaged, time in pending:
            if time <= self._left_time or time >= right_time or span <= 0:
                raise RuntimeError("Pending damage is outside its repair span.")
            # Linear per-channel estimate between the two finite endpoints;
            # healthy values inside a damaged row are never rewritten.
            weight = (time - self._left_time) / span
            estimate = self._left + weight * (right - self._left)
            repaired = np.where(np.isfinite(damaged), damaged, estimate)
            emitted_rows.append(repaired)
            emitted_times.append(time)

        # Remember the span so windows overlapping it can be flagged.
        self._intervals.append((first_time, last_time + self._settle))
        self.repaired_samples += len(pending)
        self._pending = []

    def _unsafe(self, left: np.ndarray, right: np.ndarray) -> bool:
        """Whether the two endpoints are too extreme to interpolate between."""

        if self._to_uv is None:
            return False  # units unknown: only value repair is possible
        columns = self._guard_columns
        endpoints = np.stack((left, right))
        if columns is not None:
            endpoints = endpoints[:, :columns]
        uv = endpoints * self._to_uv
        jump = np.abs(uv[1] - uv[0])
        return bool(
            np.any(np.abs(uv) >= self._saturation_uv)
            or np.any(jump > self._amplitude_uv)
        )

    def reasons(self, start: float, end: float) -> tuple[str, ...]:
        """Return ``("interpolated",)`` when a repair touches [start, end]."""

        kept = []
        flagged = False
        for left, right in self._intervals:
            if right < start:
                continue  # expired: drop while iterating
            kept.append((left, right))
            if left <= end and right >= start:
                flagged = True
        self._intervals = kept
        return ("interpolated",) if flagged else ()
