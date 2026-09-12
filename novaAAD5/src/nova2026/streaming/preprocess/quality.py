"""Raw EEG quality monitoring, independent of filtering and threading."""

import math

import numpy as np


class QualityMonitor:
    """Observe raw EEG faults and report which windows they overlap.

    Args:
        n_eeg: Number of EEG columns at the front of the samples-by-channels
            data. Auxiliary channels (e.g. EOG) are not checked.
        sfreq: Sample rate of the fed data, for threshold durations.
        amplitude_limit_uv: Excursion from the run's initial level that flags
            ``"amplitude"``.
        saturation_limit_uv: Absolute level that flags ``"saturation"``.
        flatline_seconds: Near-flat run length that flags ``"flatline"``.
        flatline_tolerance_uv: Maximum per-sample change treated as unchanged.
        warmup_seconds: Recovery settling appended to each fault interval.

    Notes:
        Feed data in the same units the thresholds are expressed in (uV). Fault
        intervals are recorded in timestamp space, so results do not depend on
        chunk boundaries. Non-finite rows are not flagged and are documented as
        the Repair stage's responsibility: it fixes short non-finite runs
        upstream, and Recovery decides what to do with the rest.
    """

    def __init__(
        self,
        *,
        n_eeg: int,
        sfreq: float,
        amplitude_limit_uv: float = 500.0,
        saturation_limit_uv: float = 75000.0,
        flatline_seconds: float = 0.5,
        flatline_tolerance_uv: float = 0.001,
        warmup_seconds: float = 2.0,
    ) -> None:
        """Validate limits and start empty."""

        if isinstance(n_eeg, bool) or not isinstance(n_eeg, int) or n_eeg < 1:
            raise ValueError("n_eeg must be a positive integer.")
        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be finite and positive.")
        # Every limit must be finite and non-negative.
        for name in (
            "amplitude_limit_uv",
            "saturation_limit_uv",
            "flatline_seconds",
            "flatline_tolerance_uv",
            "warmup_seconds",
        ):
            value = locals()[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        # Saturation is the harder rail; it must sit above the amplitude one.
        if saturation_limit_uv < amplitude_limit_uv:
            raise ValueError("saturation_limit_uv must not be below the amplitude limit.")

        self._n_eeg = n_eeg
        self._sfreq = float(sfreq)
        self._amplitude_limit = float(amplitude_limit_uv)
        self._saturation_limit = float(saturation_limit_uv)
        self._flatline_samples = round(flatline_seconds * sfreq)
        self._flatline_tolerance = float(flatline_tolerance_uv)
        self._warmup = float(warmup_seconds)
        self.reset()

    def reset(self) -> None:
        """Clear fault history and per-channel state at a segment boundary."""

        # Fault intervals live in timestamp space: (start, end, reason).
        self._intervals: list[tuple[float, float, str]] = []
        self._previous: np.ndarray | None = None
        self._initial: np.ndarray | None = None
        self._unchanged = np.zeros(self._n_eeg, dtype=int)

    def feed(self, data_uv: np.ndarray, timestamps: np.ndarray) -> None:
        """Inspect one raw chunk and remember invalid time intervals."""

        data_uv = np.asarray(data_uv)
        # EEG columns come first; auxiliary channels are ignored.
        if data_uv.ndim != 2 or data_uv.shape[1] < self._n_eeg:
            raise ValueError("Expected samples by channels with EEG columns first.")
        if timestamps.ndim != 1 or len(timestamps) != len(data_uv):
            raise ValueError("Each sample must have one timestamp.")
        if data_uv.shape[0] == 0:
            return

        eeg = data_uv[:, : self._n_eeg]
        # Non-finite rows are never faults here: Repair (upstream) fixes short
        # runs, and Recovery decides what to do with the rest.
        finite = np.isfinite(eeg).all(axis=1) & np.isfinite(timestamps)

        if self._initial is None:
            # Anchor excursions at the first finite row so DC offsets do not
            # look like amplitude faults.
            if not np.any(finite):
                return
            anchor = int(np.flatnonzero(finite)[0])
            self._initial = eeg[anchor].copy()

        # Fault 1+2: excursion from the initial level and absolute rails.
        excursion = np.abs(eeg - self._initial)
        faults = {
            "amplitude": np.any(excursion > self._amplitude_limit, axis=1),
            "saturation": np.any(np.abs(eeg) >= self._saturation_limit, axis=1),
        }

        # Fault 3: per-channel "unchanged" run, carried across chunk boundaries.
        previous = eeg[0] if self._previous is None else self._previous
        change = np.abs(np.diff(eeg, axis=0, prepend=previous[None, :]))
        changed = change > self._flatline_tolerance
        indices = np.arange(1, len(eeg) + 1)[:, None]
        last_change = np.where(
            changed, indices, -self._unchanged[None, :]
        )
        last_change = np.maximum.accumulate(last_change, axis=0)
        run_lengths = indices - last_change
        self._unchanged = run_lengths[-1].copy()
        self._previous = eeg[-1].copy()
        faults["flatline"] = np.any(
            run_lengths >= self._flatline_samples, axis=1
        )

        # Turn boolean masks into timestamp intervals so the result is
        # independent of how the stream happened to be chunked.
        for name, mask in faults.items():
            mask = mask & finite
            edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            for left, right in zip(starts, stops):
                start = float(timestamps[left])
                if name == "flatline":
                    start -= self._flatline_samples / self._sfreq
                end = float(timestamps[right - 1]) + self._warmup
                self._intervals.append((start, end, name))

    def reasons(self, start: float, end: float) -> tuple[str, ...]:
        """Return fault names overlapping [start, end] and prune history."""

        kept = []
        reasons = set()
        for left, right, name in self._intervals:
            if right < start:
                continue  # expired: drop it while iterating
            kept.append((left, right, name))
            if left <= end and right >= start:
                reasons.add(name)
        self._intervals = kept

        return tuple(sorted(reasons))
