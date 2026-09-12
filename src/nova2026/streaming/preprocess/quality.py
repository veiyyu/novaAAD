"""Raw EEG quality monitoring, independent of filtering and threading.

Two layers live here. :class:`QualityMonitor` is the *evidence* layer: it
detects channel faults, keeps them in timestamp space and answers which
channels were involved. :class:`BadChannelJudge` is the *plugin* layer: a
judge that reports bad channels without touching the verdict unless it is
explicitly asked to, so a script can watch a dry cap without letting one dead
electrode stop the run.
"""

import math

import numpy as np

from .scope import ChannelScope


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
        channel_names: Optional EEG labels in column order. They are how bad
            channels are reported; without them a channel is reported as
            ``"ch0"``, ``"ch1"``, ... after its column index.
        check_channels: When ``False``, channel faults never reject a window.
            They are still detected and still reported by
            :meth:`bad_channels`, so an electrode that is known to be dead can
            be watched without being allowed to stop the run. The default
            ``True`` keeps the historical verdict.
        max_bad_channels: How many EEG channels may carry a fault inside one
            window before that fault type rejects it. ``0`` (the default)
            rejects as soon as one channel faults, which is the historical
            behaviour. Must be below ``n_eeg``: tolerating every channel would
            disable the guard silently, so say that with
            ``check_channels=False`` instead.
        exclude_channels: Labels of channels known to be dead (for example the
            dry electrodes a cap session was labelled with). They are still
            detected and reported, but they do not count towards
            ``max_bad_channels`` and never reject a window. Requires
            ``channel_names``; an unknown label raises instead of silently
            excluding nothing.

    Notes:
        Feed data in the same units the thresholds are expressed in (uV). Fault
        intervals are recorded in timestamp space, so results do not depend on
        chunk boundaries. Non-finite rows are not flagged and are documented as
        the Repair stage's responsibility: it fixes short non-finite runs
        upstream, and Recovery decides what to do with the rest.

        Channel identity is resolved *before* the per-sample masks are
        collapsed across channels, so a window verdict can name the offending
        electrodes. Evidence is never filtered: :meth:`bad_channels` and
        :meth:`fault_channels` report every faulting channel, while
        ``check_channels``, ``max_bad_channels`` and ``exclude_channels``
        only decide whether the window is rejected.

        Queries prune expired intervals, so ask about windows in
        non-decreasing ``start`` order, as the session's window loop does.
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
        channel_names: tuple[str, ...] | None = None,
        check_channels: bool = True,
        max_bad_channels: int = 0,
        exclude_channels: tuple[str, ...] = (),
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
        if not isinstance(check_channels, bool):
            raise TypeError("check_channels must be a bool.")
        if (
            isinstance(max_bad_channels, bool)
            or not isinstance(max_bad_channels, int)
            or max_bad_channels < 0
        ):
            raise ValueError("max_bad_channels must be a non-negative integer.")
        if max_bad_channels >= n_eeg:
            raise ValueError(
                "max_bad_channels must be below n_eeg; use check_channels=False "
                "to stop checking channels on purpose."
            )
        if isinstance(channel_names, str):
            raise TypeError("channel_names must be an iterable of labels, not a string.")
        if channel_names is not None:
            channel_names = tuple(str(name) for name in channel_names)
            if len(channel_names) != n_eeg:
                raise ValueError("channel_names must name every EEG column.")

        self._n_eeg = n_eeg
        self._sfreq = float(sfreq)
        self._amplitude_limit = float(amplitude_limit_uv)
        self._saturation_limit = float(saturation_limit_uv)
        self._flatline_samples = round(flatline_seconds * sfreq)
        self._flatline_tolerance = float(flatline_tolerance_uv)
        self._warmup = float(warmup_seconds)
        # The scope owns label validation for both this monitor and Repair.
        self.scope = ChannelScope(channel_names, exclude_channels)
        self.channel_names = self.scope.channel_names
        self.exclude_channels = self.scope.exclude_channels
        self.check_channels = bool(check_channels)
        self.max_bad_channels = int(max_bad_channels)
        self._excluded = self.scope.excluded_indices
        self.reset()

    def reset(self) -> None:
        """Clear fault history and per-channel state at a segment boundary."""

        # Fault intervals live in timestamp space:
        # (start, end, reason, channels) with channels as column indices.
        self._intervals: list[tuple[float, float, str, tuple[int, ...]]] = []
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
            "amplitude": excursion > self._amplitude_limit,
            "saturation": np.abs(eeg) >= self._saturation_limit,
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
        faults["flatline"] = run_lengths >= self._flatline_samples

        # Turn per-channel masks into timestamp intervals so the result is
        # independent of how the stream happened to be chunked. The row split
        # stays a plain "any channel faulted" so interval boundaries (and the
        # settling they carry) do not move; the channel set rides along.
        for name, mask in faults.items():
            mask = mask & finite[:, None]
            rows = mask.any(axis=1)
            edges = np.diff(np.concatenate(([False], rows, [False])).astype(int))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            for left, right in zip(starts, stops):
                channels = tuple(
                    int(index)
                    for index in np.flatnonzero(mask[left:right].any(axis=0))
                )
                start = float(timestamps[left])
                if name == "flatline":
                    start -= self._flatline_samples / self._sfreq
                end = float(timestamps[right - 1]) + self._warmup
                self._intervals.append((start, end, name, channels))

    def _label(self, index: int) -> str:
        """Human-readable identity of one EEG column."""

        if self.channel_names is None:
            return f"ch{index}"
        return self.channel_names[index]

    def _overlapping(self, start: float, end: float) -> dict[str, set[int]]:
        """Collect channels per fault name over [start, end]; prune history."""

        kept = []
        found: dict[str, set[int]] = {}
        for left, right, name, channels in self._intervals:
            if right < start:
                continue  # expired: drop it while iterating
            kept.append((left, right, name, channels))
            if left <= end and right >= start:
                found.setdefault(name, set()).update(channels)
        self._intervals = kept

        return found

    def reasons(self, start: float, end: float) -> tuple[str, ...]:
        """Return fault names that reject a window over [start, end].

        The channel policy is applied here and only here: a fault name is
        returned when more than ``max_bad_channels`` non-excluded channels
        carry it inside the window. With ``check_channels=False`` no channel
        fault rejects, which is how a dry cap keeps streaming.
        """

        if not self.check_channels:
            self._overlapping(start, end)  # still prune expired history
            return ()
        reasons = set()
        for name, channels in self._overlapping(start, end).items():
            if len(channels - self._excluded) > self.max_bad_channels:
                reasons.add(name)

        return tuple(sorted(reasons))

    def bad_channels(self, start: float, end: float) -> tuple[str, ...]:
        """Return every channel faulting inside [start, end], policy aside.

        Known-dead channels (``exclude_channels``) are included: the list is
        evidence for the run record, not a verdict. Use :meth:`fault_channels`
        when the fault type behind each channel matters.
        """

        channels: set[int] = set()
        for found in self._overlapping(start, end).values():
            channels |= found

        return tuple(self._label(index) for index in sorted(channels))

    def fault_channels(self, start: float, end: float) -> dict[str, tuple[str, ...]]:
        """Return ``{fault name: channels}`` over [start, end], policy aside."""

        return {
            name: tuple(self._label(index) for index in sorted(channels))
            for name, channels in sorted(self._overlapping(start, end).items())
        }


class BadChannelJudge:
    """Report bad channels as a window judge; never reject unless asked.

    A live dry cap has electrodes that are dead for the whole session. The
    default :class:`QualityMonitor` policy stops the run when they persist, so
    the useful default here is the opposite: watch and report. Pass
    ``block_on_bad_channels=True`` to reject a window whose bad-channel census
    reaches ``min_channels``, which is the count across *all* fault types (the
    monitor's own ``max_bad_channels`` is per type).

    Args:
        monitor: The :class:`QualityMonitor` that owns the evidence.
        block_on_bad_channels: When True, a window with at least
            ``min_channels`` bad channels is rejected with the ``"bad_channels"``
            reason. Default False: the judge only reports.
        min_channels: Bad-channel count that rejects a window when
            ``block_on_bad_channels`` is set. Must be positive.
        reason: Reason name used when blocking, so consumers can tell a
            census rejection from the monitor's own fault names.

    Notes:
        The judge is a plain object with ``reasons`` and ``bad_channels``, so
        it can be handed to ``StreamSession(judges=...)`` or to
        :class:`~nova2026.auditory.streaming.AuditoryProcessor`. It shares one
        monitor, so queries must go through the same non-decreasing window
        order the monitor documents.
    """

    def __init__(
        self,
        monitor: QualityMonitor,
        *,
        block_on_bad_channels: bool = False,
        min_channels: int = 1,
        reason: str = "bad_channels",
    ) -> None:
        """Validate the policy and keep the monitor."""

        if not isinstance(monitor, QualityMonitor):
            raise TypeError("monitor must be a QualityMonitor.")
        if not isinstance(block_on_bad_channels, bool):
            raise TypeError("block_on_bad_channels must be a bool.")
        if isinstance(min_channels, bool) or not isinstance(min_channels, int):
            raise ValueError("min_channels must be a positive integer.")
        if min_channels < 1:
            raise ValueError("min_channels must be a positive integer.")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string.")
        self.monitor = monitor
        self.block_on_bad_channels = bool(block_on_bad_channels)
        self.min_channels = int(min_channels)
        self.reason = reason

    def reasons(self, start: float, end: float) -> tuple[str, ...]:
        """Return the census rejection, when configured and reached."""

        bad = self.monitor.bad_channels(start, end)
        if self.block_on_bad_channels and len(bad) >= self.min_channels:
            return (self.reason,)
        return ()

    def bad_channels(self, start: float, end: float) -> tuple[str, ...]:
        """Return the bad channels the monitor saw in this window."""

        return self.monitor.bad_channels(start, end)
