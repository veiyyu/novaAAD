"""Fixed-capacity ring storage that emits fixed-size windows."""

import math

import numpy as np


class CircularBuffer:
    """Store uniformly sampled rows and emit fixed-size windows.

    This class is storage only: it performs no unit conversion, filtering,
    resampling, quality check or validity decision. Push already processed,
    uniformly sampled rows and receive windows. Preprocessing therefore stays
    independent of the storage strategy, and another buffer implementation only
    has to satisfy the same ``push``/``reset`` contract.

    Args:
        window_samples: Rows per emitted window.
        hop_samples: Rows between consecutive window starts. A hop smaller than
            the window produces overlapping windows; a larger hop skips rows
            between windows.
        capacity_samples: Ring capacity in rows; must hold at least one window.
        sfreq: Sample rate of pushed rows, used to timestamp windows.
        n_channels: Expected columns per row.
        dtype: Storage dtype for pushed rows.

    Notes:
        The ring belongs to one thread: only its owner calls ``push``. Window
        timestamps form a uniform grid anchored at the first finite timestamp
        ever pushed; later input timestamps are not used. ``reset()`` drops that
        anchor and restarts window counting.

    Attributes:
        total_written: Rows written since construction or the last ``reset()``.
        windows: Windows emitted since construction or the last ``reset()``.
    """

    def __init__(
        self,
        window_samples: int,
        hop_samples: int,
        capacity_samples: int,
        sfreq: float,
        n_channels: int,
        dtype: type = np.float64,
    ) -> None:
        """Validate the geometry and allocate the ring."""

        # Every geometry knob must be a sane positive integer.
        for name, value in (
            ("window_samples", window_samples),
            ("hop_samples", hop_samples),
            ("capacity_samples", capacity_samples),
            ("n_channels", n_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        # A window larger than the ring could never be extracted whole.
        if capacity_samples < window_samples:
            raise ValueError("capacity_samples must hold at least one window.")
        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be finite and positive.")

        self._window_samples = window_samples
        self._hop_samples = hop_samples
        self._capacity = capacity_samples
        self._sfreq = float(sfreq)
        self._channels = n_channels
        # dtype is fixed at allocation; incoming rows are cast on push.
        self._dtype = np.dtype(dtype)
        self._ring = np.empty((capacity_samples, n_channels), dtype=self._dtype)
        self.reset()

    def reset(self) -> None:
        """Drop stored rows, forget the anchor and restart window counting."""

        self._write_index = 0      # physical position where the next row goes
        self.total_written = 0     # logical row count since reset (never wraps)
        self.windows = 0
        self._next_end = self._window_samples  # logical end of the next window
        self._anchor: float | None = None      # time of row 0 of this segment

    def push(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Store rows and return every window completed by this call.

        Args:
            data: Samples by channels; any dtype convertible to the ring dtype.
            timestamps: One source timestamp per row.

        Returns:
            Completed windows, oldest first, as
            ``(samples, timestamps, start_sample)``. ``start_sample`` is the
            first row index of the window within the current segment.

        Raises:
            ValueError: If the shapes do not match the configured geometry.
        """

        # Normalize inputs to the storage dtype and expected column count.
        rows = np.asarray(data, dtype=self._dtype)
        times = np.asarray(timestamps, dtype=np.float64)

        if rows.ndim != 2 or rows.shape[1] != self._channels:
            raise ValueError(
                f"Expected samples by {self._channels} channels, "
                f"received shape {rows.shape}."
            )
        if times.ndim != 1 or times.shape[0] != rows.shape[0]:
            raise ValueError(
                f"Received {rows.shape[0]} samples but "
                f"{times.shape[0]} timestamps."
            )
        if rows.shape[0] == 0:
            return []

        # Anchor the segment grid at the first finite timestamp, working back
        # to row 0 even when the first rows carry NaN timestamps.
        if self._anchor is None:
            finite = np.flatnonzero(np.isfinite(times))
            if finite.size:
                index = int(finite[0])
                self._anchor = (
                    float(times[index]) - (self.total_written + index) / self._sfreq
                )

        # Write up to the next window boundary, emit that window, repeat.
        windows = []
        position = 0
        while position < rows.shape[0]:
            step = min(
                self._next_end - self.total_written, rows.shape[0] - position
            )
            self._append(rows[position : position + step])
            position += step

            # A boundary was crossed: copy the finished window out now so later
            # ring writes can never corrupt it.
            if self.total_written == self._next_end:
                windows.append(self._window())
                self.windows += 1
                self._next_end += self._hop_samples

        return windows

    def _append(self, rows: np.ndarray) -> None:
        """Write rows into the ring, wrapping at the capacity boundary."""

        count = rows.shape[0]
        # Fill from the write pointer to the end of the array first...
        first = min(count, self._capacity - self._write_index)
        self._ring[self._write_index : self._write_index + first] = rows[:first]
        remaining = count - first

        # ...then wrap any leftovers around to the start of the array.
        if remaining:
            self._ring[:remaining] = rows[first:]
        self._write_index = (self._write_index + count) % self._capacity
        self.total_written += count

    def _window(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Copy the window that just ended, including a wrapped window."""

        # Logical start row of this window, used for the timestamp grid.
        start_sample = self.total_written - self._window_samples
        # Physical start of the same window inside the ring.
        start = (self._write_index - self._window_samples) % self._capacity
        end = start + self._window_samples

        # A copy is mandatory: the ring keeps being overwritten by later pushes.
        if end <= self._capacity:
            data = self._ring[start:end].copy()
        else:
            # Window straddles the end of the ring: tail + head concatenated.
            data = np.concatenate(
                (self._ring[start:], self._ring[: end - self._capacity]), axis=0
            )

        # Without an anchor there is no real time grid: mark all rows NaN.
        if self._anchor is None:
            timestamps = np.full(self._window_samples, np.nan)
        else:
            timestamps = self._anchor + (
                start_sample + np.arange(self._window_samples)
            ) / self._sfreq

        return data, timestamps, start_sample
