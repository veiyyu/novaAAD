"""Fixed-size acquisition from a manually acquired MNE-LSL stream."""

import math
from threading import Event
from time import monotonic

import numpy as np
from mne_lsl.lsl import local_clock
from mne_lsl.stream import StreamLSL


def _check_positive(name: str, value: float) -> None:
    """Reject a value that is not finite and strictly positive."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


class Acquire:
    """Deliver fixed-size sample blocks from a connected, manually acquired stream.

    The caller owns the inlet: it connects, validates metadata and selects
    channels before constructing this object. ``Acquire`` only pulls from the
    source (``stream.acquire()``, plus a one-sample ``get_data`` call that
    clears MNE-LSL's unread counter) and reassembles whatever arrives into
    blocks of exactly ``block_samples`` samples, oldest first. It keeps no ring
    buffer, emits no windows and applies no processing; buffering, unit
    conversion, filtering, resampling and window assembly belong to other
    components.

    Args:
        stream: Connected ``StreamLSL`` opened with ``acquisition_delay=None``
            whose buffer columns are already ordered. The channel count comes
            from ``stream.info['nchan']``. Any object exposing the same
            ``acquire``/``connected``/``n_new_samples``/``get_data``/
            ``add_callback``/``info``/``callbacks`` interface works as well.
        block_samples: Samples per returned block.
        sfreq: Source sampling rate in Hz, for gap statistics and for clearing
            MNE-LSL's unread counter.
        poll_interval: Seconds between manual acquisition attempts.
        no_data_timeout: Default ``read()`` timeout in seconds.
        max_lag_seconds: Age limit for the newest sample of a block; ``None``
            disables the age guard.
        max_future_seconds: Limit for a timestamp ahead of the local LSL clock.
        stop_event: Optional shared shutdown flag; ``stop()`` sets it.

    Notes:
        Manual acquisition makes MNE-LSL invoke the callback on the thread that
        calls ``read()``, so no lock is needed. Call ``close()`` after the last
        ``read()`` and before disconnecting the stream.

    Attributes:
        blocks/samples: Blocks and samples returned so far.
        gaps/max_gap: Count and largest size of spacings above the nominal rate.
        max_lag: Largest observed age of a returned block in seconds.
    """

    def __init__(
        self,
        stream: StreamLSL,
        block_samples: int,
        *,
        sfreq: float,
        poll_interval: float = 0.005,
        no_data_timeout: float = 3.0,
        max_lag_seconds: float | None = 3.0,
        max_future_seconds: float = 0.1,
        stop_event: Event | None = None,
    ) -> None:
        """Validate the block contract and start listening to the source."""

        # Every option must pass the same sanity checks before use.
        if (
            isinstance(block_samples, bool)
            or not isinstance(block_samples, int)
            or block_samples < 1
        ):
            raise ValueError("block_samples must be a positive integer.")
        _check_positive("sfreq", sfreq)
        _check_positive("poll_interval", poll_interval)
        _check_positive("no_data_timeout", no_data_timeout)
        if max_lag_seconds is not None:
            _check_positive("max_lag_seconds", max_lag_seconds)
        if not math.isfinite(max_future_seconds) or max_future_seconds < 0:
            raise ValueError("max_future_seconds must be finite and non-negative.")

        # Caller-facing contract and the connected source.
        self._stream = stream
        self._block_samples = block_samples
        self._sfreq = float(sfreq)
        self._n_channels = int(stream.info["nchan"])
        self._poll_interval = float(poll_interval)
        self._no_data_timeout = float(no_data_timeout)
        self._max_lag_seconds = max_lag_seconds
        self._max_future_seconds = float(max_future_seconds)
        # Stop signal: ours, or one shared with the caller.
        self._stop = stop_event if stop_event is not None else Event()

        # Buffered but not yet block-sized samples (never more than one block).
        self._data: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._pending_samples = 0
        self._previous_timestamp: float | None = None
        self._error: BaseException | None = None
        self._closed = False

        # Public counters.
        self.blocks = self.samples = self.gaps = 0
        self.max_gap = self.max_lag = 0.0

        # MNE-LSL calls _on_chunk on the same thread that calls acquire().
        stream.add_callback(self._on_chunk)

    @property
    def stopped(self) -> bool:
        """Whether shutdown was requested through ``stop()`` or the stop event."""

        return self._stop.is_set()

    @property
    def pending_samples(self) -> int:
        """Samples buffered toward the next block; always below block_samples."""

        return self._pending_samples

    def read(self, timeout: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the next block of exactly ``block_samples`` samples.

        Args:
            timeout: Seconds without source progress before raising; defaults to
                ``no_data_timeout``.

        Returns:
            Samples by channels and their original source timestamps, as private
            copies that never alias the MNE-LSL buffer.

        Raises:
            RuntimeError: Closed, stopped, or disconnected source.
            TimeoutError: No new samples within the timeout.
            Exception: The original error raised while handling a source chunk.
        """

        # Lifecycle guards.
        if self._closed:
            raise RuntimeError("This Acquire instance is closed.")
        if self.stopped:
            raise RuntimeError("Acquisition was stopped.")

        # Resolve and validate the effective timeout.
        limit = self._no_data_timeout if timeout is None else float(timeout)
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("timeout must be finite and non-negative.")

        # Pull until a whole block is buffered, then split one off.
        self._fill(limit)

        return self._take_block()

    def _fill(self, limit: float) -> None:
        """Pull from the source until one full block is buffered or time is out."""

        seen = self._pending_samples
        last_progress = monotonic()

        while self._pending_samples < self._block_samples:
            if self.stopped:
                raise RuntimeError("Acquisition was stopped.")

            # Ask MNE-LSL to pull whatever is available right now.
            self._stream.acquire()
            if not self._stream.connected:
                raise RuntimeError("The source disconnected during acquisition.")
            self._raise_error()  # surface any callback failure immediately
            if self._stream.n_new_samples:
                # Reset MNE-LSL's unread counter with a one-sample view so the
                # internal buffer is not copied and no size warning is logged.
                self._stream.get_data(winsize=1.0 / self._sfreq, exclude=())

            # Reset the no-data clock whenever the buffer actually grows.
            if self._pending_samples != seen:
                seen = self._pending_samples
                last_progress = monotonic()
            if self._pending_samples >= self._block_samples:
                return
            # Timeout measures progress, not wall time since the call started.
            if monotonic() - last_progress >= limit:
                raise TimeoutError(
                    f"No source data arrived within {limit:.3f} seconds."
                )
            # Sleep briefly; stop() wakes this wait immediately.
            self._stop.wait(self._poll_interval)

    def stop(self) -> None:
        """Request shutdown; a blocked ``read()`` returns by raising RuntimeError."""

        self._stop.set()

    def close(self) -> None:
        """Detach the callback and drop buffered samples.

        Call after the last ``read()`` and before disconnecting the source. The
        source itself is never disconnected here.
        """

        if self._closed:
            return
        self._closed = True
        # Discard anything that never became a full block.
        self._data.clear()
        self._timestamps.clear()
        self._pending_samples = 0
        # Stop receiving chunks; ignore if the callback is already gone.
        try:
            self._stream.callbacks.remove(self._on_chunk)
        except ValueError:
            pass

    def _on_chunk(
        self, data: np.ndarray, timestamps: np.ndarray, _
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and buffer one source chunk delivered by MNE-LSL."""

        # Ignore chunks once closed or after the first recorded failure.
        if self._closed or self._error is not None:
            return data, timestamps

        try:
            # Structural checks: shape, lengths and numeric type.
            if data.ndim != 2:
                raise ValueError(
                    "A source chunk must be samples by channels, "
                    f"received shape {data.shape}."
                )
            if timestamps.ndim != 1:
                raise ValueError(
                    "A source chunk needs one timestamp per sample, "
                    f"received shape {timestamps.shape}."
                )
            if data.shape[0] != timestamps.shape[0]:
                raise ValueError(
                    f"Received {data.shape[0]} samples but "
                    f"{timestamps.shape[0]} timestamps."
                )
            if data.shape[0] == 0:
                return data, timestamps
            if data.shape[1] != self._n_channels:
                raise ValueError(
                    f"Expected {self._n_channels} channels, "
                    f"received {data.shape[1]}."
                )
            if not np.issubdtype(data.dtype, np.floating) and not np.issubdtype(
                data.dtype, np.integer
            ):
                raise ValueError(f"Source samples must be numeric, got {data.dtype}.")

            # Record gap statistics without rejecting; repair is downstream.
            self._track_spacing(timestamps)
            # MNE-LSL rolls its internal buffer after the callback, so the
            # chunk must be copied before it is retained.
            self._data.append(np.array(data, copy=True))
            self._timestamps.append(np.array(timestamps, copy=True))
            self._pending_samples += data.shape[0]
        except BaseException as error:  # noqa: BLE001 - re-raised on read()
            # Errors raised here run on the acquisition thread; store the
            # first one and let read() re-raise it on the caller's thread.
            self._error = error

        return data, timestamps

    def _track_spacing(self, timestamps: np.ndarray) -> None:
        """Record gaps without rejecting them; repair is a processing concern."""

        finite = timestamps[np.isfinite(timestamps)]
        if finite.size == 0:
            return

        nominal = 1.0 / self._sfreq
        limit = nominal * 1.5  # more than half a sample late = a gap
        spacing = nominal

        # Largest intra-chunk spacing and the boundary to the previous chunk.
        if finite.size > 1:
            spacing = max(spacing, float(np.max(np.diff(finite))))
        if self._previous_timestamp is not None:
            spacing = max(spacing, float(finite[0]) - self._previous_timestamp)
        if spacing > limit:
            self.gaps += 1
            self.max_gap = max(self.max_gap, spacing)

        self._previous_timestamp = float(finite[-1])

    def _take_block(self) -> tuple[np.ndarray, np.ndarray]:
        """Split exactly one block off the buffered samples."""

        # One buffered chunk can be sliced directly; several must be joined.
        if len(self._data) == 1:
            data = self._data[0]
            timestamps = self._timestamps[0]
        else:
            data = np.concatenate(self._data, axis=0)
            timestamps = np.concatenate(self._timestamps, axis=0)

        # Cut the block off the front and keep the remainder for next time.
        count = self._block_samples
        block = data[:count]
        block_timestamps = timestamps[:count]
        remainder = data[count:]
        remainder_timestamps = timestamps[count:]

        self._data = [remainder] if len(remainder) else []
        self._timestamps = [remainder_timestamps] if len(remainder) else []
        self._pending_samples = len(remainder)

        # Reject stale blocks before they count as delivered.
        self._check_age(block_timestamps)
        self.blocks += 1
        self.samples += count

        return block, block_timestamps

    def _check_age(self, timestamps: np.ndarray) -> None:
        """Reject a block that is too old or too far in the future."""

        finite = timestamps[np.isfinite(timestamps)]
        if finite.size == 0:
            return

        age = local_clock() - float(finite[-1])
        self.max_lag = max(self.max_lag, age)

        if self._max_lag_seconds is not None and age > self._max_lag_seconds:
            raise RuntimeError(f"Source samples are {age:.3f} seconds old.")
        if age < -self._max_future_seconds:
            raise RuntimeError(
                "Source timestamps are ahead of the local LSL clock."
            )

    def _raise_error(self) -> None:
        """Re-raise the first error captured while handling a source chunk."""

        if self._error is not None:
            raise self._error
