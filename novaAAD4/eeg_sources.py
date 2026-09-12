"""Pluggable EEG sources for the demo.

    ReplayEEG — stream a recorded trial's EEG (already at FS) as if it were live.
                Lets the whole demo run today with no headset.
    LSLEEG    — pull live EEG from an LSL stream (the ANT Neuro eego on build day)
                via mne-lsl, resampled to FS. Tested only with hardware present.

Both expose read(n) -> (n, 64) float array at FS, and .fs.
"""
from __future__ import annotations
import numpy as np

from config import FS


class ReplayEEG:
    def __init__(self, eeg: np.ndarray):
        self.eeg = np.asarray(eeg, dtype=np.float64)   # (n_times, 64) at FS
        self.fs = FS
        self.pos = 0

    def read(self, n: int):
        if self.pos + n > len(self.eeg):
            return None
        out = self.eeg[self.pos:self.pos + n]
        self.pos += n
        return out


class LSLEEG:
    """Live EEG via mne-lsl with a CONTINUOUS (stateful) resampler.

    Review fixes:
    - **Stateful resampling (finding 3):** one `soxr.ResampleStream` fed chunk by
      chunk, so sample accounting stays exact across chunk boundaries (the old code
      re-ran `resample_poly` per chunk, adding boundary padding and rounding errors
      that inflated the sample count).
    - **Waiting != ended, no fabricated data (finding 4):** an empty pull returns a
      0-row array meaning "no samples yet"; `None` means the stream has ended.
      Nothing is padded with zeros.

    For rigorous timestamp-synchronized live use, prefer the NOVA acquisition path
    (`nova/`), which carries per-sample timestamps and validity — this class is the
    standalone/legacy live source and aligns only by sample counts.
    """

    ENDED = None  # read() returns this only when the stream is genuinely closed

    def __init__(self, stream_name: str | None = None, n_channels: int = 64):
        from mne_lsl.lsl import resolve_streams, StreamInlet   # hardware only
        import soxr
        streams = resolve_streams()
        if stream_name:
            streams = [s for s in streams if s.name == stream_name]
        if not streams:
            raise RuntimeError("No matching LSL EEG stream found.")
        self.inlet = StreamInlet(streams[0])
        self.inlet.open_stream()
        self.src_fs = float(self.inlet.sfreq)
        self.fs = FS
        self.n_channels = n_channels
        self._closed = False
        self._buf = np.empty((0, n_channels), dtype=np.float64)
        self._rs = soxr.ResampleStream(self.src_fs, self.fs, n_channels, dtype="float64")

    def read(self, n: int):
        """Return up to n freshly-resampled samples (n, ch).

        Returns a 0-row array while waiting for data (NOT end-of-stream), and
        ``LSLEEG.ENDED`` (None) only once the inlet has closed.
        """
        if self._closed:
            return self.ENDED
        chunk, ts = self.inlet.pull_chunk()               # nonblocking; may be empty
        if chunk is not None and len(chunk):
            chunk = np.asarray(chunk, dtype=np.float64)[:, :self.n_channels]
            ds = self._rs.resample_chunk(chunk)           # continuous, stateful
            if len(ds):
                self._buf = np.vstack([self._buf, ds]) if len(self._buf) else ds
        take = min(n, len(self._buf))
        out, self._buf = self._buf[:take], self._buf[take:]
        return out                                        # may be 0 rows == "waiting"

    def close(self):
        self._closed = True
        try:
            self.inlet.close_stream()
        except Exception:
            pass
