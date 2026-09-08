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
    """Live EEG via mne-lsl. Resamples the incoming stream to FS on the fly."""

    def __init__(self, stream_name: str | None = None, n_channels: int = 64):
        from mne_lsl.lsl import resolve_streams, StreamInlet   # hardware only
        streams = resolve_streams()
        if stream_name:
            streams = [s for s in streams if s.name == stream_name]
        if not streams:
            raise RuntimeError("No matching LSL EEG stream found.")
        self.inlet = StreamInlet(streams[0])
        self.inlet.open_stream()
        self.src_fs = self.inlet.sfreq
        self.fs = FS
        self.n_channels = n_channels
        self._buf = np.empty((0, n_channels))

    def read(self, n: int):
        from scipy.signal import resample_poly
        need_src = int(np.ceil(n * self.src_fs / self.fs)) + 8
        chunk, _ = self.inlet.pull_chunk(max_samples=need_src)
        if chunk is None or len(chunk) == 0:
            return None
        chunk = np.asarray(chunk)[:, :self.n_channels]
        ds = resample_poly(chunk, self.fs, int(round(self.src_fs)), axis=0)
        self._buf = np.vstack([self._buf, ds]) if len(self._buf) else ds
        if len(self._buf) < n:
            return np.zeros((n, self.n_channels))       # warm-up padding
        out, self._buf = self._buf[:n], self._buf[n:]
        return out
