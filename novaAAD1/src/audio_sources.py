"""Pluggable audio sources for the demo.

Both sources yield synchronized blocks from two talkers. The rest of the pipeline
does not care which one is used:

    FileSource  — two known audio tracks (default; needs no hardware). Use for
                  development, rehearsal, and the safe headphone demo.
    MicSource   — two live input channels, e.g. one clip-on mic per talker on a
                  2-channel USB interface (left = talker A, right = talker B).

Envelopes for the attention decision are computed live from whatever is playing
(see envelope_from_array), so files and mics work identically downstream.
"""
from __future__ import annotations
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

from config import AUDIO_FS, BLOCK_S


def _to_mono_float(x):
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    return x.astype(np.float32)


class FileSource:
    """Two known tracks, streamed in BLOCK_S chunks at AUDIO_FS."""

    def __init__(self, path_a: str, path_b: str, fs: int = AUDIO_FS, block_s: float = BLOCK_S):
        self.fs = fs
        self.block = int(round(block_s * fs))
        a = self._load(path_a); b = self._load(path_b)
        n = min(len(a), len(b))
        self.a, self.b = a[:n], b[:n]
        self.pos = 0
        self.n_channels = 2

    def _load(self, path):
        fs, x = wav.read(path)
        x = _to_mono_float(x)
        if fs != self.fs:
            x = resample_poly(x, self.fs, fs)
        return x

    def read(self):
        """Return (block_a, block_b) or (None, None) at end of stream."""
        if self.pos + self.block > len(self.a):
            return None, None
        sl = slice(self.pos, self.pos + self.block)
        self.pos += self.block
        return self.a[sl], self.b[sl]


class MicSource:
    """Two live mic channels via sounddevice (one clip-on per talker).

    Requires a 2-input audio device and the `sounddevice` package. Left channel is
    treated as talker A, right as talker B. Tested only with hardware present.
    """

    def __init__(self, fs: int = AUDIO_FS, block_s: float = BLOCK_S, device=None):
        import sounddevice as sd                       # imported lazily; hardware only
        self.fs = fs
        self.block = int(round(block_s * fs))
        self.stream = sd.InputStream(samplerate=fs, channels=2, blocksize=self.block,
                                     dtype="float32", device=device)
        self.stream.start()
        self.n_channels = 2

    def read(self):
        data, _ = self.stream.read(self.block)         # (block, 2)
        return data[:, 0].copy(), data[:, 1].copy()

    def close(self):
        self.stream.stop(); self.stream.close()
