"""Speech-envelope extraction from the raw stimulus WAV files.

The KU Leuven .mat files reference two audio tracks per trial but do not store
precomputed envelopes, so we derive them here: Hilbert amplitude envelope of the
presented audio, anti-alias low-passed and downsampled to FS (64 Hz). Results are
cached to a single .npz keyed by WAV filename, so this only runs once.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly, hilbert, butter, sosfiltfilt

from config import FS


def _envelope(path: str) -> np.ndarray:
    """Return the FS-rate amplitude envelope of one WAV file (float32)."""
    fs, x = wav.read(path)
    return envelope_from_array(x, fs, FS)


def build_envelope_cache(stim_dir: str, out_path: str) -> dict[str, np.ndarray]:
    """Compute envelopes for every WAV under ``stim_dir`` and save to ``out_path``.

    Returns the {filename: envelope} dict. Skips work if the cache already exists.
    """
    if os.path.exists(out_path):
        return dict(np.load(out_path))
    files = sorted(glob.glob(os.path.join(stim_dir, "**", "*.wav"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {stim_dir!r}")
    cache = {os.path.basename(f): _envelope(f) for f in files}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **cache)
    return cache


def envelope_from_array(x, fs_in, fs_out=FS):
    """Amplitude envelope of a raw audio array (mono or multi-channel) at fs_out."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != 8000:
        x = resample_poly(x, 8000, fs_in)
    env = np.abs(hilbert(x))
    sos = butter(4, 20, "low", fs=8000, output="sos")
    env = sosfiltfilt(sos, env)
    return resample_poly(env, fs_out, 8000).astype(np.float32)
