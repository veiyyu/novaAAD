"""Loader for the KU Leuven auditory-attention dataset (one subject per .mat).

Each subject file holds a ``trials`` array. Per trial we extract the 64-channel
EEG, the sampling rate, the two presented stimulus filenames and which track was
attended, then return time-aligned (EEG, attended envelope, unattended envelope).
Field names are read from the struct, not assumed.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt

from config import FS, BAND

_BP = butter(4, list(BAND), "bandpass", fs=FS, output="sos")


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-9)


def preprocess_eeg(eeg_64: np.ndarray) -> np.ndarray:
    """Band-pass to BAND and per-channel z-score EEG already at FS.

    This is the exact transform the decoder expects at its input, so the live path
    (raw EEG from the headset) and the offline path apply the *same* function.
    """
    return _zscore(sosfiltfilt(_BP, np.asarray(eeg_64, dtype=np.float64), axis=0))


def _load_mat(path: str):
    """Load a subject .mat, transparently handling both v5 and v7.3 (HDF5)."""
    import scipy.io as sio
    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return d["trials"], "v5"
    except NotImplementedError:
        import mat73                     # v7.3 / HDF5 fallback
        return mat73.loadmat(path)["trials"], "v73"


def load_kuleuven_subject(path: str, env_cache: dict[str, np.ndarray],
                          filtered: bool = True) -> list[dict]:
    """Return a list of trial dicts: eeg (n,64), att, unatt envelopes, fs, meta.

    EEG is resampled to FS. With ``filtered=True`` (default, used by the offline
    analysis) it is also band-pass filtered to BAND and per-channel z-scored. With
    ``filtered=False`` the raw resampled EEG is returned, so the real-time demo can
    apply the identical ``preprocess_eeg`` per window that it also applies to live
    headset data. Envelopes are always filtered to BAND. All signals are trimmed to
    a common length.
    """
    trials_raw, _ = _load_mat(path)
    out = []
    for t in trials_raw:
        eeg = np.asarray(t.RawData.EegData, dtype=np.float64)   # (n_times, 64)
        fs_in = int(t.FileHeader.SampleRate)
        if fs_in != FS:
            eeg = resample_poly(eeg, FS, fs_in, axis=0)
        if filtered:
            eeg = preprocess_eeg(eeg)

        stims = [str(s) for s in t.stimuli]
        att_track = int(t.attended_track)                      # 1 or 2
        att_fn = next(s for s in stims if f"track{att_track}_" in s)
        unatt_fn = next(s for s in stims if s != att_fn)

        att = sosfiltfilt(_BP, env_cache[att_fn].astype(np.float64))
        unatt = sosfiltfilt(_BP, env_cache[unatt_fn].astype(np.float64))

        n = min(len(eeg), len(att), len(unatt))
        out.append(dict(eeg=eeg[:n], att=att[:n], unatt=unatt[:n], fs=FS,
                        att_fn=att_fn, unatt_fn=unatt_fn,
                        condition=str(t.condition), attended_ear=str(t.attended_ear)))
    return out
