"""Run the AAD decoder on NOVA2026's real-time streaming windows.

NOVA's `Streamer` delivers `EEGWindow` objects (timestamped, quality-checked, canonical
channel order) to a user `pipeline(window) -> result`, and hands the result to
`on_result(result)`. This module provides:

    AADPipeline    — a `pipeline` callable: turns one EEGWindow into an AADResult
                     (reconstruct attended envelope, correlate with timestamp-aligned
                     candidate envelopes, score, or abstain on invalid input).
    ReliabilityController — an `on_result` sink that turns AADResults into audio gains,
                     abstaining to a neutral mix on invalid/uncertain/STALE evidence.
                     Freshness is re-checked at the moment of action (the review's
                     key NOVA fix), not only before inference.

Design notes addressing the review:
- **Timestamp-synchronized pairing (findings 2/3):** candidate speech envelopes are
  sampled at the window's own `timestamps`, not by sample-count guessing.
- **Montage contract (finding 12):** the decoder validates its channel count against
  the window; a KU-Leuven 64ch decoder on NOVA's 56ch input raises, by design — a
  real deployment needs an AAD decoder trained on NOVA's montage.
- **Invalid/stale handling (findings 4/7 + NOVA freshness):** invalid windows abstain;
  the controller decays to neutral when no fresh valid evidence is present.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

from config import FS, BAND, HYSTERESIS, ATTEN_DB
from decoder import RealtimeDecoder, _corr, MAX_LAG
from attention_mixer import db_to_lin

_BP = butter(4, list(BAND), "bandpass", fs=FS, output="sos")


@dataclass
class AADResult:
    """A per-window attention decision, with provenance and an action expiry."""
    valid: bool
    reasons: tuple
    scores: list                  # correlation per candidate source (or [] if invalid)
    pick: int | None              # chosen source index, or None to abstain
    confidence: float             # margin between top-2 scores (0 if abstaining)
    start_time: float
    end_time: float
    available_at: float | None
    emitted_at: float
    expires_at: float             # after this source-time, the decision is stale
    model_id: str = "aad-backward-ridge"

    def to_dict(self):
        return self.__dict__.copy()


def window_to_decoder_input(data: np.ndarray, in_fs: float) -> np.ndarray:
    """NOVA window EEG (samples×ch, µV, in_fs, 1–45 Hz) -> decoder input at FS.

    Applies the decoder's own band (1–9 Hz), resamples to FS, and z-scores per channel
    — the same transform the decoder was trained under (`preprocess_eeg`), re-derived
    for NOVA's rate. Returns (n_times, n_channels).
    """
    x = np.asarray(data, dtype=np.float64)
    if in_fs != FS:
        x = resample_poly(x, int(round(FS)), int(round(in_fs)), axis=0)
    x = sosfiltfilt(_BP, x, axis=0)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


class AADPipeline:
    """`pipeline(window) -> AADResult` for NOVA's Streamer.

    Parameters
    ----------
    decoder : RealtimeDecoder      trained on THIS montage (channels/rate/band).
    envelopes : callable           envelopes(t0, t1, n) -> (n_sources, n) array of
                                    candidate speech envelopes sampled on the window's
                                    time grid (source time). This is where the live
                                    audio system supplies timestamp-aligned candidates.
    in_fs : float                  window sampling rate (NOVA output_sfreq, e.g. 128).
    hysteresis, action_horizon_s : decision margin and how long a decision stays fresh.
    """

    def __init__(self, decoder: RealtimeDecoder, envelopes, in_fs: float = 128.0,
                 hysteresis: float = HYSTERESIS, action_horizon_s: float = 2.0):
        self.decoder = decoder
        self.envelopes = envelopes
        self.in_fs = float(in_fs)
        self.hysteresis = hysteresis
        self.horizon = action_horizon_s

    def __call__(self, window) -> AADResult:
        now = time.monotonic()
        t0, t1 = float(window.timestamps[0]), float(window.timestamps[-1])
        expires = t1 + self.horizon
        if not window.valid:
            return AADResult(False, tuple(window.reasons), [], None, 0.0, t0, t1,
                             window.available_at, now, expires)
        try:
            eeg = window_to_decoder_input(window.data, self.in_fs)
            recon = self.decoder.reconstruct(eeg)          # raises on montage mismatch
            usable = len(recon) - MAX_LAG                  # drop zero-padded tail (finding 9)
            if usable < FS:                                # <1 s usable -> not enough evidence
                return AADResult(True, ("insufficient_window",), [], None, 0.0, t0, t1,
                                 window.available_at, now, expires)
            envs = self.envelopes(t0, t1, len(recon))       # (n_sources, len)
            scores = [float(_corr(recon[:usable], np.asarray(e)[:usable])) for e in envs]
        except Exception as exc:                            # never stall the stream
            return AADResult(False, (f"pipeline_error:{type(exc).__name__}",), [], None,
                             0.0, t0, t1, window.available_at, now, expires)
        order = np.argsort(scores)
        margin = float(scores[order[-1]] - scores[order[-2]]) if len(scores) > 1 else 0.0
        pick = int(order[-1]) if margin > self.hysteresis else None
        return AADResult(True, (), scores, pick, max(margin, 0.0), t0, t1,
                         window.available_at, now, expires)


class ReliabilityController:
    """Turn AADResults into audio gains, abstaining on invalid/uncertain/STALE evidence.

    Provides `gains(now_source_time)` for an independent audio path. The audio callback
    calls `gains()` every block; this class only updates *targets*. Crucially it
    RE-CHECKS freshness at the moment of action: a decision past its `expires_at`
    (relative to the current source time) decays the mix back to neutral even if a new
    result never arrives — closing the review's "stale result still acts" gap.
    """

    def __init__(self, n_sources: int = 2, atten_db: float = ATTEN_DB,
                 ramp_s: float = 0.4, block_s: float = 0.032):
        self.n = n_sources
        self.duck = db_to_lin(-abs(atten_db))
        self.neutral = 1.0 / n_sources
        self.gain = np.full(n_sources, self.neutral)
        self._target = np.full(n_sources, self.neutral)
        self.alpha = 1.0 - np.exp(-block_s / max(ramp_s, 1e-6))
        self._pick: int | None = None
        self._expires_at: float = -np.inf

    def update(self, result: AADResult) -> None:
        """`on_result` sink: accept a fresh, valid, confident pick; else abstain."""
        if result.valid and result.pick is not None:
            self._pick = int(result.pick)
            self._expires_at = float(result.expires_at)
        else:
            self._pick = None                              # invalid/uncertain -> neutral target

    def _refresh_target(self, now_source_time: float) -> None:
        stale = now_source_time is not None and now_source_time > self._expires_at
        if self._pick is None or stale:                    # freshness re-check at action time
            self._target = np.full(self.n, self.neutral)
        else:
            self._target = np.full(self.n, self.duck)
            self._target[self._pick] = 1.0

    def gains(self, now_source_time: float | None = None) -> np.ndarray:
        self._refresh_target(now_source_time if now_source_time is not None else np.inf)
        self.gain += self.alpha * (self._target - self.gain)
        return self.gain.copy()
