"""Run the AAD decoder as a NOVA2026 Pipeline *tube* (main @ da4d4c8, Sept 2026).

NOVA2026's streaming layer was rebuilt. The old `Streamer` / `StreamConfig` /
`EEGWindow` classes this adapter used to target are gone. Real-time EEG now
flows:

    LSL  ->  DefaultStream        (connect + validate + ChannelSelectionContract)
         ->  AcquisitionQueue     (validated chunks; raises on bad/late data)
         ->  Pipeline of *tubes*  (each a Callable[[data], (result, new_data)])

This module provides the two pieces NOVA2026 has no equivalent for:

    AADTube               a NOVA tube: one assembled EEG window -> AADResult
                          (reconstruct the attended envelope, correlate with the
                          live candidate talker envelopes, pick or abstain).
    ReliabilityController turns AADResults into per-talker audio gains, abstaining
                          to a neutral (both-audible) mix on invalid / uncertain /
                          STALE evidence. Freshness is re-checked at the moment of
                          action, not only at inference time.

Contract with NOVA (the seam):
- A window is presented to the tube as an ``AADWindow``: ``data`` shaped
  (n_times, n_channels) in NOVA's canonical channel order (128 Hz, broadband),
  its ``timestamps`` (source time, seconds), and ``channel_names``. The tube
  re-bands to AAD's 1-9 Hz, resamples to the decoder rate, aligns channels to
  the decoder's trained set *by name*, and z-scores — the same transform the
  decoder was trained under, re-derived for NOVA's rate.
- NOVA surfaces invalid data as *exceptions* from AcquisitionQueue
  (StreamDataValidityError / StreamDataLagError / discontinuity), not a per-window
  flag. So the tube abstains on any exception, on an insufficient window, on a
  low decision margin, and on a montage mismatch (a decoder trained on a
  different channel set raises, by design).

This module imports only novaAAD internals, never NOVA2026, so the logic is
testable without NOVA installed (see test_adapter.py). The live wiring onto
NOVA's Pipeline + AcquisitionQueue lives in run_pipeline_replay.py.
"""
from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FS, BAND, HYSTERESIS, ATTEN_DB
from decoder import RealtimeDecoder, DecoderContractError, _corr, MAX_LAG
from attention_mixer import db_to_lin

# NOVA delivers EEG at this rate (NOVA2026 src/nova2026/config.py: SAMPLE_RATE = 128).
NOVA_SFREQ = 128.0

_BP = butter(4, list(BAND), "bandpass", fs=FS, output="sos")


@dataclass
class AADWindow:
    """One assembled EEG window handed to the tube.

    data          : (n_times, n_channels) float, NOVA canonical order, NOVA_SFREQ.
    timestamps    : (n_times,) source time in seconds.
    channel_names : per-column channel names (NOVA canonical order).
    valid         : optional caller-supplied validity. NOVA usually enforces
                    validity upstream by raising, so most windows arrive valid;
                    a caller that already knows a window is bad can pass False.
    reasons       : optional invalidity reasons.
    """
    data: np.ndarray
    timestamps: np.ndarray
    channel_names: tuple = ()
    valid: bool = True
    reasons: tuple = ()


@dataclass
class AADResult:
    """A per-window attention decision, with provenance and an action expiry."""
    valid: bool
    reasons: tuple
    scores: list                 # correlation per candidate talker (or [] if abstaining)
    pick: int | None             # chosen talker index, or None to abstain
    confidence: float            # margin between the top-2 scores (0 if abstaining)
    start_time: float
    end_time: float
    emitted_at: float
    expires_at: float            # after this source-time the decision is stale
    model_id: str = "aad-backward-ridge"

    def to_dict(self):
        return self.__dict__.copy()


def build_channel_selector(window_names, decoder_names):
    """Return column indices that reorder a window's channels into the decoder's
    trained order, selecting by NAME.

    Raises DecoderContractError if the decoder needs a channel the window lacks —
    the safe montage-mismatch behaviour (a decoder trained on a different cap
    must not be silently misapplied).

    If ``decoder_names`` is falsy (a legacy decoder saved without names), returns
    None and the caller falls back to a positional channel-count check.
    """
    if not decoder_names:
        return None
    pos = {name: i for i, name in enumerate(window_names)}
    missing = [c for c in decoder_names if c not in pos]
    if missing:
        raise DecoderContractError(
            f"window is missing {len(missing)} channel(s) the decoder was trained "
            f"on: {missing[:6]}{'...' if len(missing) > 6 else ''}. Retrain on this "
            f"montage or map the channels."
        )
    return np.asarray([pos[c] for c in decoder_names], dtype=int)


def window_to_decoder_input(data, in_fs, selector=None):
    """NOVA window EEG (n_times x n_channels, in_fs, broadband) -> decoder input at FS.

    Aligns channels (if a selector is given), re-bands to the decoder's 1-9 Hz,
    resamples to FS, and z-scores per channel — the transform the decoder was
    trained under, re-derived for NOVA's rate. Returns (n_times, n_channels).
    """
    x = np.asarray(data, dtype=np.float64)
    if selector is not None:
        x = x[:, selector]
    if in_fs != FS:
        x = resample_poly(x, int(round(FS)), int(round(in_fs)), axis=0)
    x = sosfiltfilt(_BP, x, axis=0)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


class AADTube:
    """A NOVA `Pipeline` tube: ``__call__(window) -> (AADResult, window)``.

    The window is passed through unchanged as ``new_data`` so the tube composes
    inside a NOVA Pipeline (downstream tubes still see the window). The decision
    is the tube's ``result``.

    Parameters
    ----------
    decoder : RealtimeDecoder     trained on THIS montage (channels/rate/band).
    envelopes : callable          envelopes(t0, t1, n) -> list of candidate talker
                                   envelopes sampled on the window's own time grid
                                   (source time). This is where the live audio
                                   system supplies timestamp-aligned candidates.
    in_fs : float                 window sampling rate (NOVA SAMPLE_RATE, 128).
    hysteresis, action_horizon_s: decision margin, and how long a decision stays
                                   fresh before the controller lets it go neutral.
    """

    def __init__(self, decoder: RealtimeDecoder, envelopes, in_fs: float = NOVA_SFREQ,
                 hysteresis: float = HYSTERESIS, action_horizon_s: float = 2.0):
        self.decoder = decoder
        self.envelopes = envelopes
        self.in_fs = float(in_fs)
        self.hysteresis = hysteresis
        self.horizon = action_horizon_s
        self._decoder_names = (decoder.meta or {}).get("channel_names")

    def __call__(self, window):
        result = self._decide(window)
        return result, window            # NOVA tube contract: (result, new_data)

    def _decide(self, window) -> AADResult:
        now = time.monotonic()
        ts = np.asarray(window.timestamps, dtype=float)
        t0, t1 = float(ts[0]), float(ts[-1])
        expires = t1 + self.horizon
        if not getattr(window, "valid", True):
            return AADResult(False, tuple(window.reasons), [], None, 0.0,
                             t0, t1, now, expires)
        try:
            selector = build_channel_selector(
                tuple(getattr(window, "channel_names", ())), self._decoder_names)
            eeg = window_to_decoder_input(window.data, self.in_fs, selector)
            recon = self.decoder.reconstruct(eeg)          # raises on montage mismatch
            usable = len(recon) - MAX_LAG                  # drop zero-padded tail
            if usable < FS:                                # < 1 s usable -> not enough
                return AADResult(True, ("insufficient_window",), [], None, 0.0,
                                 t0, t1, now, expires)
            envs = self.envelopes(t0, t1, len(recon))       # list of candidate envelopes
            scores = [float(_corr(recon[:usable], np.asarray(e)[:usable])) for e in envs]
        except Exception as exc:                            # never stall the pipeline
            return AADResult(False, (f"pipeline_error:{type(exc).__name__}",), [], None,
                             0.0, t0, t1, now, expires)
        order = np.argsort(scores)
        margin = float(scores[order[-1]] - scores[order[-2]]) if len(scores) > 1 else 0.0
        pick = int(order[-1]) if margin > self.hysteresis else None
        return AADResult(True, (), scores, pick, max(margin, 0.0),
                         t0, t1, now, expires)


class ReliabilityController:
    """Turn AADResults into audio gains, abstaining on invalid / uncertain / STALE
    evidence.

    Provides ``gains(now_source_time)`` for an independent audio path. The audio
    callback calls ``gains()`` every block; this class only updates *targets*. It
    RE-CHECKS freshness at the moment of action: a decision past its ``expires_at``
    (relative to the current source time) decays the mix back to neutral even if a
    new result never arrives — closing the review's "stale result still acts" gap.
    Neutral = balanced, both talkers audible.
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
        """NOVA on-result sink: accept a fresh, valid, confident pick; else abstain."""
        if getattr(result, "valid", False) and result.pick is not None:
            self._pick = int(result.pick)
            self._expires_at = float(result.expires_at)
        else:
            self._pick = None                              # invalid/uncertain -> neutral

    def _refresh_target(self, now_source_time: float) -> None:
        stale = now_source_time is not None and now_source_time > self._expires_at
        if self._pick is None or stale:                    # freshness re-check at action
            self._target = np.full(self.n, self.neutral)
        else:
            self._target = np.full(self.n, self.duck)
            self._target[self._pick] = 1.0

    def gains(self, now_source_time: float | None = None) -> np.ndarray:
        self._refresh_target(now_source_time if now_source_time is not None else np.inf)
        self.gain += self.alpha * (self._target - self.gain)
        return self.gain.copy()
