"""Central configuration for the AAD viability analysis.

All processing rates, filter bands and decoder hyper-parameters live here so a
single import gives every module the same, reproducible settings.
"""
from __future__ import annotations

FS = 64                     # Hz — common working rate for EEG and speech envelopes
BAND = (1.0, 9.0)           # Hz — delta/theta band where envelope tracking lives
LAG_MIN, LAG_MAX = 0.0, 0.40  # s — decoder integration window (post-stimulus lags)
ALPHA = 100.0               # ridge regularisation (accuracy is flat across low values)

# Decision-window lengths (seconds) at which accuracy is reported.
WINDOWS = [1, 2, 5, 10, 20, 30, 60]

SUBJECTS = ["S1", "S2", "S3"]   # default subjects to analyse (dataset ships 16)


# ---- real-time demo / feedback settings (see demo_realtime.py) --------------
AUDIO_FS = 44100          # Hz — playback / mixing sample rate
BLOCK_S = 0.032           # s  — audio block size (~32 ms; small for low latency)
DECISION_STEP_S = 1.0     # s  — how often a new attention decision is made
DECISION_WINDOW_S = 15.0  # s  — history the decision correlates over (smoothing)
ATTEN_DB = 10.0           # dB — how much the *ignored* talker is turned down
GAIN_RAMP_S = 0.40        # s  — time constant for gains to glide to their target
HYSTERESIS = 0.015        # min corr-gap before the decision is allowed to flip
