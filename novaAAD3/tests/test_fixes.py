"""Regression tests reproducing the review's exact failure cases — now asserting the fixes.

Run:  PYTHONPATH=. python tests/test_fixes.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_sources import _to_mono_float
from attention_mixer import AttentionMixer
from decoder import RealtimeDecoder, DecoderContractError, LAGS, fit_decoder, design
from evaluate import randomized_accuracy

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")


# 1 — stereo PCM scaling (finding 5): stereo must match mono (~0.5), not 16384
mono = _to_mono_float(np.array([16384, -16384], dtype=np.int16))
stereo = _to_mono_float(np.column_stack([np.array([16384, -16384], dtype=np.int16)] * 2))
check("audio scaling: mono ~0.5", abs(mono[0] - 0.5) < 1e-3, f"mono[0]={mono[0]:.4f}")
check("audio scaling: stereo == mono", np.allclose(stereo, mono, atol=1e-4),
      f"stereo[0]={stereo[0]:.4f}")

# 2 — abstention (findings 1 & 7): no evidence must NOT duck a talker
m = AttentionMixer()
att = m.set_decision([0.0, 0.0])
check("mixer abstains on zero evidence", att is None and abs(m.target[0] - m.target[1]) < 1e-9,
      f"attended={att}, target={np.round(m.target,3).tolist()}")

# 3 — output limiting (finding 6): summed sources cannot exceed full scale
m2 = AttentionMixer()
peak = 0.0
m2.set_decision([1.0, 0.0])                     # confident: attended 0 dB + ducked source
for _ in range(30):
    peak = max(peak, float(np.max(np.abs(m2.process([np.ones(64), np.ones(64)])))))
check("audio output bounded to <= 1.0", peak <= 1.0 + 1e-6, f"peak={peak:.4f}")

# 4 — stateful resampler (finding 3): 16-sample chunks @500Hz -> ~2048 out, not 3000
import soxr
from eeg_sources import LSLEEG
class _FakeInlet:
    def pull_chunk(self, *a, **k):
        return np.ones((16, 2)), np.arange(16) / 500.0
src = LSLEEG.__new__(LSLEEG)
src.src_fs, src.fs, src.n_channels, src._closed = 500.0, 64.0, 2, False
src._buf = np.empty((0, 2), dtype=np.float64)
src.inlet = _FakeInlet()
src._rs = soxr.ResampleStream(500.0, 64.0, 2, dtype="float64")
emitted = 0
for _ in range(1000):
    out = src.read(2)
    emitted += len(out)
total = emitted + len(src._buf)                 # all output produced from 16000 input samples
expected = 16000 * 64 / 500                      # = 2048 (minus soxr's fixed latency tail)
# The old per-chunk resample OVER-produced (~3000); a continuous resampler tracks the
# true rate (soxr withholds a small latency tail until flush, so slightly under 2048).
check("stateful resampler tracks true rate (not ~3000)", 1500 < total < 2200,
      f"generated={total} expected~{expected:.0f}; old buggy code gave ~3000")

# 5 — decoder input contract (finding 12): 64-ch weights on 56-ch input must raise
w64 = np.zeros(64 * len(LAGS))
try:
    RealtimeDecoder(w64).reconstruct(np.random.randn(100, 56))
    check("decoder rejects wrong channel count", False, "no error raised")
except DecoderContractError as e:
    check("decoder rejects wrong channel count", True, "raised DecoderContractError")

# 6 — synthetic AAD: real decoder > chance, ZERO decoder ~ chance (finding 1 core)
rng = np.random.default_rng(0)
FS = 64
def smooth_env(n):
    x = rng.standard_normal(n)
    k = np.ones(16) / 16
    return np.convolve(x, k, "same")
fwd = rng.standard_normal((64, len(LAGS)))       # random forward map envelope->EEG
def make_eeg(env):
    eeg = np.zeros((len(env), 64))
    for li, lag in enumerate(LAGS):
        sh = np.roll(env, lag); sh[:lag] = 0
        eeg += np.outer(sh, fwd[:, li])
    return eeg + 1.5 * rng.standard_normal(eeg.shape)
trials = []
for _ in range(6):
    ea, eb = smooth_env(60 * FS), smooth_env(60 * FS)
    trials.append({"eeg": make_eeg(ea), "att": ea, "unatt": eb})
w = fit_decoder(trials[1:])
recon_real = design(trials[0]["eeg"]) @ w
c, t = randomized_accuracy(recon_real, trials[0]["att"], trials[0]["unatt"], 10,
                           np.random.default_rng(1))
acc_real = c / t
c, t = randomized_accuracy(np.zeros(len(trials[0]["eeg"])), trials[0]["att"],
                           trials[0]["unatt"], 10, np.random.default_rng(1))
acc_zero = c / t
check("synthetic: real decoder beats chance", acc_real > 0.65, f"real={acc_real*100:.0f}%")
check("synthetic: ZERO decoder ~ chance (not perfect)", 0.4 <= acc_zero <= 0.6,
      f"zero={acc_zero*100:.0f}%")

print("\n" + "=" * 60)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"{n_pass}/{len(RESULTS)} checks passed")
sys.exit(0 if n_pass == len(RESULTS) else 1)
