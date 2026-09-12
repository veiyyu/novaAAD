"""Tests for the NOVA2026 AAD tube (numpy/scipy only — NOVA not required).

The current NOVA2026 streaming API is tube-based; the old EEGWindow class is
gone, so these tests exercise the tube against synthetic windows shaped to
NOVA's contract (64 canonical channels, broadband) using this module's own
AADWindow. They cover: decoding the attended talker, the controller steering
and returning to neutral, insufficient / invalid / STALE handling, montage
mismatch, channel-NAME alignment (a window whose channels arrive in a different
order still decodes correctly), and a 128 Hz window resampling to the decoder
rate.

Run:  PYTHONPATH=. python nova/test_adapter.py
"""
import os, sys
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FS
from decoder import fit_decoder, RealtimeDecoder, default_meta
from nova.aad_adapter import (
    AADWindow, AADResult, AADTube, ReliabilityController, window_to_decoder_input,
    NOVA_SFREQ,
)

R = []
def check(name, ok, detail=""):
    R.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")

rng = np.random.default_rng(0)
NAMES = tuple(f"CH{i:02d}" for i in range(64))   # 64-name montage, canonical order
NCH = len(NAMES)
ENV_CH = 10                                       # channel carrying the attended envelope
DUR_TOTAL = 30.0                                  # s of source signal
N = int(DUR_TOTAL * FS)
t = np.arange(N) / FS
env_a = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*5*t)      # attended envelope (64 Hz)
env_b = np.sin(2*np.pi*4*t + 1) + 0.5*np.sin(2*np.pi*2*t)  # competing envelope (64 Hz)


def make_window(start_s, dur_s=2.0, sfreq=FS, names=NAMES, valid=True, reasons=()):
    """A time-consistent synthetic window: channel ENV_CH carries the attended
    envelope for the source segment [start_s, start_s+dur_s), at `sfreq`."""
    n = int(round(dur_s * sfreq))
    ts = start_s + np.arange(n) / sfreq
    s0 = int(round(start_s * FS))
    seg = env_a[s0: s0 + int(round(dur_s * FS))]              # source segment at 64 Hz
    ch = seg if sfreq == FS else resample_poly(seg, int(sfreq), int(FS))
    data = 0.3 * rng.standard_normal((n, NCH))
    m = min(n, len(ch)); data[:m, ENV_CH] = ch[:m]
    return AADWindow(data=data, timestamps=ts, channel_names=names,
                     valid=valid, reasons=reasons)


def envelopes(t0, t1, n):
    """Candidate talker envelopes sampled on the window's source-time grid."""
    s = int(round(t0 * FS))
    return [env_a[s:s + n], env_b[s:s + n]]


# Train a decoder on the same transform the tube uses (channels already aligned).
def train_eeg():
    e = 0.3 * rng.standard_normal((N, NCH)); e[:, ENV_CH] = env_a; return e
trials = [{"eeg": window_to_decoder_input(train_eeg(), FS), "att": env_a} for _ in range(3)]
dec = RealtimeDecoder(fit_decoder(trials), default_meta(NCH, channel_names=NAMES))
tube = AADTube(dec, envelopes, in_fs=FS, action_horizon_s=2.0)

# 1 — valid window -> tube returns (result, window); picks attended talker A
res, passthrough = tube(make_window(5.0, 2.0))
check("tube returns (result, window)", passthrough is not None and hasattr(res, "pick"))
check("valid window decodes attended (A)", res.valid and res.pick == 0 and res.confidence > 0,
      f"pick={res.pick} scores={np.round(res.scores,3).tolist()}")

# 2 — controller steers toward A after a confident result
ctl = ReliabilityController()
ctl.update(res)
for _ in range(40):
    g = ctl.gains(now_source_time=res.end_time)
check("controller steers to A", g[0] > g[1] + 0.3, f"gains={np.round(g,3).tolist()}")

# 3 — caller-flagged invalid window -> abstain -> controller stays neutral (no commit)
inv, _ = tube(make_window(8.0, 2.0, valid=False, reasons=("amplitude",)))
ctl_inv = ReliabilityController()
ctl_inv.update(inv)
for _ in range(60):
    g = ctl_inv.gains(now_source_time=inv.end_time)
check("invalid window -> neutral mix", abs(g[0] - g[1]) < 0.05 and inv.pick is None,
      f"gains={np.round(g,3).tolist()} reasons={inv.reasons}")

# 4 — STALE decision decays to neutral without a new result (freshness at action time)
ctl2 = ReliabilityController()
ctl2.update(tube(make_window(10.0, 2.0))[0])
for _ in range(80):
    g = ctl2.gains(now_source_time=10.0 + 10.0)          # 10 s later, past the 2 s horizon
check("stale decision -> neutral", abs(g[0] - g[1]) < 0.05, f"gains={np.round(g,3).tolist()}")

# 5 — insufficient window (<1 s usable) -> valid abstain, no pick
short, _ = tube(make_window(12.0, 0.5))
check("insufficient window -> abstain", short.valid and short.pick is None
      and short.reasons == ("insufficient_window",), f"reasons={short.reasons}")

# 6 — channel-NAME alignment: reorder the SAME window's columns + names -> identical decode
w = make_window(5.0, 2.0)
res_w, _ = tube(w)
perm = rng.permutation(NCH)
w_shuf = AADWindow(data=w.data[:, perm], timestamps=w.timestamps,
                   channel_names=tuple(NAMES[i] for i in perm), valid=True)
res_shuf, _ = tube(w_shuf)
check("channel-name alignment: reordered window decodes identically",
      res_shuf.pick == res_w.pick and np.allclose(res_shuf.scores, res_w.scores, atol=1e-9),
      f"pick={res_shuf.pick} scores={np.round(res_shuf.scores,4).tolist()}")

# 7 — montage mismatch: decoder needs a channel the window lacks -> safe abstain
w_missing = AADWindow(data=w.data, timestamps=w.timestamps,
                      channel_names=("NOTPRESENT",) + NAMES[1:], valid=True)
res_mm, _ = tube(w_missing)
check("montage mismatch -> safe abstain (no crash)",
      (not res_mm.valid) and res_mm.reasons and res_mm.reasons[0].startswith("pipeline_error"),
      f"reasons={res_mm.reasons}")

# 8 — a 128 Hz (NOVA-rate) window resamples to the decoder rate and still decodes A
res128, _ = AADTube(dec, envelopes, in_fs=NOVA_SFREQ)(
    make_window(5.0, 3.0, sfreq=int(NOVA_SFREQ)))
check("128 Hz window resamples + decodes attended", res128.valid and res128.pick == 0,
      f"pick={res128.pick} scores={np.round(res128.scores,3).tolist()}")

# --- dwell-time smoothing (review finding 6): a switch needs sustained evidence ---
E = 100.0
def mkres(pick, end=E, horizon=2.0):
    scores = [1.0, 0.0] if pick == 0 else [0.0, 1.0]
    return AADResult(True, (), scores, pick, 0.5, end - 2.0, end, 0.0, end + horizon)
def steer(ctl, t=E):
    for _ in range(80):
        g = ctl.gains(now_source_time=t)
    return g

# 9 — commit A, hold through 2 B windows, switch only on the 3rd (min_switch_windows=3)
dctl = ReliabilityController(min_switch_windows=3)
for _ in range(3):
    dctl.update(mkres(0))                     # 3 A windows -> commit A
gA = steer(dctl)
dctl.update(mkres(1)); dctl.update(mkres(1))  # 2 B windows -> not enough to switch
gHold = steer(dctl)
dctl.update(mkres(1))                          # 3rd B -> switch to B
gB = steer(dctl)
check("dwell: hold A through 2 B's, switch on the 3rd",
      gA[0] > gA[1] + 0.3 and gHold[0] > gHold[1] + 0.3 and gB[1] > gB[0] + 0.3,
      f"A={np.round(gA,2).tolist()} hold={np.round(gHold,2).tolist()} B={np.round(gB,2).tolist()}")

# 10 — a single spurious opposite window does not flip a committed pick
sctl = ReliabilityController(min_switch_windows=3)
for _ in range(3):
    sctl.update(mkres(0))                      # commit A
sctl.update(mkres(1))                          # lone spurious B
sctl.update(mkres(0))                          # A again -> B run resets
gSpur = steer(sctl)
check("dwell: single spurious B does not flip A", gSpur[0] > gSpur[1] + 0.3,
      f"gains={np.round(gSpur,2).tolist()}")

print("\n" + "=" * 60)
print(f"{sum(R)}/{len(R)} adapter checks passed")
sys.exit(0 if all(R) else 1)
