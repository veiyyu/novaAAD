"""Adapter tests against NOVA's REAL EEGWindow class (numpy-only, importable).

Run:  PYTHONPATH=. python nova/test_adapter.py
(Point NOVA_REPO at your NOVA2026 checkout if it isn't the default path.)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # our pkg wins
NOVA = os.environ.get("NOVA_REPO", "/home/claude/NOVA2026")
# Import NOVA's EEGWindow by file path so its sibling `config.py` can't shadow ours.
import importlib.util as _il
_spec = _il.spec_from_file_location(
    "nova_window", os.path.join(NOVA, "scripts", "dataproc", "streaming", "window.py"))
_wm = _il.module_from_spec(_spec); _spec.loader.exec_module(_wm)
EEGWindow = _wm.EEGWindow                          # NOVA's real window class

from config import FS
from decoder import fit_decoder, RealtimeDecoder, default_meta, LAGS
from nova.aad_adapter import AADPipeline, ReliabilityController, window_to_decoder_input

R = []
def check(name, ok, detail=""):
    R.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")

rng = np.random.default_rng(0)
N, NCH = 20 * FS, 56
t = np.arange(N) / FS
env_a = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*5*t)      # in-band attended envelope
env_b = np.sin(2*np.pi*4*t + 1) + 0.5*np.sin(2*np.pi*2*t)  # competing

def mk_eeg(env):                                   # ch0 carries the attended envelope
    e = 0.3 * rng.standard_normal((N, NCH)); e[:, 0] = env
    return e

prep = lambda e: window_to_decoder_input(e, FS)    # identical transform to the adapter
trials = [{"eeg": prep(mk_eeg(env_a)), "att": env_a} for _ in range(3)]
dec = RealtimeDecoder(fit_decoder(trials), default_meta(NCH))

def envelopes(t0, t1, n):
    s = int(round(t0 * FS))
    return [env_a[s:s + n], env_b[s:s + n]]

pipe = AADPipeline(dec, envelopes, in_fs=FS, action_horizon_s=2.0)

def make_window(s, win=2 * FS, valid=True, reasons=()):
    return EEGWindow(data=mk_eeg(env_a)[s:s + win], eog=np.zeros((win, 1)),
                     timestamps=np.arange(s, s + win) / FS, valid=valid, reasons=reasons,
                     start_sample=s, channel_names=tuple(f"C{i}" for i in range(NCH)))

# 1 — valid window -> picks the attended source A with positive confidence
res = pipe(make_window(5 * FS))
check("valid window decodes attended (A)", res.valid and res.pick == 0 and res.confidence > 0,
      f"pick={res.pick} scores={np.round(res.scores,3).tolist()}")

# 2 — controller steers toward A after a confident result
ctl = ReliabilityController()
ctl.update(res)
g = None
for _ in range(40):
    g = ctl.gains(now_source_time=res.end_time)     # fresh
check("controller steers to A", g[0] > g[1] + 0.3, f"gains={np.round(g,3).tolist()}")

# 3 — invalid window -> abstain -> controller returns to neutral
inv = pipe(make_window(6 * FS, valid=False, reasons=("amplitude",)))
ctl.update(inv)
for _ in range(60):
    g = ctl.gains(now_source_time=inv.end_time)
check("invalid window -> neutral mix", abs(g[0] - g[1]) < 0.05 and inv.pick is None,
      f"gains={np.round(g,3).tolist()} reasons={inv.reasons}")

# 4 — STALE decision decays to neutral even without a new result (freshness at action time)
ctl2 = ReliabilityController()
ctl2.update(pipe(make_window(7 * FS)))              # confident pick
stale_time = 7 * FS / FS + 10.0                     # 10 s later, past the 2 s horizon
for _ in range(80):
    g = ctl2.gains(now_source_time=stale_time)
check("stale decision -> neutral", abs(g[0] - g[1]) < 0.05, f"gains={np.round(g,3).tolist()}")

# 5 — montage mismatch (64ch decoder on 56ch window) -> safe error-abstain, no crash
dec64 = RealtimeDecoder(np.zeros(64 * len(LAGS)), default_meta(64))
res64 = AADPipeline(dec64, envelopes, in_fs=FS)(make_window(8 * FS))
check("montage mismatch -> safe abstain", (not res64.valid) and res64.reasons
      and res64.reasons[0].startswith("pipeline_error"), f"reasons={res64.reasons}")

print("\n" + "=" * 60)
print(f"{sum(R)}/{len(R)} adapter checks passed")
sys.exit(0 if all(R) else 1)
