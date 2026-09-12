"""Offline == streamed equivalence (review Tests 1-2, the pre-hardware gate).

The review's most important pre-hardware check: a recording processed *offline*
must produce the same decoder output as the *same* recording fed incrementally
through the live streaming path. If they differ, the live demo will not reproduce
the offline accuracy no matter how well each module runs on its own.

This proves the property for the current design: `WindowAssembler` accumulates
arbitrary-sized chunks (as NOVA's AcquisitionQueue hands them out) and emits a
fixed window; `AADTube` then does *per-window* band-pass + resample + z-score +
reconstruct. So a streamed window must be byte-identical to the offline slice of
the same samples, and the decoder output identical — here we feed deliberately
irregular chunk sizes to catch any boundary/timestamp/ordering bug.

The equivalence is a property of the data *path*, not of decoder quality, so a
fixed decoder is fine. Reproducing the offline *accuracy number* on real KU
Leuven data is the offline analysis (run_viability.py / evaluate.py on your Mac);
running that plus this streamed-equivalence check together covers Tests 1-2.

Run:  PYTHONPATH=. python nova/test_equivalence.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FS
from decoder import fit_decoder, RealtimeDecoder, default_meta
from nova.aad_adapter import AADWindow, AADTube, window_to_decoder_input, NOVA_SFREQ
from nova.run_adapter_replay import WindowAssembler

R = []
def check(name, ok, detail=""):
    R.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")

rng = np.random.default_rng(7)
NAMES = tuple(f"CH{i:02d}" for i in range(64))
NCH = len(NAMES)
ENV_CH = 10
SFREQ = int(NOVA_SFREQ)                          # NOVA rate, 128 Hz
DUR = 40.0
N = int(DUR * SFREQ)
t = np.arange(N) / SFREQ
env_full = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*5*t)     # attended envelope at 128 Hz

# A full "recording" (N x 64) at the NOVA rate; ENV_CH carries the attended envelope.
recording = 0.3 * rng.standard_normal((N, NCH))
recording[:, ENV_CH] = env_full
timestamps = np.arange(N) / SFREQ

# Candidate envelopes at the decoder rate (64 Hz), indexed on source time.
env64 = env_full[::2][: int(DUR * FS)]                        # ~64 Hz view for scoring
env_b = np.roll(env64, 137)
def envelopes(t0, t1, n):
    s = int(round(t0 * FS))
    return [env64[s:s + n], env_b[s:s + n]]

# A decoder (its quality is irrelevant to equivalence; train a light one for realism).
train = []
for _ in range(3):
    e = 0.3 * rng.standard_normal((int(DUR*FS), NCH)); e[:, ENV_CH] = env64[:int(DUR*FS)]
    train.append({"eeg": window_to_decoder_input(e, FS), "att": env64[:int(DUR*FS)]})
dec = RealtimeDecoder(fit_decoder(train), default_meta(NCH, channel_names=NAMES))
tube = AADTube(dec, envelopes, in_fs=SFREQ)

WIN_S = 5.0
win = int(WIN_S * SFREQ)
END = 20 * SFREQ                                  # window we compare ends here

# --- OFFLINE: one contiguous window taken straight from the recording ---
off_win = AADWindow(data=recording[END - win:END].copy(),
                    timestamps=timestamps[END - win:END].copy(),
                    channel_names=NAMES, valid=True)
res_off, _ = tube(off_win)

# --- STREAMED: same samples pushed as irregular chunks through WindowAssembler ---
asm = WindowAssembler(NAMES, sfreq=SFREQ, window_s=WIN_S, step_s=WIN_S)
emitted = []
i = 0
while i < END:
    step = int(rng.integers(7, 53))               # irregular chunk sizes, like a real queue
    j = min(i + step, END)
    asm.push(recording[i:j], timestamps[i:j])
    w = asm.maybe_emit()
    if w is not None:
        emitted.append(w)
    i = j
stream_win = emitted[-1]                            # some window; compare on its OWN span
res_stream, _ = tube(stream_win)

# The streamed window covers a real span of the recording; build the offline window
# from exactly those indices and require the two to match bit-for-bit.
start_idx = int(round(stream_win.timestamps[0] * SFREQ))
off_same = AADWindow(data=recording[start_idx:start_idx + win].copy(),
                     timestamps=timestamps[start_idx:start_idx + win].copy(),
                     channel_names=NAMES, valid=True)
res_off_same, _ = tube(off_same)

check("streamed window is byte-identical to the offline slice of the same samples",
      stream_win.data.shape == off_same.data.shape
      and np.array_equal(stream_win.data, off_same.data)
      and np.allclose(stream_win.timestamps, off_same.timestamps),
      f"span=[{start_idx}:{start_idx + win}] shape={stream_win.data.shape}")

check("offline and streamed decoder scores are identical (irregular chunking)",
      np.allclose(res_off_same.scores, res_stream.scores, atol=1e-12)
      and res_off_same.pick == res_stream.pick,
      f"off={np.round(res_off_same.scores,8).tolist()} "
      f"stream={np.round(res_stream.scores,8).tolist()}")

# The assembler should tile the stream at ~one window per `step` samples.
check("assembler tiles the stream at the expected hop",
      abs(len(emitted) - END // int(WIN_S * SFREQ)) <= 1,
      f"emitted={len(emitted)} ~expected={END // int(WIN_S * SFREQ)}")

print("\n" + "=" * 60)
print(f"{sum(R)}/{len(R)} equivalence checks passed")
sys.exit(0 if all(R) else 1)
