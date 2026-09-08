"""Personalize the decoder for one listener — the "stronger result" path.

Records a few minutes where the listener is told which talker to attend, then trains
a decoder tuned to their brain and saves it for the live demo:
    python live_demo.py --decoder results/personal_decoder.npy

Two ways to run it:

  Real headset (build day) — plays two talkers, guides the listener to attend one
  then the other, records EEG over LSL, trains:
      python calibrate.py --data-dir /path/to/AAD --eeg lsl --minutes 4 \
             --files talkerA.wav talkerB.wav --out-file personal_decoder.npy

  Test / no hardware — uses a recorded subject as a stand-in "calibration recording"
  so you can see the whole flow work offline:
      python calibrate.py --data-dir /path/to/AAD --eeg replay --subject S3 --minutes 3

Optional: --base results/generic_decoder.npy warm-starts from a generic decoder,
which helps when the calibration is short (adapts the base instead of starting cold).
"""
from __future__ import annotations
import argparse
import os
import time
import numpy as np

from config import FS, AUDIO_FS, BLOCK_S, ALPHA, DECISION_WINDOW_S
from envelopes import build_envelope_cache, envelope_from_array
from dataset import load_kuleuven_subject, preprocess_eeg
from decoder import _trial_cov, fit_decoder
from demo_realtime import _live_envelope


def _cov(trials):
    XtX = Xty = None
    for t in trials:
        A, b = _trial_cov(t["eeg"], t["att"])
        XtX = A if XtX is None else XtX + A
        Xty = b if Xty is None else Xty + b
    return XtX, Xty


def _train(trials, alpha, base_path=None, alpha_base=3000.0):
    """Fit a decoder from calibration trials; warm-start from a base decoder if given."""
    if base_path:
        w_base = np.load(base_path)
        XtX, Xty = _cov(trials)
        return np.linalg.solve(XtX + alpha_base * np.eye(XtX.shape[0]),
                               Xty + alpha_base * w_base)          # adapt the base
    return fit_decoder(trials, alpha)                              # from scratch


# ---- REPLAY (offline test): use a recorded subject as the calibration recording ----
def collect_replay(args, env):
    trials = load_kuleuven_subject(os.path.join(args.data_dir, f"{args.subject}.mat"),
                                   env, filtered=True)
    if args.exclude is not None:
        trials = [t for i, t in enumerate(trials) if i != args.exclude]
    need = int(args.minutes * 60 * FS)
    out, got = [], 0
    for t in trials:
        if got >= need:
            break
        k = min(len(t["eeg"]), need - got)
        out.append({"eeg": t["eeg"][:k], "att": t["att"][:k]})
        got += k
    print(f"Collected {got / FS:.0f}s of calibration from {args.subject}.")
    return out


# ---- LSL (real headset): guided attend-A / attend-B recording ----------------------
def collect_lsl(args):
    import sounddevice as sd
    from audio_sources import FileSource
    from eeg_sources import LSLEEG
    if not args.files:
        raise SystemExit("--files A.wav B.wav is required for --eeg lsl calibration.")

    eeg_src = LSLEEG(args.lsl_name)
    out_stream = sd.OutputStream(samplerate=AUDIO_FS, channels=1, dtype="float32")
    out_stream.start()

    block_dur = 45                      # seconds attending each talker per block
    schedule, total = [], 0
    while total < args.minutes * 60:
        tgt = "A" if len(schedule) % 2 == 0 else "B"
        schedule.append(tgt); total += block_dur

    trials = []
    for target in schedule:
        src = FileSource(*args.files)
        print(f"\n>>> FOCUS ON TALKER {target}  (for {block_dur}s) <<<")
        for c in (3, 2, 1):
            print(f"   starting in {c}…", end="\r"); time.sleep(1)
        eeg_buf = np.zeros((0, 64)); a_buf = np.zeros(0, np.float32); b_buf = np.zeros(0, np.float32)
        audio_n = eeg_read = 0; t0 = time.time()
        while time.time() - t0 < block_dur:
            a, b = src.read()
            if a is None:
                break
            out_stream.write(np.ascontiguousarray(0.5 * (a + b), np.float32))
            audio_n += len(a)
            tgt_n = int(round(audio_n * FS / AUDIO_FS))
            eb = eeg_src.read(tgt_n - eeg_read)
            if eb is None:
                break
            eeg_read = tgt_n
            if eeg_buf.shape[1] != eb.shape[1]:
                eeg_buf = np.zeros((0, eb.shape[1]))
            eeg_buf = np.vstack([eeg_buf, eb])
            a_buf = np.concatenate([a_buf, a]); b_buf = np.concatenate([b_buf, b])
        att_raw = a_buf if target == "A" else b_buf
        att_env = _live_envelope(att_raw)
        eeg_f = preprocess_eeg(eeg_buf)
        n = min(len(eeg_f), len(att_env))
        trials.append({"eeg": eeg_f[:n], "att": att_env[:n]})
    out_stream.stop(); out_stream.close()
    print("\nRecording done.")
    return trials


def main():
    p = argparse.ArgumentParser(description="Personalize the AAD decoder for one listener.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--eeg", choices=["replay", "lsl"], default="replay")
    p.add_argument("--subject", default="S3", help="(replay mode) recorded subject to stand in")
    p.add_argument("--exclude", type=int, default=None, help="(replay) trial index to leave out")
    p.add_argument("--minutes", type=float, default=3.0, help="Calibration length")
    p.add_argument("--files", nargs=2, metavar=("A.wav", "B.wav"), help="(lsl) two talker tracks")
    p.add_argument("--base", default=None, help="Optional generic decoder .npy to warm-start from")
    p.add_argument("--lsl-name", default=None)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--out", default="results")
    p.add_argument("--out-file", default="personal_decoder.npy")
    args = p.parse_args()

    stim = os.path.join(args.data_dir, "stimuli")
    stim = stim if os.path.isdir(stim) else args.data_dir
    env = build_envelope_cache(stim, os.path.join(args.out, "envelopes_64hz.npz"))

    trials = collect_replay(args, env) if args.eeg == "replay" else collect_lsl(args)
    w = _train(trials, args.alpha, base_path=args.base)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.out_file)
    np.save(path, w)
    print(f"\nSaved personalized decoder -> {path}")
    print(f"Run the demo with it:  python live_demo.py --data-dir {args.data_dir} "
          f"--decoder {path}")


if __name__ == "__main__":
    main()
