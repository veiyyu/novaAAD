"""Live interactive demo — a local web UI + the real-time neuro-steering engine.

Starts a small local web page (opens in your browser) that shows, in real time,
which talker the listener is attending to and how the two voices are being mixed.
The audio plays out of your headphones; the EEG can be a recorded trial (works with
no hardware) or the live headset.

Quick start (no hardware — recorded EEG, plays the two talkers, live UI):
    python live_demo.py --data-dir /path/to/AAD --subject S3 --trial 4

Two recorded files of your own:
    python live_demo.py --data-dir /path/to/AAD --audio file --files talkerA.wav talkerB.wav

Live people, two clip-on mics (one per talker) on a 2-channel interface:
    python live_demo.py --data-dir /path/to/AAD --audio mic

Build day, live headset over LSL + live mics:
    python live_demo.py --eeg lsl --audio mic

Add --no-audio to run the visual without playing sound (e.g. for a quick test).
Live audio needs the `sounddevice` package; live EEG needs `mne-lsl` (see requirements).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from config import FS, AUDIO_FS, BLOCK_S, DECISION_STEP_S, DECISION_WINDOW_S, ALPHA
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject, preprocess_eeg
from decoder import fit_decoder, RealtimeDecoder, load_decoder, save_decoder, default_meta
from attention_mixer import AttentionMixer
from audio_sources import FileSource, MicSource
from eeg_sources import ReplayEEG, LSLEEG
from demo_realtime import _live_envelope, _corrs

HERE = os.path.dirname(os.path.abspath(__file__))
DISP_SAMPLES = 128        # EEG samples (~2 s at FS) shown scrolling in the UI
STATE = {"running": False, "done": False, "t": 0.0, "attended": -1,
         "gain_a_db": -0.0, "gain_b_db": -0.0, "corr_a": 0.0, "corr_b": 0.0,
         "live_pick_acc": None, "abstain_frac": 0.0,
         "eeg_source": "", "audio_source": "", "mode": "", "eeg": []}
LOCK = threading.Lock()


def _set(**kw):
    with LOCK:
        STATE.update(kw)


# ---------------------------------------------------------------- engine ------
def engine(args):
    stim = os.path.join(args.data_dir, "stimuli")
    stim = stim if os.path.isdir(stim) else args.data_dir
    env = build_envelope_cache(stim, os.path.join(args.out, "envelopes_64hz.npz"))
    mat = os.path.join(args.data_dir, f"{args.subject}.mat")
    raw_trials = load_kuleuven_subject(mat, env, filtered=False)   # for replay + meta
    trial = raw_trials[args.trial]

    if args.decoder:
        # Use a pre-trained (e.g. generic) decoder — zero calibration, instant start.
        w, meta = load_decoder(args.decoder)
    else:
        # Personalized: train on this subject's other trials. Cached after first launch.
        filt_trials = [{**t, "eeg": preprocess_eeg(t["eeg"])} for t in raw_trials]
        cache = os.path.join(args.out, f"dec_{args.subject}_h{args.trial}.npz")
        if os.path.exists(cache):
            w, meta = load_decoder(cache)
        else:
            w = fit_decoder([t for i, t in enumerate(filt_trials) if i != args.trial], ALPHA)
            meta = default_meta(raw_trials[0]["eeg"].shape[1], subject=args.subject,
                                split="personalized-cache")
            os.makedirs(args.out, exist_ok=True); save_decoder(cache, w, meta)
    decoder = RealtimeDecoder(w, meta)

    # EEG source
    if args.eeg == "replay":
        eeg_src = ReplayEEG(raw_trials[args.trial]["eeg"]); has_truth = True
    else:
        eeg_src = LSLEEG(args.lsl_name); has_truth = False

    # audio source
    truth_idx = 0                                  # index of the attended source (replay only)
    if args.audio == "mic":
        audio_src = MicSource(); has_truth = False
    elif args.files:
        audio_src = FileSource(*args.files); has_truth = False
    else:                                      # replay: this trial's own two talkers
        def _find(fn):
            for r, _, fs in os.walk(stim):
                if fn in fs:
                    return os.path.join(r, fn)
            raise FileNotFoundError(fn)
        # FIX (review finding 1): randomize which physical source is A vs B, so the
        # scored decision cannot be right just because the attended file is always A.
        paths = [_find(trial["att_fn"]), _find(trial["unatt_fn"])]
        truth_idx = 0
        if random.random() < 0.5:
            paths = paths[::-1]; truth_idx = 1
        audio_src = FileSource(*paths)

    mixer = AttentionMixer(n_sources=2)
    out_stream = None
    if not args.no_audio:
        import sounddevice as sd
        out_stream = sd.OutputStream(samplerate=AUDIO_FS, channels=1, dtype="float32")
        out_stream.start()

    _set(running=True, done=False,
         eeg_source=("recorded trial" if args.eeg == "replay" else "live LSL"),
         audio_source=("two mics" if args.audio == "mic"
                       else "two tracks"),
         mode=("no audio" if args.no_audio else "playing"))

    decide_every = max(1, int(round(DECISION_STEP_S / BLOCK_S)))
    win_audio = int(DECISION_WINDOW_S * AUDIO_FS); win_eeg = int(DECISION_WINDOW_S * FS)
    ring_a = np.zeros(0, np.float32); ring_b = np.zeros(0, np.float32)
    eeg_ring = np.zeros((0, 64)); audio_samples = 0; eeg_read = 0
    cA = cB = 0.0; raw_correct = raw_total = abstain = 0; blk = 0
    try:
        while True:
            a, b = audio_src.read()
            if a is None:
                break
            audio_samples += len(a)
            target = int(round(audio_samples * FS / AUDIO_FS))
            eeg_block = eeg_src.read(target - eeg_read)
            if eeg_block is None:
                break
            eeg_read = target
            if eeg_ring.shape[1] != eeg_block.shape[1]:
                eeg_ring = np.zeros((0, eeg_block.shape[1]))
            eeg_ring = np.vstack([eeg_ring, eeg_block])[-win_eeg:]
            ring_a = np.concatenate([ring_a, a])[-win_audio:]
            ring_b = np.concatenate([ring_b, b])[-win_audio:]

            if blk % decide_every == 0 and len(eeg_ring) >= FS * 2:
                recon = decoder.reconstruct(preprocess_eeg(eeg_ring))
                cA, cB = _corrs(recon, [_live_envelope(ring_a), _live_envelope(ring_b)])
                att = mixer.set_decision([cA, cB])
                # Honest scoring (finding 1): raw decoder pick, scored against the
                # RANDOMIZED truth index — separate from the mixer's controller state.
                if has_truth:
                    raw_total += 1
                    raw_correct += int(np.argmax([cA, cB]) == truth_idx)
                if att is None:
                    abstain += 1

            out = mixer.process([a, b])
            if out_stream is not None:
                out_stream.write(np.ascontiguousarray(out, np.float32))  # paces to real time
            elif args.audio != "mic":
                time.sleep(BLOCK_S)                                       # sim pacing

            g = mixer.gains_db
            att_idx = mixer.attended
            payload = dict(t=blk * BLOCK_S,
                           attended=(-1 if att_idx is None else int(att_idx)),
                           gain_a_db=round(g[0], 1), gain_b_db=round(g[1], 1),
                           corr_a=round(cA, 3), corr_b=round(cB, 3),
                           live_pick_acc=(raw_correct / raw_total if raw_total else None),
                           abstain_frac=(abstain / max(1, blk // decide_every + 1)))
            if blk % 2 == 0:                       # ~15 Hz EEG trace for the UI
                disp = eeg_ring[-DISP_SAMPLES:]
                if len(disp) >= 8:
                    nch = disp.shape[1]
                    chans = np.linspace(0, nch - 1, min(6, nch)).astype(int)
                    d = disp[:, chans]
                    d = (d - d.mean(0)) / (d.std(0) + 1e-9)
                    payload["eeg"] = np.round(d.T, 2).tolist()   # channels x samples
            _set(**payload)
            blk += 1
    finally:
        if out_stream is not None:
            out_stream.stop(); out_stream.close()
        if hasattr(audio_src, "close"):
            audio_src.close()
        _set(running=False, done=True)


# ---------------------------------------------------------------- server ------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console quiet
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            with open(os.path.join(HERE, "ui.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description="Live neuro-steered hearing demo (web UI).")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subject", default="S3")
    p.add_argument("--trial", type=int, default=4)
    p.add_argument("--eeg", choices=["replay", "lsl"], default="replay")
    p.add_argument("--audio", choices=["file", "mic"], default="file")
    p.add_argument("--files", nargs=2, metavar=("A.wav", "B.wav"))
    p.add_argument("--no-audio", action="store_true", help="Run the visual without sound")
    p.add_argument("--decoder", default=None,
                   help="Path to a pre-trained decoder .npy (skip calibration; e.g. a generic one)")
    p.add_argument("--lsl-name", default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    threading.Thread(target=engine, args=(args,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  Live demo UI:  {url}\n  (Ctrl-C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
