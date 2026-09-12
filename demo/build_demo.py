"""Build the self-contained neuro-steered-hearing demo from the engine's replay output.

The decoding is done by NOVA2026's `nova2026.auditory` pipeline (audio_v2 branch):
run `python -m scripts.auditory.replay --trial T.npz --model M.npz --out DIR`, which
writes `DIR/estimates.json` (per-window AttentionEstimate: scores / valid / emitted_at
/ evidence_end) and `DIR/mixed.wav`. This script turns those estimates into a frame
timeline and inlines it into `ui.html`, producing a standalone `demo.html` that plays
back the live EEG trace and the talker amplification — no server, no engine needed to
VIEW it.

The display controller here mirrors `nova2026.auditory.AttentionController` (margin
0.5, three-window dwell, 3 s evidence expiry, 6 dB duck) purely so the picture matches
what the real controller would do; the DECISIONS themselves come from the engine's
estimates.json, not from this file.

Usage:
    python demo/build_demo.py --estimates output/replay/estimates.json \
           --trial data/converted/S1_test.npz --out demo/demo.html
    python demo/build_demo.py --synthetic --out demo/demo.html      # UI preview, no engine
"""
from __future__ import annotations
import argparse
import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISPLAY_HZ = 10.0
MARGIN = 0.5
MIN_SWITCH = 3
MAX_AGE = 3.0
DUCK_DB = 6.0


class DisplayController:
    """Faithful mirror of nova2026.auditory.AttentionController, for the picture only."""

    def __init__(self, margin=MARGIN, min_switch=MIN_SWITCH, max_age=MAX_AGE):
        self.margin = margin; self.min_switch = min_switch; self.max_age = max_age
        self.selected = None; self.evidence_end = -math.inf
        self.pending = None; self.pending_count = 0

    def update(self, scores, valid, evidence_end, emitted_at, now):
        if not valid or scores is None or evidence_end <= self.evidence_end:
            if not valid:
                self.selected = None; self.pending, self.pending_count = None, 0
            return
        age = now - evidence_end
        if age < 0 or age > self.max_age or emitted_at > now:
            self.selected = None; self.pending, self.pending_count = None, 0
            return
        self.evidence_end = evidence_end
        diff = scores[0] - scores[1]
        if abs(diff) < self.margin:
            self.selected = None; self.pending, self.pending_count = None, 0
        else:
            cand = int(diff < 0)
            if self.selected == cand:
                self.pending, self.pending_count = None, 0
            else:
                self.pending_count = self.pending_count + 1 if self.pending == cand else 1
                self.pending = cand
                if self.pending_count >= self.min_switch:
                    self.selected = cand; self.pending, self.pending_count = None, 0

    def choice(self, now):
        if now < self.evidence_end or now - self.evidence_end > self.max_age:
            return None
        return self.selected


def _load_eeg_trace(trial_path):
    """Best-effort z-scored EEG (n_times, n_ch<=8) + timestamps from a saved trial npz."""
    if not trial_path or not os.path.exists(trial_path):
        return None, None
    try:
        d = np.load(trial_path, allow_pickle=True)
        eeg = np.asarray(d["eeg"], dtype=float)
        ts = np.asarray(d["timestamps"], dtype=float) if "timestamps" in d else \
            np.arange(len(eeg)) / 64.0
        eeg = (eeg - eeg.mean(0)) / (eeg.std(0) + 1e-9)
        return eeg[:, :8], ts
    except Exception:
        return None, None


def _frames_from_estimates(estimates, eeg, ts):
    est = [e for e in estimates if e.get("emitted_at") is not None]
    est.sort(key=lambda e: e["emitted_at"])
    if not est:
        raise SystemExit("estimates.json has no emitted results.")
    t0 = est[0]["emitted_at"]; t1 = est[-1]["emitted_at"] + 1.0
    grid = np.arange(t0, t1, 1.0 / DISPLAY_HZ)
    ctl = DisplayController()
    i = 0; last = {"scores": None}
    duck = 10 ** (-DUCK_DB / 20)
    frames = []
    hits = 0; scored = 0
    for now in grid:
        while i < len(est) and est[i]["emitted_at"] <= now:
            e = est[i]
            ctl.update(e["scores"], e["valid"], e["evidence_end"], e["emitted_at"], now)
            if e["scores"] is not None:
                last = e
            i += 1
        sel = ctl.choice(now)
        ca, cb = (last["scores"] if last["scores"] is not None else (0.0, 0.0))
        gains = [1.0, 1.0]
        if sel is not None:
            gains[1 - sel] = duck
            scored += 1
            if (ca > cb) == (sel == 0):
                hits += 1
        eeg_frame = None
        if eeg is not None and ts is not None:
            k = int(np.searchsorted(ts, now))
            lo = max(0, k - 128)
            seg = eeg[lo:k] if k > lo else eeg[:1]
            step = max(1, len(seg) // 200)
            eeg_frame = seg[::step].T.round(3).tolist()
        frames.append({
            "t": round(float(now - t0), 2),
            "attended": -1 if sel is None else int(sel),
            "gain_a_db": round(20 * math.log10(max(gains[0], 1e-6)), 1),
            "gain_b_db": round(20 * math.log10(max(gains[1], 1e-6)), 1),
            "corr_a": round(float(ca), 4), "corr_b": round(float(cb), 4),
            "eeg": eeg_frame,
        })
    live_pick_acc = (hits / scored) if scored else None
    return frames, live_pick_acc


def _synthetic_frames():
    rng = np.random.default_rng(0)
    frames = []
    n = int(30 * DISPLAY_HZ)
    # attend A for 10 s, B for 10 s, A for 10 s
    for k in range(n):
        t = k / DISPLAY_HZ
        att = 0 if t < 10 else (1 if t < 20 else 0)
        ca = 0.6 + 0.1 * rng.standard_normal() if att == 0 else 0.1 * rng.standard_normal()
        cb = 0.6 + 0.1 * rng.standard_normal() if att == 1 else 0.1 * rng.standard_normal()
        warm = t < 3
        sel = -1 if warm else att
        eeg = (0.4 * rng.standard_normal((6, 120))).round(3).tolist()
        frames.append({
            "t": round(t, 2), "attended": sel,
            "gain_a_db": 0.0 if sel == 0 else (-6.0 if sel == 1 else 0.0),
            "gain_b_db": 0.0 if sel == 1 else (-6.0 if sel == 0 else 0.0),
            "corr_a": round(float(ca), 4), "corr_b": round(float(cb), 4), "eeg": eeg,
        })
    return frames, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimates", help="estimates.json from scripts.auditory.replay")
    ap.add_argument("--trial", help="the replayed trial .npz (for the EEG trace)")
    ap.add_argument("--synthetic", action="store_true", help="UI preview without the engine")
    ap.add_argument("--out", default=os.path.join(HERE, "demo.html"))
    ap.add_argument("--mode", default="quality")
    args = ap.parse_args()

    if args.synthetic or not args.estimates:
        frames, acc = _synthetic_frames()
        meta = {"eeg_source": "synthetic preview", "audio_source": "synthetic A/B",
                "mode": "preview"}
    else:
        estimates = json.loads(open(args.estimates).read())
        eeg, ts = _load_eeg_trace(args.trial)
        frames, acc = _frames_from_estimates(estimates, eeg, ts)
        meta = {"eeg_source": "recorded replay (nova2026.auditory)",
                "audio_source": "two candidate tracks", "mode": args.mode}

    template = open(os.path.join(HERE, "ui.html")).read()
    payload = ("const FRAMES = " + json.dumps(frames) + ";\n"
               + "const META = " + json.dumps(meta) + ";\n"
               + "const LIVE_PICK_ACC = " + json.dumps(acc) + ";\n")
    html = template.replace("/*__DATA__*/", payload)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(frames)} frames, {frames[-1]['t']:.0f}s)")


if __name__ == "__main__":
    main()
