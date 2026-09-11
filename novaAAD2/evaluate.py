"""Honest evaluation for the AAD decoder — randomized ordering + asserted null controls.

This replaces the biased live "correct_frac" (review finding 1) and the print-only
`verify.py` with a rigorous offline measurement:

- **Randomized A/B ordering.** Each scored window randomly assigns the attended talker
  to slot A or B, so a decoder that always outputs the same slot scores ~50%, not 100%.
- **Asserted null controls.** A zero decoder, time-shuffled EEG, and mismatched audio
  must all land near chance; the run FAILS (raises) if any null control decodes above
  a chance band, or if the real decoder is not above it.

Run on the KU Leuven data:
    python evaluate.py --data-dir /path/to/AAD --subjects S1 S2 S3
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from config import FS, ALPHA, WINDOWS
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject
from decoder import fit_decoder, design, MAX_LAG, _corr


def randomized_accuracy(recon, env_att, env_unatt, win_s, rng, step_s=None):
    """Slide windows; randomly place the attended envelope in slot A or B each window.

    Returns (n_correct, n_windows). A decoder with no signal (or a constant output)
    scores ~50% here by construction.
    """
    w = int(win_s * FS)
    step = int((step_s if step_s else win_s) * FS)
    usable = len(recon) - MAX_LAG                      # exclude zero-padded tail (finding 9)
    if w > usable:
        return 0, 0
    correct = total = 0
    for s in range(0, usable - w + 1, step):
        sl = slice(s, s + w)
        truth = int(rng.random() < 0.5)                # 0 -> attended is A, 1 -> attended is B
        a, b = (env_att, env_unatt) if truth == 0 else (env_unatt, env_att)
        pick = 0 if _corr(recon[sl], a[sl]) >= _corr(recon[sl], b[sl]) else 1
        correct += int(pick == truth); total += 1
    return correct, total


def _recon(trials, held, w, mode="real", rng=None):
    """Reconstruction for a held-out trial under a condition (real / zero / shuffle)."""
    t = trials[held]
    if mode == "zero":
        return np.zeros(len(t["eeg"])), t["att"], t["unatt"]
    if mode == "shuffle":                              # break EEG-audio time correspondence
        eeg = t["eeg"].copy(); rng.shuffle(eeg)        # shuffle time rows
        return design(eeg) @ w, t["att"], t["unatt"]
    if mode == "mismatch":                             # pair recon with a different trial's audio
        j = (held + 1) % len(trials)
        r = design(t["eeg"]) @ w
        m = min(len(r), len(trials[j]["att"]))
        return r[:m], trials[j]["att"][:m], trials[j]["unatt"][:m]
    return design(t["eeg"]) @ w, t["att"], t["unatt"]  # real


def evaluate_subject(trials, win_s=30, alpha=ALPHA, seed=0):
    """Leave-one-trial-out; return accuracy for real + each null control (randomized)."""
    from decoder import _trial_cov
    XtX = [None] * len(trials); Xty = [None] * len(trials)
    for i, tr in enumerate(trials):
        XtX[i], Xty[i] = _trial_cov(tr["eeg"], tr["att"])
    Xs = np.sum(XtX, 0); ys = np.sum(Xty, 0); I = np.eye(Xs.shape[0])
    rng = np.random.default_rng(seed)
    acc = {k: [0, 0] for k in ("real", "zero", "shuffle", "mismatch")}
    for i in range(len(trials)):
        w = np.linalg.solve(Xs - XtX[i] + alpha * I, ys - Xty[i])
        for mode in acc:
            recon, ea, eb = _recon(trials, i, w, mode=mode, rng=rng)
            c, t = randomized_accuracy(recon, ea, eb, win_s, rng)
            acc[mode][0] += c; acc[mode][1] += t
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--win", type=float, default=30.0)
    p.add_argument("--chance-band", type=float, default=0.58,
                   help="null controls must stay below this; real must exceed it")
    args = p.parse_args()
    stim = os.path.join(args.data_dir, "stimuli")
    env = build_envelope_cache(stim if os.path.isdir(stim) else args.data_dir,
                               "results/envelopes_64hz.npz")
    rows = []
    for s in args.subjects:
        tr = load_kuleuven_subject(os.path.join(args.data_dir, f"{s}.mat"), env, filtered=True)
        a = evaluate_subject(tr, win_s=args.win)
        rows.append(a)
        print(f"{s} @ {args.win:.0f}s | real {a['real']*100:4.1f}% | "
              f"zero {a['zero']*100:4.1f}% | shuffle {a['shuffle']*100:4.1f}% | "
              f"mismatch {a['mismatch']*100:4.1f}%")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"MEAN     | real {mean['real']*100:4.1f}% | zero {mean['zero']*100:4.1f}% | "
          f"shuffle {mean['shuffle']*100:4.1f}% | mismatch {mean['mismatch']*100:4.1f}%")

    # ---- the asserts: null controls near chance, real above it ----
    band = args.chance_band
    fails = []
    for k in ("zero", "shuffle", "mismatch"):
        if mean[k] > band:
            fails.append(f"NULL CONTROL '{k}' decoded at {mean[k]*100:.1f}% (> {band*100:.0f}% chance band)")
    if mean["real"] <= band:
        fails.append(f"REAL decoding only {mean['real']*100:.1f}% (<= {band*100:.0f}% band)")
    if fails:
        raise SystemExit("EVALUATION FAILED:\n  " + "\n  ".join(fails))
    print(f"\nPASS: real decoding {mean['real']*100:.1f}% > {band*100:.0f}%; "
          f"all null controls at chance. Result is genuine.")


if __name__ == "__main__":
    main()
