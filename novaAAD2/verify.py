"""Sanity-check controls that the decoded signal is genuine attention tracking.

1. Reconstruction should correlate with the attended envelope more than the
   ignored one (the AAD effect), even though absolute correlations are small.
2. Mismatch null: pairing a subject's EEG with a DIFFERENT trial's audio should
   drop decision accuracy back to chance (~50%).

Usage:
    python verify.py --data-dir /path/to/AAD
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from config import SUBJECTS, ALPHA
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject
from decoder import design, _trial_cov, decode_accuracy, _corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--win", type=float, default=30.0)
    args = ap.parse_args()

    stim = os.path.join(args.data_dir, "stimuli")
    env = build_envelope_cache(stim if os.path.isdir(stim) else args.data_dir,
                               "results/envelopes_64hz.npz")
    for s in args.subjects:
        tr = load_kuleuven_subject(os.path.join(args.data_dir, f"{s}.mat"), env)
        XtX = []; Xty = []
        for t in tr:
            A, b = _trial_cov(t["eeg"], t["att"]); XtX.append(A); Xty.append(b)
        Xs = np.sum(XtX, 0); bs = np.sum(Xty, 0); I = np.eye(Xs.shape[0])
        r_att = []; r_un = []; rc = rt = nc = nt = 0
        n = len(tr)
        for i, t in enumerate(tr):
            w = np.linalg.solve(Xs - XtX[i] + ALPHA * I, bs - Xty[i])
            recon = design(t["eeg"]) @ w
            r_att.append(_corr(recon, t["att"])); r_un.append(_corr(recon, t["unatt"]))
            c, tt = decode_accuracy(recon, t["att"], t["unatt"], args.win); rc += c; rt += tt
            j = (i + 1) % n                                   # mismatched audio
            oa, ob = tr[j]["att"], tr[j]["unatt"]
            m = min(len(recon), len(oa), len(ob))
            c, tt = decode_accuracy(recon[:m], oa[:m], ob[:m], args.win); nc += c; nt += tt
        print(f"{s}: r_att={np.mean(r_att):.3f} r_unatt={np.mean(r_un):.3f} "
              f"(att>unatt {sum(a>u for a,u in zip(r_att,r_un))}/{n}) | "
              f"REAL@{args.win:.0f}s={100*rc/rt:.1f}%  NULL(mismatched)={100*nc/nt:.1f}%")


if __name__ == "__main__":
    main()
