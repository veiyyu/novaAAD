"""End-to-end AAD viability run: envelopes -> decode all subjects -> CSV/PNG/JSON.

Usage:
    python run_viability.py --data-dir /path/to/AAD

The data directory must contain the subject files (S1.mat, ...) and the unzipped
stimuli (a folder of .wav files, searched recursively). Download both from:
    https://zenodo.org/records/4004271
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import numpy as np

from config import SUBJECTS, WINDOWS, ALPHA, FS, BAND, LAG_MIN, LAG_MAX
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject
from decoder import run_subject


def _find_stim_dir(data_dir: str) -> str:
    for cand in (os.path.join(data_dir, "stimuli"), data_dir):
        if any(f.endswith(".wav") for _, _, fs in os.walk(cand) for f in fs):
            return cand
    raise FileNotFoundError(f"No .wav stimuli found under {data_dir!r} — unzip stimuli.zip first.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AAD viability analysis (KU Leuven dataset).")
    ap.add_argument("--data-dir", required=True, help="Folder with S*.mat and stimuli/")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", default="results", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stim_dir = _find_stim_dir(args.data_dir)
    env = build_envelope_cache(stim_dir, os.path.join(args.out, "envelopes_64hz.npz"))
    print(f"Envelopes ready ({len(env)} clips).")

    per_subject, counts = {}, {}
    for s in args.subjects:
        trials = load_kuleuven_subject(os.path.join(args.data_dir, f"{s}.mat"), env)
        acc, ntot = run_subject(trials, args.alpha)
        per_subject[s] = acc; counts[s] = ntot
        print(s, " ".join(f"{w}s:{acc[w]*100:4.1f}%" for w in WINDOWS))

    mean = {w: float(np.mean([per_subject[s][w] for s in args.subjects])) for w in WINDOWS}
    sd = {w: float(np.std([per_subject[s][w] for s in args.subjects])) for w in WINDOWS}
    print("MEAN", " ".join(f"{w}s:{mean[w]*100:4.1f}%" for w in WINDOWS))

    result = dict(alpha=args.alpha, subjects=args.subjects, windows=WINDOWS,
                  per_subject=per_subject, windows_counts=counts, mean=mean, sd=sd,
                  config=dict(fs=FS, band_hz=list(BAND), lags_s=[LAG_MIN, LAG_MAX],
                              cv="leave-one-trial-out", step="non-overlapping",
                              envelope="Hilbert amplitude of presented audio"))
    json.dump(result, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    with open(os.path.join(args.out, "aad_results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_s", *args.subjects, "mean", "sd"])
        for w in WINDOWS:
            wr.writerow([w, *[f"{per_subject[s][w]*100:.1f}" for s in args.subjects],
                         f"{mean[w]*100:.1f}", f"{sd[w]*100:.1f}"])

    _plot(per_subject, mean, sd, args.subjects, os.path.join(args.out, "aad_accuracy_curve.png"))
    print(f"Saved results to {args.out}/")


def _plot(per_subject, mean, sd, subjects, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    palette = ["#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#0072B2"]  # Okabe-Ito
    W = WINDOWS
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.axhspan(40, 50, color="#f2f2f2", zorder=0)
    ax.axhline(50, color="#999", ls="--", lw=1.2); ax.axhline(80, color="#C0504D", ls=":", lw=1.5)
    ax.text(1, 50.6, "chance (50%)", color="#666", fontsize=10)
    ax.text(1, 80.7, "80% viability target", color="#C0504D", fontsize=10)
    for i, s in enumerate(subjects):
        y = [per_subject[s][w] * 100 for w in W]
        ax.plot(W, y, "-o", color=palette[i % len(palette)], lw=1.6, ms=5, alpha=.9, label=s)
    m = [mean[w] * 100 for w in W]; sdv = [sd[w] * 100 for w in W]
    ax.fill_between(W, [a - b for a, b in zip(m, sdv)], [a + b for a, b in zip(m, sdv)],
                    color="#333", alpha=.10)
    ax.plot(W, m, "-o", color="#111", lw=3, ms=7, label=f"Mean (n={len(subjects)})")
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(W))
    ax.xaxis.set_major_formatter(FixedFormatter([str(w) for w in W]))
    ax.set_xlim(0.85, 75); ax.set_ylim(45, 95)
    ax.set_xlabel("Decision window length (seconds)")
    ax.set_ylabel("Attention-decoding accuracy (%)")
    ax.set_title("Auditory Attention Decoding — real EEG (KU Leuven)")
    ax.grid(True, axis="y", color="#eee", lw=.8); ax.legend(loc="lower right", frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
