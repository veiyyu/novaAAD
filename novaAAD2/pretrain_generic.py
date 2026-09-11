"""Train a GENERIC decoder from several people, for zero-calibration walk-up demos.

A decoder trained on a pool of people works on a *new* person with no calibration
(lower accuracy than a personalized one, but instant). Record a few teammates on
your own rig, point this at those recordings, and save one decoder that any new
volunteer can use immediately via:  live_demo.py --decoder generic_decoder.npy

    python pretrain_generic.py --data-dir /path/to/AAD --subjects S1 S2 --out-file generic_decoder.npy
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from config import ALPHA
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject
from decoder import fit_decoder, save_decoder, default_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subjects", nargs="+", required=True, help="Subjects/recordings to pool")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--out", default="results")
    p.add_argument("--out-file", default="generic_decoder.npy")
    args = p.parse_args()
    stim = os.path.join(args.data_dir, "stimuli")
    stim = stim if os.path.isdir(stim) else args.data_dir
    env = build_envelope_cache(stim, os.path.join(args.out, "envelopes_64hz.npz"))

    pool = []
    for s in args.subjects:
        pool += load_kuleuven_subject(os.path.join(args.data_dir, f"{s}.mat"), env, filtered=True)
    w = fit_decoder(pool, args.alpha)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.out_file)
    if path.endswith(".npy"):
        path = path[:-4] + ".npz"                        # bundles carry their contract
    meta = default_meta(pool[0]["eeg"].shape[1], subject=list(args.subjects),
                        split="generic-pooled")
    save_decoder(path, w, meta)
    print(f"Trained generic decoder on {len(args.subjects)} people "
          f"({len(pool)} trials) -> {path}")
    print(f"Use it with:  python live_demo.py --data-dir {args.data_dir} "
          f"--decoder {path}   (no calibration needed)")


if __name__ == "__main__":
    main()
