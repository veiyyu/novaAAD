"""Leave-one-stimulus-group-out CV, accuracy-vs-window-length, through the
current (re-vendored) engine, on the real KU Leuven S1-S3 recordings.

Reproduces the old results/aad_results.csv methodology (per-subject accuracy
swept over decision-window length) but on the current causal-envelope feature
contract instead of the old offline Hilbert/zero-phase one. Folds are held
out by AuditoryTrial.group (the stimulus-pair filenames), not naive per-trial
leave-one-out, because nova2026.auditory.evaluation.check_split (ISS-25)
rejects a train/validation split that shares a stimulus group.

Checkpointed at the fold level (data/cv_cache/) because this box is slow
enough (real 64-channel/128Hz EEG, real ridge fits) that a single
(subject, window) run can exceed one shell call's time budget; re-running
this script resumes any incomplete window from its next unfinished fold
instead of restarting. Not a vendored-code change: reuses
tools.replay_windows_tolerant (itself a parameter-forwarding copy of
scripts.auditory.runner.replay_windows) and nova2026.auditory.{decoder,
config,evaluation} exactly as train.py does, with alpha fixed at 100.0 (per
the old config note: "regularisation ... accuracy is flat across low
values").
"""
import argparse
import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import load_trial
from nova2026.auditory.decoder import RidgeDecoder

from tools.replay_windows_tolerant import replay_windows_tolerant

ALPHA = 100.0
ALL_WINDOWS = [1, 2, 5, 10, 20, 30, 60]
ALL_SUBJECTS = ["S1", "S2", "S3"]
CACHE_DIR = Path("data/cv_cache")


def window_key(window):
    return str(int(window) if float(window).is_integer() else window)


def prepare_tolerant(trial, config, history):
    """Same accept/label logic as scripts.auditory.train.prepare, one trial,
    but through replay_windows_tolerant (check_channels=False): this is
    offline analysis of already-recorded EEG, not live hardware monitoring,
    so a per-channel live-run fault policy shouldn't abort a real ~6 minute
    recording over an ordinary artifact the old pipeline never screened for
    either."""
    from scripts.auditory.train import labels_for_window
    examples = []
    next_start = -float("inf")
    for window in replay_windows_tolerant(trial, config, history, 1.0, check_channels=False):
        if window.valid and window.timestamps[0] >= next_start:
            examples.append((window, labels_for_window(trial, window)))
            next_start = window.timestamps[0] + history
    return examples


def examples_by_group(subject, history, config):
    """One trial in memory at a time; returns {group: [(window, labels), ...]}."""
    groups = defaultdict(list)
    for path in sorted(Path(f"data/converted/{subject}").glob("trial_*.npz")):
        trial = load_trial(path)
        groups[trial.group].extend(prepare_tolerant(trial, config, history))
        del trial
        gc.collect()
    return groups


def fit_score(training_examples, validation_examples, config):
    model = RidgeDecoder(config, ALPHA)
    model.fit(training_examples, {"alpha": ALPHA})
    correct = 0
    total = 0
    for window, labels in validation_examples:
        unique = np.unique(labels)
        if len(unique) != 1 or unique[0] not in (0, 1):
            continue
        scores = model.score(window)
        if abs(scores[0] - scores[1]) > 1e-12:
            correct += int(np.argmax(scores) == unique[0])
        total += 1
    return correct, total


def run_subject_window(subject, window, config, budget_seconds):
    import time
    start = time.monotonic()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{subject}_w{window_key(window)}_examples.pkl"
    progress_path = CACHE_DIR / f"{subject}_w{window_key(window)}_progress.json"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            groups = pickle.load(f)
    else:
        groups = examples_by_group(subject, window, config)
        with open(cache_path, "wb") as f:
            pickle.dump(groups, f)

    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    all_group_names = list(groups.keys())
    for held_out in all_group_names:
        if held_out in progress:
            continue
        if time.monotonic() - start > budget_seconds:
            return None  # out of time this call; resume next call
        training_examples = [ex for g, exs in groups.items() if g != held_out for ex in exs]
        validation_examples = groups[held_out]
        correct, total = fit_score(training_examples, validation_examples, config)
        progress[held_out] = [correct, total]
        progress_path.write_text(json.dumps(progress))
        print(f"  fold done: held_out={held_out!r} correct={correct} total={total}", flush=True)

    total_correct = sum(c for c, _ in progress.values())
    total_windows = sum(t for _, t in progress.values())
    return total_correct, total_windows, len(all_group_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS)
    parser.add_argument("--windows", nargs="+", type=float, default=ALL_WINDOWS)
    parser.add_argument("--out", default="results/results_revendor.json")
    parser.add_argument("--budget-seconds", type=float, default=140.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "alpha": ALPHA,
        "subjects": ALL_SUBJECTS,
        "windows": ALL_WINDOWS,
        "engine": "nova2026.auditory (audio_v2, re-vendored @ 13b01f4)",
        "cv": "leave-one-stimulus-group-out (AuditoryTrial.group; check_split-shaped)",
        "step": "non-overlapping",
        "per_subject": {},
        "windows_counts": {},
    }
    config = AuditoryConfig()
    for subject in args.subjects:
        results["per_subject"].setdefault(subject, {})
        results["windows_counts"].setdefault(subject, {})
        for window in args.windows:
            key = window_key(window)
            if key in results["per_subject"][subject]:
                print(f"{subject} window={key}: already done, skipping")
                continue
            outcome = run_subject_window(subject, window, config, args.budget_seconds)
            if outcome is None:
                print(f"{subject} window={key}: out of time this call, will resume next call")
                out_path.write_text(json.dumps(results, indent=2))
                return
            correct, total, n_groups = outcome
            accuracy = correct / total if total else float("nan")
            results["per_subject"][subject][key] = accuracy
            results["windows_counts"][subject][key] = total
            print(f"{subject} window={key:>3}s  groups={n_groups}  windows={total:>5}  "
                  f"correct={correct:>5}  accuracy={accuracy:.4f}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
