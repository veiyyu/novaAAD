"""Train a ridge decoder with a separate, explicitly grouped validation split."""

import argparse
import json
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import check_split

from .runner import labels_for_window, replay_windows


def prepare(trials, config, history):
    examples = []
    for trial in trials:
        next_start = -np.inf
        for window in replay_windows(trial, config, history, 1.0):
            if window.valid and window.timestamps[0] >= next_start:
                examples.append((window, labels_for_window(trial, window)))
                next_start = window.timestamps[0] + history
    return examples


def train(training, validation, history=5.0, alphas=(10.0, 100.0, 1000.0), base=None):
    check_split(training, validation)
    config = AuditoryConfig()
    # Training and inference use the same history and hop contract. We use
    # nonoverlapping windows initially to avoid counting the same EEG repeatedly.
    training_examples = prepare(training, config, history)
    validation_examples = prepare(validation, config, history)
    best_model = None
    best_accuracy = -1.0
    report = []
    for alpha in alphas:
        model = RidgeDecoder(config, alpha)
        info = {
            "training": [(t.subject, t.trial_id, t.group) for t in training],
            "validation": [(t.subject, t.trial_id, t.group) for t in validation],
            "history": history,
            "step": 1.0,
        }
        if base is not None:
            from nova2026.auditory.evaluation import assert_held_out
            for trial in validation:
                assert_held_out(trial, base)
            info["training"] += base.training_info.get("training", [])
            info["training"] += base.training_info.get("validation", [])
        model.fit(training_examples, info, base=base)
        correct = 0
        total = 0
        for window, labels in validation_examples:
            # Mixed/unknown windows cannot provide a single classification target.
            unique = np.unique(labels)
            if len(unique) != 1 or unique[0] not in (0, 1):
                continue
            scores = model.score(window)
            if abs(scores[0] - scores[1]) > 1e-12:
                correct += int(np.argmax(scores) == unique[0])
            total += 1
        if total == 0:
            raise ValueError("No steady, labeled validation windows.")
        accuracy = correct / total
        report.append({"alpha": alpha, "accuracy": accuracy, "windows": total})
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
    if best_model is None:
        raise ValueError("At least one regularization candidate is required.")
    return best_model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", "--window", type=float, default=5.0)
    parser.add_argument("--base", help="Optional .npz decoder prior with matching contract")
    parser.add_argument("--timing-profile", help="Measured audio profile for calibration recorded through that exact path")
    args = parser.parse_args()
    training = [load_trial(path) for path in args.train]
    validation = [load_trial(path) for path in args.validation]
    base = RidgeDecoder.load(args.base) if args.base else None
    model, report = train(training, validation, args.history, base=base)
    if args.timing_profile:
        from nova2026.auditory.timing import validate_audio_profile
        profile = json.loads(Path(args.timing_profile).read_text())
        for trial in training + validation:
            validate_audio_profile(profile, trial.audio_rate, max(1, round(trial.audio_rate * .032)))
        model.training_info["audio_timing_profile"] = profile
    path = Path(args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    path.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved decoder: {path}")


if __name__ == "__main__":
    main()
