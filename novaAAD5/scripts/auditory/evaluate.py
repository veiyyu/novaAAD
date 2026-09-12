"""Compare controllers and fault conditions with a fixed held-out trial."""

import argparse
import json
import copy
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from nova2026.auditory.data import load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.config import MIN_MARGIN
from nova2026.auditory.evaluation import assert_held_out, inject_fault

from .replay import replay
from .runner import replay_windows, labels_for_window


def score_controls(trial, model, seed=0):
    """Blinded physical candidate order, nonoverlapping windows and explicit ties.

    Intervals describe window counts, not independent human subjects. Null arms
    are diagnostics; no fixed chance-band assertion is used. Zero output must
    abstain, and is never scored as a random correct decision.
    """
    assert_held_out(trial, model)
    random = np.random.default_rng(seed)
    blinded = copy.deepcopy(trial)
    if random.integers(2):
        blinded.audio = blinded.audio[:, ::-1].copy()
        known = blinded.labels >= 0
        blinded.labels[known] = 1 - blinded.labels[known]
    results = {}
    for arm in ("real", "zero", "shuffle", "mismatch"):
        recording = copy.deepcopy(blinded)
        if arm == "shuffle":
            recording.eeg = recording.eeg[random.permutation(len(recording.eeg))]
        if arm == "mismatch":
            recording.audio = np.roll(recording.audio, round(7 * recording.audio_rate), axis=0)
        decoder = copy.deepcopy(model)
        if arm == "zero":
            decoder.weights[:] = 0
        total = correct = abstained = 0
        next_start = -np.inf
        for window in replay_windows(recording, decoder.config, decoder.training_info["history"]):
            if not window.valid or window.timestamps[0] < next_start:
                continue
            next_start = window.timestamps[0] + decoder.training_info["history"]
            labels = np.unique(labels_for_window(recording, window))
            if len(labels) != 1 or labels[0] < 0:
                continue
            scores = decoder.score(window)
            total += 1
            if abs(scores[0] - scores[1]) < MIN_MARGIN:
                abstained += 1
            else:
                correct += int(np.argmax(scores) == labels[0])
        committed = total - abstained
        interval = binomtest(correct, committed).proportion_ci() if committed else None
        results[arm] = {"windows": total, "committed": committed, "abstained": abstained,
                        "correct": correct, "accuracy": correct / committed if committed else None,
                        "binomial_95_interval": list(interval) if interval else None}
    return {"seed": seed, "kind": "held-out software diagnostics; no human-benefit claim", "controls": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    trial = load_trial(args.trial)
    model = RidgeDecoder.load(args.model)
    for partition in ("training", "validation"):
        for subject, identity, group in model.training_info.get(partition, []):
            if (subject, identity) == (
                trial.subject,
                trial.trial_id,
            ) or group == trial.group:
                raise ValueError("Evaluation overlaps model development data.")
    results = []
    for fault in ("none", "dropout", "artifact", "mismatched"):
        recording = inject_fault(trial, fault)
        for mode in ("quality", "hysteresis", "neutral", "oracle"):
            for margin in (0.22, MIN_MARGIN, 0.6):
                _, _, metrics = replay(recording, model, margin=margin, mode=mode)
                results.append({"fault": fault, "margin": margin, **metrics})
    for delay in (1.0, 5.0):
        _, _, metrics = replay(trial, model, delay=delay)
        results.append({"fault": "delayed_inference", "delay": delay, **metrics})
    for offset in (-0.25, 0.25):
        _, _, metrics = replay(trial, model, audio_offset=offset)
        results.append({"fault": "audio_clock_offset", "offset": offset, **metrics})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**score_controls(trial, model, args.seed), "comparisons": results}, indent=2))
    print(f"Wrote {len(results)} comparisons to {path}")


if __name__ == "__main__":
    main()
