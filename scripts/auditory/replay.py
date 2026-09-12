"""Replay held-out EEG and matching audio; save audio, estimates and metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.audio import AudioMixer
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.config import MIN_MARGIN
from nova2026.auditory.data import AttentionEstimate, load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import assert_held_out, inject_fault, selection_metrics
from nova2026.auditory.pipeline import AuditoryPipeline

from .runner import ReplayFailure, replay_windows


def replay(trial, model, margin=MIN_MARGIN, delay=0.0, audio_offset=0.0, mode="quality"):
    """Evaluate emitted decisions on a virtual clock, never historical hindsight."""
    assert_held_out(trial, model)
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Inference delay cannot be negative.")
    history = model.training_info["history"]
    now = [0.0]
    pipeline = AuditoryPipeline(model, lambda: now[0])
    estimates = []
    failure = None
    try:
        for window in replay_windows(
            trial, model.config, history, audio_offset=audio_offset
        ):
            now[0] = window.available_at + delay
            estimate, _ = pipeline.rundown(window)
            estimates.append(estimate)
    except ReplayFailure as error:
        failure = str(error)
        estimates = [
            estimate for estimate in estimates if estimate.emitted_at <= error.timestamp
        ]
        estimates.append(
            AttentionEstimate(
                None, error.timestamp, error.timestamp, False, ("processing_failed",)
            )
        )
    estimates.sort(key=lambda estimate: estimate.emitted_at)
    controller = AttentionController(margin=margin)
    mixer = AudioMixer(trial.audio_rate)
    block_size = round(0.032 * trial.audio_rate)
    output = []
    times = []
    choices = []
    labels = []
    index = 0
    held = None
    start = float(trial.timestamps[0])
    end = min(
        len(trial.audio), round((trial.timestamps[-1] - start) * trial.audio_rate)
    )
    for position in range(0, end, block_size):
        timestamp = start + position / trial.audio_rate
        while index < len(estimates) and estimates[index].emitted_at <= timestamp:
            estimate = estimates[index]
            controller.update(estimate, timestamp)
            if estimate.valid and estimate.scores is not None:
                scores = estimate.scores
                if abs(scores[0] - scores[1]) >= margin:
                    held = int(np.argmax(scores))
            index += 1
        truth_index = np.searchsorted(trial.timestamps, timestamp, side="right") - 1
        truth = int(trial.labels[max(0, truth_index)])
        if mode == "neutral":
            selected = None
        elif mode == "oracle":
            selected = truth if truth in (0, 1) else None
        elif mode == "hysteresis":
            selected = held
        elif mode == "quality":
            selected = controller.choice(timestamp)
        else:
            raise ValueError("Unknown controller mode.")
        gains = np.ones(2)
        if selected is not None:
            gains[1 - selected] = controller.duck
        block = trial.audio[position : min(position + block_size, end)]
        output.append(mixer.process(block, gains))
        times.append(timestamp)
        choices.append(-1 if selected is None else selected)
        labels.append(truth)
    if not times:
        raise ValueError("Trial contains no playable interval.")
    metrics = selection_metrics(times, choices, labels, end_time=start + end / trial.audio_rate)
    metrics["invalid_estimates"] = sum(not estimate.valid for estimate in estimates)
    metrics["audio_underruns"] = None  # Offline rendering is not a hardware test.
    metrics["mode"] = mode
    metrics["processing_failure"] = failure
    return (
        np.concatenate(output),
        [estimate.to_dict() for estimate in estimates],
        metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=["quality", "hysteresis", "neutral", "oracle"],
        default="quality",
    )
    parser.add_argument(
        "--fault", choices=["none", "dropout", "artifact", "mismatched"], default="none"
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--audio-offset", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=MIN_MARGIN)
    parser.add_argument("--zero-model", action="store_true")
    args = parser.parse_args()
    trial = inject_fault(load_trial(args.trial), args.fault)
    model = RidgeDecoder.load(args.model)
    if args.zero_model:
        if model.weights is None:
            raise ValueError("Loaded model has no weights.")
        model.weights[:] = 0
    for partition in ("training", "validation"):
        for subject, identity, group in model.training_info.get(partition, []):
            if (subject, identity) == (
                trial.subject,
                trial.trial_id,
            ) or group == trial.group:
                raise ValueError("Replay evaluation overlaps model development data.")
    audio, estimates, metrics = replay(
        trial, model, args.margin, args.delay, args.audio_offset, args.mode
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    wavfile.write(output / "mixed.wav", round(trial.audio_rate), audio)
    (output / "estimates.json").write_text(json.dumps(estimates, indent=2))
    metrics["fault"] = args.fault
    metrics["zero_model"] = args.zero_model
    metrics["injected_delay"] = args.delay
    metrics["injected_audio_offset"] = args.audio_offset
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
