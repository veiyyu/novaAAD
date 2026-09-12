"""Generate synthetic fixtures and exercise training and replay without downloads."""

import argparse
from pathlib import Path

from nova2026.auditory.data import save_trial

from .replay import replay
from .synthetic import synthetic_trial
from .train import train


def main():
    import json

    from scipy.io import wavfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/auditory_demo")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for name in ("train", "validation", "test"):
        trial = synthetic_trial(name)
        save_trial(trial, output / f"{name}.npz")
        trials.append(trial)
    model, validation = train(trials[:1], trials[1:2], alphas=(10.0, 100.0))
    model.save(output / "decoder.npz")
    audio, estimates, metrics = replay(trials[2], model)
    wavfile.write(output / "mixed.wav", round(trials[2].audio_rate), audio)
    report = {
        "kind": "synthetic plumbing check, not human accuracy",
        "validation": validation,
        "metrics": metrics,
        "estimates": estimates,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Synthetic replay written to {output}")


if __name__ == "__main__":
    main()
