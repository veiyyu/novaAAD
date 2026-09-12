"""Convert KU Leuven or explicitly mapped AASD CNT recordings to numeric trials."""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from nova2026.auditory.audio import read_audio
from nova2026.auditory.data import AuditoryTrial, load_kuleuven, save_trial


def load_aasd_manifest(path):
    """Read an explicit recording manifest; never guess trigger or source identities.

    Each row specifies EEG crop, two isolated matching audio files, and reported
    attention changes in seconds relative to the crop. Boundary labels are masked.
    """
    import mne

    path = Path(path)
    specification = json.loads(path.read_text())
    trials = []
    for row in specification["trials"]:
        raw = mne.io.read_raw_cnt(path.parent / row["eeg_file"], preload=False)
        rate = raw.info["sfreq"]
        start = round(row["start_seconds"] * rate)
        stop = round(row["stop_seconds"] * rate)
        data = raw.get_data(picks=row["channel_names"], start=start, stop=stop)
        eeg = cast(np.ndarray, data).T * 1e6
        times = np.arange(len(eeg)) / rate
        tracks = []
        audio_rate = None
        for filename in row["audio_files"]:
            track, current_rate = read_audio(path.parent / filename)
            if audio_rate is not None and current_rate != audio_rate:
                raise ValueError("Candidate audio rates differ.")
            audio_rate = current_rate
            tracks.append(track)
        if len(tracks) != 2 or len(tracks[0]) != len(tracks[1]):
            raise ValueError("Supply two aligned, equal-length candidate recordings.")
        labels = np.full(len(eeg), -1)
        previous = -1.0
        for event in row["attention_events"]:
            timestamp = float(event["seconds"])
            candidate = int(event["candidate"])
            if timestamp <= previous or candidate not in (0, 1):
                raise ValueError("Attention events must increase and identify A or B.")
            labels[times >= timestamp] = candidate
            previous = timestamp
        uncertainty = row.get("switch_uncertainty_seconds", 0.5)
        for event in row["attention_events"]:
            labels[np.abs(times - event["seconds"]) <= uncertainty] = -1
        trials.append(
            AuditoryTrial(
                eeg,
                times,
                np.column_stack(tracks),
                audio_rate,
                labels,
                row["subject"],
                row["trial_id"],
                row["channel_names"],
                row["reference"],
                row["upstream_processing"],
                row["group"],
            )
        )
        raw.close()
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["kuleuven", "aasd"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stimuli")
    parser.add_argument(
        "--metadata", help="JSON with channel_names, reference, upstream_processing"
    )
    args = parser.parse_args()
    if args.kind == "aasd":
        trials = load_aasd_manifest(args.input)
    else:
        if not args.metadata or not args.stimuli:
            parser.error("KU Leuven needs --metadata and --stimuli.")
        metadata = json.loads(Path(args.metadata).read_text())
        trials = load_kuleuven(args.input, args.stimuli, **metadata)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for index, trial in enumerate(trials):
        save_trial(trial, output / f"trial_{index:03d}.npz")


if __name__ == "__main__":
    main()
