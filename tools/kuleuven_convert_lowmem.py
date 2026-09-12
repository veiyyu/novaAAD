"""Memory-safe KU Leuven conversion: identical output to scripts.auditory.convert
--kind kuleuven, but processes one trial at a time and caches decoded stimulus
audio by filename (float32) instead of holding all trials' full-length audio
arrays simultaneously. load_kuleuven in the vendored engine returns all trials
at once, which needs ~5.5GB peak for a 20-trial/64-ch/128Hz subject file; this
box has 3.8GB. Not a vendored-code change -- same AuditoryTrial/save_trial/
read_audio calls, same output format, just different orchestration so it fits
in memory. Resumable: skips a trial whose output file already exists. See
VENDORED.md for the "nothing in the engine itself is modified" policy this
respects.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nova2026.auditory.audio import read_audio
from nova2026.auditory.data import AuditoryTrial, save_trial


def convert_subject(mat_path, stimuli_dir, metadata, out_dir):
    mat_path = Path(mat_path)
    stimuli_dir = Path(stimuli_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    records = np.atleast_1d(d["trials"])
    del d
    gc.collect()

    audio_cache = {}

    def load_cached(name):
        if name not in audio_cache:
            matches = list(stimuli_dir.rglob(name))
            if len(matches) != 1:
                raise ValueError(f"Expected one stimulus file for {name}.")
            samples, rate = read_audio(matches[0])
            audio_cache[name] = (samples.astype(np.float32, copy=False), rate)
        return audio_cache[name]

    written = []
    for index, record in enumerate(records):
        dest = out_dir / f"trial_{index:03d}.npz"
        if dest.exists():
            print(f"  skip {dest} (already converted)")
            written.append(str(dest))
            continue
        eeg = np.asarray(record.RawData.EegData, dtype=float)
        names = sorted(str(name) for name in np.atleast_1d(record.stimuli))
        if len(names) != 2:
            raise ValueError("Expected two stimulus files per trial.")
        tracks = []
        audio_rate = None
        for name in names:
            samples, rate = load_cached(name)
            if audio_rate is not None and rate != audio_rate:
                raise ValueError("Candidate audio rates differ.")
            audio_rate = rate
            tracks.append(samples)
        count = min(len(tracks[0]), len(tracks[1]))
        audio = np.column_stack([tracks[0][:count], tracks[1][:count]])
        target = f"track{int(record.attended_track)}_"
        attended = [i for i, name in enumerate(names) if target in name]
        if len(attended) != 1:
            raise ValueError("Cannot identify the attended track.")
        labels = np.full(len(eeg), attended[0])
        group = "|".join(names)
        trial = AuditoryTrial(
            eeg,
            np.arange(len(eeg)) / float(record.FileHeader.SampleRate),
            audio,
            audio_rate,
            labels,
            mat_path.stem,
            index,
            metadata["channel_names"],
            metadata["reference"],
            metadata["upstream_processing"],
            group,
        )
        save_trial(trial, dest)
        written.append(str(dest))
        print(f"  wrote {dest}  (trial {index}: {group}, attended={attended[0]})", flush=True)
        del eeg, audio, tracks, trial
        gc.collect()
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text())
    written = convert_subject(args.input, args.stimuli, metadata, args.out)
    print(f"Converted {len(written)} trials to {args.out}")


if __name__ == "__main__":
    main()
