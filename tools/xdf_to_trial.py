"""Convert a LabRecorder .xdf calibration recording (EEG stream + the
"AAD_Markers" stream from tools/calibration_session.py) into the engine's
AuditoryTrial format, the same way scripts.auditory.convert turns KU Leuven
or AASD recordings into trial_*.npz.

Not a vendored-code change: builds an AuditoryTrial exactly like
nova2026.auditory.data.load_kuleuven/load_aasd_manifest do, just from a
different raw source (LabRecorder's .xdf instead of a KU Leuven .mat or an
AASD .cnt manifest), because neither existing loader reads LSL's own
recording format.

IMPORTANT -- check_split (an audit fix, ISS-25) rejects any train/validation
split that shares a stimulus filename across partitions. This trial's group
is literally the (audio_a, audio_b) pair you recorded with, so a decoder
cannot be validated against another trial made from the SAME two audio
files -- record at least two calibration sessions with two DIFFERENT audio
pairs (like KU Leuven's four story parts) before trying to train/validate a
real decoder, or pass --validation-trial pointing at a trial from a
different pair.

Usage:
    python tools/xdf_to_trial.py --xdf session1.xdf \\
        --audio-a stimuli/partA.wav --audio-b stimuli/partB.wav \\
        --channel-names Fp1 Fp2 F3 ... --reference "eego CMS/DRL" \\
        --upstream-processing "ANT Neuro eego, 24ch, saline, LSL @ 500Hz" \\
        --subject P1 --trial-id session1 --out data/converted/calib/session1.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pyxdf

from nova2026.auditory.audio import read_audio
from nova2026.auditory.data import AuditoryTrial, save_trial

SWITCH_UNCERTAINTY_SECONDS = 0.5


def find_stream(streams, stype=None, name_contains=None):
    matches = []
    for stream in streams:
        info = stream["info"]
        this_type = (info.get("type") or [""])[0]
        this_name = (info.get("name") or [""])[0]
        if stype is not None and this_type.lower() != stype.lower():
            continue
        if name_contains is not None and name_contains.lower() not in this_name.lower():
            continue
        matches.append(stream)
    if len(matches) != 1:
        found = [(s["info"].get("name"), s["info"].get("type")) for s in streams]
        raise ValueError(
            f"Expected exactly one stream matching type={stype!r} name_contains={name_contains!r}, "
            f"found {len(matches)}. Streams in file: {found}"
        )
    return matches[0]


def xdf_channel_labels(eeg_stream, fallback_count):
    """Best-effort extraction of channel labels from the XDF stream description.

    XDF stores this as an XML-derived nested dict; the exact path varies by
    what published the stream, so this tries the common shape and falls back
    to Ch1..N (with a loud warning) rather than guessing wrong silently.
    """
    try:
        channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = [c["label"][0] for c in channels]
        if len(labels) == fallback_count:
            return labels
    except (KeyError, IndexError, TypeError):
        pass
    print(
        f"WARNING: could not read channel labels from the .xdf stream description; "
        f"using Ch1..Ch{fallback_count}. Pass --channel-names to override."
    )
    return [f"Ch{i + 1}" for i in range(fallback_count)]


def build_labels(eeg_times, marker_times, marker_values):
    """-1 (unknown) by default; 0/1 from the most recent attend:a/attend:b
    marker; masked back to -1 within SWITCH_UNCERTAINTY_SECONDS of any
    switch, mirroring load_aasd_manifest's own switch-uncertainty masking."""
    labels = np.full(len(eeg_times), -1, dtype=int)
    switch_times = []
    for t, value in zip(marker_times, marker_values):
        if value == "attend:a":
            labels[eeg_times >= t] = 0
            switch_times.append(t)
        elif value == "attend:b":
            labels[eeg_times >= t] = 1
            switch_times.append(t)
        elif value == "session:end":
            labels[eeg_times >= t] = -1
    for t in switch_times:
        labels[np.abs(eeg_times - t) <= SWITCH_UNCERTAINTY_SECONDS] = -1
    return labels


def convert(xdf_path, audio_a_path, audio_b_path, channel_names, reference,
            upstream_processing, subject, trial_id, marker_stream_name):
    streams, _ = pyxdf.load_xdf(xdf_path)
    eeg_stream = find_stream(streams, stype="EEG")
    marker_stream = find_stream(streams, name_contains=marker_stream_name)

    eeg_raw = np.asarray(eeg_stream["time_series"], dtype=float)
    eeg_times_raw = np.asarray(eeg_stream["time_stamps"], dtype=float)
    marker_times = np.asarray(marker_stream["time_stamps"], dtype=float)
    marker_values = [row[0] for row in marker_stream["time_series"]]

    starts = [t for t, v in zip(marker_times, marker_values) if v == "session:start"]
    ends = [t for t, v in zip(marker_times, marker_values) if v == "session:end"]
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one session:start marker, found {len(starts)}.")
    t0 = starts[0]
    t1 = ends[0] if ends else eeg_times_raw[-1]

    start_idx = np.searchsorted(eeg_times_raw, t0, side="left")
    end_idx = np.searchsorted(eeg_times_raw, t1, side="right")
    if end_idx - start_idx < 2:
        raise ValueError("Cropped EEG window is too short (check the markers landed inside the recording).")
    eeg = eeg_raw[start_idx:end_idx]
    eeg_times = eeg_times_raw[start_idx:end_idx] - t0

    if not np.all(np.diff(eeg_times) > 0):
        order = np.argsort(eeg_times, kind="stable")
        eeg, eeg_times = eeg[order], eeg_times[order]
        keep = np.concatenate([[True], np.diff(eeg_times) > 0])
        eeg, eeg_times = eeg[keep], eeg_times[keep]

    if channel_names is None:
        channel_names = xdf_channel_labels(eeg_stream, eeg.shape[1])
    if len(channel_names) != eeg.shape[1]:
        raise ValueError(
            f"--channel-names has {len(channel_names)} names but the EEG stream has "
            f"{eeg.shape[1]} channels."
        )

    labels = build_labels(eeg_times, marker_times - t0, marker_values)

    track_a, rate_a = read_audio(audio_a_path)
    track_b, rate_b = read_audio(audio_b_path)
    if rate_a != rate_b:
        raise ValueError(f"Candidate audio rates differ ({rate_a} vs {rate_b}).")
    count = min(len(track_a), len(track_b))
    audio = np.column_stack([track_a[:count], track_b[:count]])
    group = "|".join([Path(audio_a_path).name, Path(audio_b_path).name])

    return AuditoryTrial(
        eeg, eeg_times, audio, rate_a, labels,
        subject, trial_id, channel_names, reference, upstream_processing, group,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xdf", required=True)
    parser.add_argument("--audio-a", required=True, help="Same file passed to calibration_session.py as --audio-a")
    parser.add_argument("--audio-b", required=True, help="Same file passed to calibration_session.py as --audio-b")
    parser.add_argument("--channel-names", nargs="+", help="Override; default reads the .xdf stream description")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--upstream-processing", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--marker-stream-name", default="AAD_Markers")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    trial = convert(
        args.xdf, args.audio_a, args.audio_b, args.channel_names, args.reference,
        args.upstream_processing, args.subject, args.trial_id, args.marker_stream_name,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_trial(trial, out_path)
    known = int(np.sum(trial.labels >= 0))
    print(f"Wrote {out_path}  ({len(trial.eeg)} EEG samples, {known} labeled, "
          f"{len(trial.eeg) - known} unknown/masked, group={trial.group!r})")


if __name__ == "__main__":
    main()
