"""Task runner for a live AAD calibration session: plays two known talkers
dichotically (candidate A in the left ear, candidate B in the right) and
pushes an LSL marker at session start, at every attention switch, and at
session end -- so a separate recorder (LabRecorder) capturing the EEG LSL
stream alongside this marker stream gets everything needed to build ground
truth later, with LSL's own clock sync handling the alignment.

This script does NOT record EEG. Start LabRecorder first, select both the
EEG stream (from the eego/amplifier) and this script's marker stream
("AAD_Markers" by default), click Record, THEN run this script. Convert the
resulting .xdf with tools/xdf_to_trial.py afterwards.

Uses mne_lsl.lsl (already a project dependency for the live path) for the
marker outlet -- no new LSL binding is required. Needs `sounddevice` for
playback (same optional extra the live audio-output path already documents
in requirements.txt).

Example:
    python tools/calibration_session.py \\
        --audio-a stimuli/partA.wav --audio-b stimuli/partB.wav \\
        --segment-seconds 180 --segments 4 --start-with a
"""
import argparse
import time

import numpy as np
import sounddevice as sd
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from nova2026.auditory.audio import read_audio


def build_marker_outlet(name="AAD_Markers", source_id=None):
    info = StreamInfo(
        name=name, stype="Markers", n_channels=1, sfreq=0.0,
        dtype="string", source_id=source_id or f"{name}-source",
    )
    return StreamOutlet(info)


def dichotic_stereo(track_a, track_b, rate_a, rate_b):
    if rate_a != rate_b:
        raise ValueError(
            f"Candidate audio rates differ ({rate_a} vs {rate_b}); resample to match first."
        )
    count = min(len(track_a), len(track_b))
    stereo = np.zeros((count, 2), dtype=np.float32)
    stereo[:, 0] = track_a[:count]  # left ear: candidate A
    stereo[:, 1] = track_b[:count]  # right ear: candidate B
    return stereo, rate_a


def run(audio_a_path, audio_b_path, segment_seconds, segments, start_with,
        marker_name, countdown_seconds, device=None):
    track_a, rate_a = read_audio(audio_a_path)
    track_b, rate_b = read_audio(audio_b_path)
    stereo, rate = dichotic_stereo(track_a, track_b, rate_a, rate_b)
    total_needed = segment_seconds * segments
    if len(stereo) / rate < total_needed:
        raise ValueError(
            f"Audio is only {len(stereo) / rate:.1f}s long; "
            f"{segments} x {segment_seconds}s segments need {total_needed:.1f}s."
        )

    outlet = build_marker_outlet(marker_name)
    print(f"Marker outlet '{marker_name}' created. Start LabRecorder now, select the "
          f"EEG stream and this marker stream, click Record, then press Enter here.")
    input()

    schedule = []
    current = start_with.lower()
    for _ in range(segments):
        schedule.append(current)
        current = "b" if current == "a" else "a"

    print(f"Starting in {countdown_seconds}s ... candidate A = left ear "
          f"({audio_a_path}), candidate B = right ear ({audio_b_path})")
    time.sleep(countdown_seconds)

    session_start = local_clock()
    outlet.push_sample(["session:start"], session_start)
    print(f"[{0.0:>7.1f}s] session:start")

    position = 0
    block = int(rate * 0.05)
    with sd.OutputStream(samplerate=rate, channels=2, dtype="float32", device=device) as stream:
        for index, attended in enumerate(schedule):
            switch_time = local_clock()
            outlet.push_sample([f"attend:{attended}"], switch_time)
            print(f"[{switch_time - session_start:>7.1f}s] attend:{attended.upper()}  "
                  f"({'LEFT/A' if attended == 'a' else 'RIGHT/B'}) "
                  f"-- segment {index + 1}/{segments}")
            end_position = position + int(segment_seconds * rate)
            while position < end_position:
                chunk = stereo[position:min(position + block, end_position)]
                stream.write(chunk)
                position += len(chunk)

    session_end = local_clock()
    outlet.push_sample(["session:end"], session_end)
    print(f"[{session_end - session_start:>7.1f}s] session:end")
    print("Stop LabRecorder now.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-a", required=True, help="Candidate A (played to the left ear)")
    parser.add_argument("--audio-b", required=True, help="Candidate B (played to the right ear)")
    parser.add_argument("--segment-seconds", type=float, default=180.0)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--start-with", choices=("a", "b"), default="a")
    parser.add_argument("--marker-stream-name", default="AAD_Markers")
    parser.add_argument("--countdown-seconds", type=float, default=5.0)
    parser.add_argument("--device", help="sounddevice output device (index or name); default system output")
    args = parser.parse_args()
    run(args.audio_a, args.audio_b, args.segment_seconds, args.segments,
        args.start_with, args.marker_stream_name, args.countdown_seconds, args.device)


if __name__ == "__main__":
    main()
