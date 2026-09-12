# Calibrating on a new cap (e.g. ANT Neuro eego, 24-channel, saline)

Recording is done by **LabRecorder** (https://github.com/labstreaminglayer/App-LabRecorder),
the standard LSL recording app -- not our code, not something this repo bundles.
Install it separately (Homebrew on macOS: `brew install labstreaminglayer/tap/labrecorder`,
or a prebuilt release from its GitHub page).

This repo adds two small scripts around it:

- `tools/calibration_session.py` -- plays two known talkers dichotically (candidate
  A to the left ear, candidate B to the right) and pushes an LSL marker at session
  start, every attention switch, and session end. It does **not** record EEG.
- `tools/xdf_to_trial.py` -- converts LabRecorder's `.xdf` output (EEG stream +
  the marker stream) into the engine's `trial_*.npz` format, the same way
  `scripts.auditory.convert` does for KU Leuven or AASD recordings.

## Prerequisites

1. eego software: Application options -> Network Operation -> enable LSL EEG
   streaming (same requirement as `scripts.getlive` -- see its own README).
2. `pip install -r requirements.txt` (adds `pyxdf`; `sounddevice` is also needed
   for playback, already in this file).
3. Two audio files per session, known in advance, ideally already split into
   left/right or two isolated tracks (KU Leuven's `stimuli/` files are a fine
   stand-in while testing this pipeline; for a real calibration use whatever
   two-talker content you intend to demo with).

## Running one session

1. Open LabRecorder. It should show the eego's EEG stream and, once you start
   step 2, an `AAD_Markers` string stream. Check both boxes.
2. `python tools/calibration_session.py --audio-a partA.wav --audio-b partB.wav
   --segment-seconds 180 --segments 4 --start-with a` -- prints instructions and
   waits for Enter before playing, so you can click **Record** in LabRecorder
   first, then hit Enter here. The script prints which side to attend and when.
3. When it prints `session:end`, stop the LabRecorder recording. You now have
   one `.xdf` file.
4. Convert it:
   ```
   python tools/xdf_to_trial.py --xdf session1.xdf \
       --audio-a partA.wav --audio-b partB.wav \
       --reference "eego CMS/DRL" \
       --upstream-processing "ANT Neuro eego, 24ch, saline, LSL" \
       --subject P1 --trial-id session1 \
       --out data/converted/calib/session1.npz
   ```
   Channel names are read from the `.xdf` stream description automatically;
   pass `--channel-names` explicitly if that lookup warns it couldn't find them.

## Before you can train: record at least two DIFFERENT audio pairs

`nova2026.auditory.evaluation.check_split` (an audit fix, ISS-25) refuses any
train/validation split that shares a stimulus filename across the two sides.
A trial's "group" here is literally `(audio_a, audio_b)` -- so two trials made
from the *same* two files (even different sessions, different segments) can
never be split against each other, the same way KU Leuven needed four distinct
story parts rather than reusing one. Record calibration with at least two
different audio pairs (e.g. two different podcast segments) before running:

```
python -m scripts.auditory.train \
    --train data/converted/calib/session1.npz \
    --validation data/converted/calib/session2.npz \
    --model models/eego_calib.npz --history 5
```

A single session is still useful as an end-to-end smoke test of the whole
recording -> conversion path even before a second pair exists -- it just can't
train+validate a real decoder alone.

## Then run live

```
python -m scripts.auditory.live --trial <a-known-two-track-file> \
    --model models/eego_calib.npz --stream <lsl-name-from-getlive-probe> \
    --output wav
```
