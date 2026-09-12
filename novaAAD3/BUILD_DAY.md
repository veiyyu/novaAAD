# Build-day guide — what's tested, and what to do with the ANT Neuro headset

Read this before build day. It's the honest status and the checklist for going live.

## What is already tested and will "just run" after install

- **Viability analysis** (`run_viability.py`, `verify.py`) — fully tested; reproduces
  the results.
- **Real-time demo in replay mode** (`demo_realtime.py --eeg replay --audio file
  --output wav`) — fully tested; renders `demo_output.wav` where the attended talker
  is boosted. This uses recorded EEG + known audio and needs **no hardware**.

To run these you install Python 3.10+, `pip install -r requirements.txt`, and put the
Zenodo data in a folder (see README). This is your guaranteed, can't-fail demo — have
it ready as the fallback.

## What is written but NOT yet tested (needs your hardware/laptop)

These code paths exist and are wired to the right libraries, but I could not test them
here because they need audio hardware and the headset. Budget time to shake them out:

- **Live audio out** (`--output play`, uses `sounddevice`) and **live mics**
  (`--audio mic`). Likely quick to get working; test audio round-trip latency.
- **Live EEG over LSL** (`--eeg lsl`, uses `mne-lsl`). This is the real integration
  task — details below.

## Going live with the ANT Neuro eego — checklist

1. **Get EEG onto LSL.** The demo reads EEG from a Lab Streaming Layer (LSL) stream.
   Confirm your eego setup publishes one (ANT Neuro provides an LSL connector / the
   eego software can stream to LSL; there is also a community LSL app for eego). Note
   the stream's name and pass it: `--eeg lsl --lsl-name "<name>"`. If eego can't emit
   LSL, we bridge via its SDK — tell me the acquisition method and I'll adapt
   `eeg_sources.py`.

2. **Set the channel count/order.** The decoder here was trained on the KU Leuven
   **64-channel BioSemi** layout. Your eego cap has its own channel count, order and
   reference. Set `n_channels` and drop any non-EEG channels in `LSLEEG`.

3. **Calibrate the decoder on the actual rig — the important one.** A decoder trained
   on KU Leuven data will **not** transfer channel-for-channel to a different amplifier
   and cap. Plan a short calibration: have your chosen volunteer attend a known talker
   (alternating left/right for a few minutes) while you record their eego EEG and know
   which side they attended, then train a fresh decoder on that recording with the same
   `fit_decoder`. This subject-and-rig-specific calibration is standard for AAD and is
   what makes it work live. Once you know the LSL stream format, I can write you a
   `calibrate.py` that records, trains, and saves the decoder in ~10 minutes of setup.

4. **Match preprocessing.** `LSLEEG` resamples to 64 Hz and the demo band-passes to
   1–9 Hz and z-scores per window (same transform used in training). Just confirm units
   and that the montage is average/again-referenced similarly; the 1–9 Hz band already
   rejects 50/60 Hz line noise.

5. **Rehearse.** Screen a few teammates and pick the strongest decoder (like S3 here).
   Keep the wearer relatively still (jaw/muscle artifacts hurt). Use a ~20–30 s window
   and let attention "lock on" over a few seconds rather than promising instant switching.

## Calibration strategy — how to run it live with real people

A decoder must be tuned to a brain, so the question is *how* you tune it on the day.
We tested this on the dataset (cross-person). Findings:

- **1 minute of calibration from scratch is not worth it** — it's actually *worse*
  than no calibration (~57% vs 72% at a 60 s window), because a minute is too little
  data to learn a brain from scratch.
- **The fast, reliable option is NO calibration: a "generic" decoder.** Train one
  decoder on a pool of people recorded ahead of time *on your own rig*, and any new
  volunteer uses it instantly. Cross-person we measured ~72% at 60 s / ~66% at 30 s —
  above chance and demo-worthy, with zero wait for the judge.
- More per-person calibration keeps helping (climbs past a few minutes), so offer an
  optional "personalize" mode for a stronger result when you have time.

**Recommended plan:** during practice, record a handful of teammates on the ANT Neuro
cap and build a generic decoder once:

```bash
python pretrain_generic.py --data-dir /path/to/recordings --subjects P1 P2 P3 \
       --out-file generic_decoder.npy
```

Then on the day a judge just puts on the cap — no calibration:

```bash
python live_demo.py --eeg lsl --audio mic --decoder results/generic_decoder.npy
```

Use a 60 s (or long) decision window for the walk-up case, and let attention "lock
on" over time on screen. (The generic decoder still must be trained on *your* cap —
KU-Leuven-trained weights don't transfer across amplifiers.)

**Optional stronger path — personalize (~3 min).** For a hero demo with a specific
volunteer, calibrate on them first (guides them to attend each talker, records, trains):

```bash
python calibrate.py --data-dir /path/to/recordings --eeg lsl --minutes 3 \
       --files talkerA.wav talkerB.wav --out-file personal_decoder.npy
python live_demo.py --eeg lsl --audio mic --decoder results/personal_decoder.npy
```

Add `--base results/generic_decoder.npy` to `calibrate.py` to warm-start from the
generic decoder (helps when calibration is short). Personalized reaches ~82%+ vs the
generic ~72%. Rule of thumb: **generic for walk-ups, personalize for the hero demo;
never a 1-minute from-scratch calibration** (it's worse than generic).

## Bottom line

The science is proven and the pipeline is built and demonstrably works in replay. The
live headset version is **integration work, not plug-and-play** — mainly the LSL
connection and a short on-rig calibration. Lead with the replay demo as your safe
showcase, and treat the live eego version as the stretch you attempt after a calibration
recording.
