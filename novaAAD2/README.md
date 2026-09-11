# neuro-attention

Neuro-steered hearing: read which of two talkers a listener is attending to from
their EEG, and amplify that talker in real time.

**Start here:** [`PROJECT.md`](PROJECT.md) — the full explanation, product pitch,
viability results, and demo architecture in one document.

**Status:** viability proven on real data (mean 82% at a 60 s window, up to 90%);
real-time demo runs today with no hardware via EEG replay.

## Layout

```
neuro-attention/
├── PROJECT.md          # ⭐ full write-up: pitch + method + results + demo
├── README.md           # this file
├── config.py           # ALL settings (rates, band, lags, ridge, demo gains…)
│
│   # --- viability analysis ---
├── envelopes.py        # speech envelopes from the stimulus WAVs (cached)
├── dataset.py          # load_kuleuven_subject(): EEG + attended/ignored envelopes
├── decoder.py          # backward decoder, LOO cross-validation, realtime decoder
├── run_viability.py    # CLI: decode all subjects -> results.json / .csv / .png
├── verify.py           # controls: r_attended>r_ignored, mismatched-audio null
│
│   # --- real-time demo (the "feedback" side) ---
├── audio_sources.py    # FileSource (two tracks) | MicSource (two clip-ons)
├── eeg_sources.py      # ReplayEEG (recorded) | LSLEEG (ANT Neuro via mne-lsl)
├── attention_mixer.py  # smoothed, ramping gain controller (the core feedback)
├── demo_realtime.py    # headless engine: wav render or live playback
│
│   # --- the demo front-end ---
├── live_demo.py        # ⭐ live browser UI + audio engine (mics / files, headphones)
├── ui.html             # the live demo web page (two talker cards, meters)
├── render_demo.py      # render a shareable MP4 of the demo
├── pretrain_generic.py # build a zero-calibration "generic" decoder from several people
├── calibrate.py        # personalize a decoder for one listener (stronger, ~3 min)
├── run_live_demo.command  # double-click launcher (macOS) for the live demo
├── render_video.command   # double-click launcher to make the MP4
├── START_HERE.txt      # plain-text quickstart
│
├── requirements.txt
└── results/            # accuracy curve, demo video (mp4), demo audio, lock-on plot, csv
```

## The demo front-end

The showable demo is `live_demo.py`: it opens a browser page with two "Talker" cards
where the one the listener is attending to lights up and is amplified, driven live by
the decoder. Runs today on any laptop with recorded EEG — no headset needed.

```bash
# double-click run_live_demo.command, or:
python live_demo.py --data-dir /path/to/AAD                 # recorded EEG + two tracks
python live_demo.py --data-dir /path/to/AAD --audio file --files A.wav B.wav
python live_demo.py --data-dir /path/to/AAD --audio mic     # two clip-on mics
python live_demo.py --data-dir /path/to/AAD --eeg lsl --audio mic   # build day, live
python render_demo.py --data-dir /path/to/AAD               # -> results/demo_video.mp4
```

Live sound needs `sounddevice`; live EEG needs `mne-lsl` (both optional, in
`requirements.txt`). Add `--no-audio` to run the visual without playing sound.

### Two ways to run it live (choose per demo)

The decoder must be tuned to a brain. You have both options ready — pick with
`live_demo.py --decoder <file>`:

```bash
# 1) INSTANT, zero calibration — a generic decoder built from several people ahead of time.
python pretrain_generic.py --data-dir /path/to/AAD --subjects P1 P2 P3   # once, on your rig
python live_demo.py --data-dir /path/to/AAD --decoder results/generic_decoder.npy

# 2) STRONGER, ~3 min personalize — calibrate on the specific listener, then run.
python calibrate.py  --data-dir /path/to/AAD --eeg lsl --minutes 3 --files A.wav B.wav
python live_demo.py  --data-dir /path/to/AAD --decoder results/personal_decoder.npy
```

Measured trade-off (`results/calibration_analysis.png`): generic ≈ 72% at a 60 s
window with **no** wait; personalized reaches ~82%+ but needs a few minutes. **1 minute
of calibration is worse than the generic decoder — don't do it.** Use generic for
walk-up judges, personalize for a hero demo.

## Get the data (not committed — too large)

Download the subject files and `stimuli.zip` from
<https://zenodo.org/records/4004271> and arrange:

```
AAD/
├── S1.mat  S2.mat  S3.mat      # subject EEG
└── stimuli/ … *.wav            # unzip stimuli.zip here
```

## Run

```bash
pip install -r requirements.txt

# 1) reproduce the viability result (writes results/)
python run_viability.py --data-dir /path/to/AAD

# 2) sanity-check controls
python verify.py --data-dir /path/to/AAD

# 3) the real-time demo — today, no hardware (renders demo_output.wav):
python demo_realtime.py --data-dir /path/to/AAD --subject S3 --trial 4 --output wav

# 3b) build day — live headset + live mics + live playback:
python demo_realtime.py --data-dir /path/to/AAD --eeg lsl --audio mic --output play
```

All tunables (decision window, ducking dB, gain ramp, hysteresis) live in
`config.py`. See `PROJECT.md` for the day-of setup and the honest scope on the
audio front-end, and **[`BUILD_DAY.md`](BUILD_DAY.md)** for exactly what's tested vs.
what to integrate with the ANT Neuro headset on the day.

> **Tested:** the analysis and the replay demo (no hardware). **Needs on-rig
> integration:** live audio out, live mics, and live EEG over LSL — including a short
> decoder **calibration on the ANT Neuro cap** (KU-Leuven-trained weights don't
> transfer channel-for-channel to a different amplifier). See `BUILD_DAY.md`.
