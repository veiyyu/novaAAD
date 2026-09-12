# novaAAD × NOVA2026 — integration map

How the two codebases fit together. **Target:** run novaAAD's auditory-attention
decoding on NOVA2026's robust EEG plumbing. **Reference:** NOVA2026 `main` @
`da4d4c8` (Sept 2026).

## The core split

The codebases are complementary, not overlapping:

- **NOVA2026** = robust EEG **plumbing + a training scaffold**, with no concept of
  audio or auditory attention.
- **novaAAD** = the **AAD brains + the audio-feedback half**, with lighter,
  homegrown plumbing.

So integration = novaAAD's brains riding on NOVA2026's plumbing. Almost nothing is
thrown away; the pieces slot into different layers.

## Keep from NOVA2026 (don't reinvent)

| NOVA2026 component | Gives you | Replaces (novaAAD) |
|---|---|---|
| `DefaultStream` / `DefaultStreamer` + LSL | amp connect, config validation, manual acquisition | connection half of `eeg_sources.py` |
| `AcquisitionQueue` | validated chunks (timestamp-continuity, value, consumer-lag checks; raises on bad data) | buffering/resampling half of `eeg_sources.py` |
| `ChannelSelectionContract` | maps the amp's channels onto the canonical **64-ch** montage, reorders, drops extras | (novaAAD had no montage contract) |
| `Pipeline` (tube chain) | orchestration: `add_tube`, `feed`/`step`/`rundown` | ad-hoc call flow |
| `DefaultPipe` (MNE) | band-pass + resample + channel picks as a tube | filtering in `decoder.preprocess` (optional) |
| `SupervisedTrainer` + `Metric` + LOSO/bootstrap | cross-validation and confidence intervals | augments `run_viability.py` |

## Bring in from novaAAD (NOVA has none of this)

- `decoder.py` — the linear backward / stimulus-reconstruction decoder. **This is
  the AAD model.** (NOVA's EEGNet is a classifier on EEG windows — a possible
  *future* neural AAD model, not the proven approach here.)
- `envelopes.py` — speech-envelope extraction. Essential; NOVA-less.
- `attention_mixer.py` + `audio_sources.py` — audio steering / ducking with abstain
  + limiter. NOVA has no audio output path at all.
- `evaluate.py` — the null-control honesty harness (zero / shuffle / mismatch must
  hit chance). Wire its metric through NOVA's `Metric`.
- `nova/aad_adapter.py::ReliabilityController` — freshness/abstain → gains.
- `calibrate.py` — per-person, day-of decoder training.
- `ui.html` + `live_demo` — the browser demo, sits on top unchanged.

## The seams (rewritten in `nova/aad_adapter.py`)

1. **Tube, not Streamer.** The AAD step is now `AADTube`, a
   `Callable[[window], (AADResult, window)]` added to a NOVA `Pipeline`.
2. **Validity via exceptions.** NOVA dropped the per-window `valid`/`reasons`
   object; validity is acquisition-time exceptions. The tube abstains on any
   exception + its own short-window / low-margin checks.
3. **Band + rate.** Re-band 128 Hz broadband → 1–9 Hz, resample to 64 Hz, z-score
   (inside `window_to_decoder_input`).
4. **Channel alignment.** Map the decoder's trained channels onto the window's
   `channel_names` by name (via `build_channel_selector`); refuse on mismatch.

## What NOVA genuinely can't give (honest limits)

- **No AAD data.** NOVA's data is PVT vigilance; its new regression trainer
  predicts log reaction time, not envelopes. A working decoder needs two-talker
  AAD recorded on the NOVA rig → the day-of calibration recording is unavoidable.
- **No audio presentation/steering.** 100% novaAAD's contribution.
- **EEGNet ≠ AAD decoder.** It is a softmax classifier; the linear backward
  decoder stays the AAD engine.

## End-to-end flow

**Prep / calibration:** present two talkers over headphones → EEG via
`DefaultStream` → `AcquisitionQueue` (validated) → label the attended talker →
extract both envelopes (`envelopes.py`) → `fit_decoder` on the 64-ch NOVA montage,
saved **with channel names** → optionally score with NOVA's LOSO/bootstrap harness
plus novaAAD's null controls.

**Runtime (live):** `DefaultStream` connects + validates + applies the channel
contract → `AcquisitionQueue.get()` yields validated chunks → `WindowAssembler`
assembles N-second windows → a NOVA `Pipeline` runs
`[optional preprocess tube] → [AADTube]` → `ReliabilityController` re-checks
freshness and outputs per-talker gains → an independent audio thread mixes the two
talkers with those gains (`attention_mixer`) to the headphones; abstain or stale →
neutral, both audible → the browser UI shows the EEG trace and which talker is
amplified.

**Bottom line:** ~60% of novaAAD ports over as-is (all the AAD/audio brains, since
NOVA has no equivalent), the plumbing (`eeg_sources.py`) is *replaced* by NOVA's
more robust acquisition, and the ~3 seam files are rewritten to NOVA's tube API +
64-ch contract. The AAD approach and the accuracy are unchanged — proven brains on
a sturdier chassis.
