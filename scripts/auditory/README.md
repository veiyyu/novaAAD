# Auditory attention

The implementation decodes attention between two available speech streams.
It uses the current `nova2026.streaming` core for training, replay and live EEG.
It does not separate a room microphone
or reuse reaction-time model weights. Human accuracy and hearing benefit have not
been established by the software checks below.

## Run the complete software demonstration

From the repository root, using the existing environment:

```powershell
.venv/Scripts/python.exe -B -m scripts.auditory.demo
.venv/Scripts/python.exe -B -m scripts.auditory.evaluate --trial output/auditory_demo/test.npz --model output/auditory_demo/decoder.npz --out output/auditory_demo/comparisons.json
```

The first command creates independent synthetic fixtures, trains a decoder,
selects regularization using a separate fixture and writes `mixed.wav`,
`decoder.npz`, and `report.json`. This verifies plumbing, not physiological
performance. All synthetic targets currently attend A; candidate-swap and zero
model checks separately exercise label symmetry and abstention.

The second command compares quality-aware, ordinary hysteresis, neutral and oracle
selection. It includes missing EEG, artifacts, mismatched audio, delayed inference,
and audio clock offsets. Margins form a coverage/error curve; do not pick a margin
on the test recording and then report its result as held-out performance.

## Module ownership

Reusable components live in `src/nova2026/auditory`. Dataset conversion and runners
live in `scripts/auditory`. The auditory path uses `AuditoryProcessor`, composed
from the current streaming repair, quality, causal filter, resampler and buffer.
The live entry point acquires through `StreamSession` and `Acquire`.

```text
Recorded audio -> EnvelopeExtractor -> EnvelopeBuffer
Recorded EEG   -> AuditoryProcessor -> EEGWindow
                                   -> aligned AuditoryWindow
                                   -> AuditoryPipeline / RidgeDecoder
                                   -> AttentionEstimate
                                   -> controller -> full-band audio mixer
```

The decoder has no access to ground-truth labels. Candidate columns retain their
source identity; neither column is automatically the attended candidate.
`AuditoryPipeline.rundown(window)` returns `(estimate, original_window)`.
The preprocessing branch is 1-9 Hz at 64 Hz with a causal rectified envelope.
This is a new feature contract, not an exact reproduction of novaAAD's offline
Hilbert/zero-phase processing. Old novaAAD weights must not be loaded.

Audio envelopes use an eighth-order causal 20 Hz low-pass and third-order target
band-pass before sampling a continuous time grid with linear interpolation.
The preceding sample is retained for interpolation across chunk boundaries.
This avoids multi-second audio-to-64-Hz batching. The existing EEG SoXR resampler
is preserved and its availability is still tracked.

EEG quality is checked by the existing processor before filtering. Rejected windows
skip inference. Continuous filters and resamplers retain state. Windows already
contain their complete history, so the auditory code never concatenates overlapping
EEG windows. The backward decoder uses post-stimulus EEG, removing incomplete
trailing lag rows. Training normalization is frozen for inference.

## Training and saved models

```powershell
.venv/Scripts/python.exe -B -m scripts.auditory.train --train data/train01.npz data/train02.npz --validation data/validation01.npz --model models/auditory.npz --history 5
.venv/Scripts/python.exe -B -m scripts.auditory.replay --trial data/test01.npz --model models/auditory.npz --out output/auditory_test
.venv/Scripts/python.exe -B -m scripts.auditory.replay --trial data/test01.npz --model models/auditory.npz --out output/auditory_null --zero-model
```

`scripts.auditory.train.train(training, validation)` is also an ordinary callable.
Training selects nonoverlapping valid windows from the same one-second-hop
processing contract used at inference. Regularization is selected on validation
data. The model is not refitted on validation data. Development trial identities
and groups are recorded, and evaluation rejects overlap with either partition.
Group related repetitions/excerpts manually when their filenames differ. For new
participant claims also make the partitions subject-disjoint; the built-in group
check alone does not enforce that research design.

NPZ models contain numeric weights, normalization and JSON metadata only. Metadata
includes the EEG processing contract, audio feature definition, lags and split.
No executable pickle is used. Saving/loading a model does not perform fitting.

## Dataset conversion

KU Leuven MATLAB v5 input is supported:

```powershell
.venv/Scripts/python.exe -B -m scripts.auditory.convert --kind kuleuven --input data/S1.mat --stimuli data/stimuli --metadata data/recording.json --out data/converted/S1
```

`recording.json` must explicitly contain `channel_names`, `reference`, and
`upstream_processing`. Confirm these against the actual release; do not substitute
the ANT cap order for BioSemi order. The loader assumes EEG is already in uV and
that the release has cropped EEG to the stimulus onset. Confirm those properties
before using the converted trials. Unsupported HDF5 files fail rather than silently
using a dictionary with an incompatible schema.

AASD raw CNT import is manifest-driven because the archive has not been available
locally to validate trigger/file mappings. Supply verified mappings, not guessed
event codes:

```json
{
  "trials": [{
    "eeg_file": "subject01.cnt",
    "start_seconds": 10,
    "stop_seconds": 70,
    "audio_files": ["candidate_a.wav", "candidate_b.wav"],
    "channel_names": ["F3", "F4"],
    "reference": "verified recording reference",
    "upstream_processing": "verified acquisition processing",
    "subject": "subject01",
    "trial_id": "trial01",
    "group": "shared-stimulus-group01",
    "attention_events": [{"seconds": 0, "candidate": 0}, {"seconds": 20, "candidate": 1}],
    "switch_uncertainty_seconds": 0.5
  }]
}
```

Paths are relative to the manifest. EEG is cropped from the continuous CNT file;
MNE's volts are converted to uV. Audio files must be the two separate candidate
signals aligned to the crop. Left/right channels of a spatialized mixture are not
automatically separate speakers. Event times are relative to the crop. A margin
around reported switches is labeled unknown to avoid treating button timing as
exact attention timing. Import with `--kind aasd --input manifest.json --out ...`.
This importer needs validation against the real release before scientific use.

The interchange NPZ format contains `eeg` (samples x channels, uV), `timestamps`
(seconds), `audio` (samples x two candidates, normalized), `labels` (-1/0/1 per EEG
sample), and JSON `metadata`. Audio starts at the first EEG timestamp; real offsets
must be corrected when preparing a trial. Included metadata is documented by
`save_trial` and `load_trial` in `data.py`.

## Timing and sound control

Nominal signal time, resampler availability, and inference emission are separate.
Replay advances a virtual clock, allows bounded waiting for envelope availability,
and rejects unfinished tails. It never flushes a fake end-of-stream tail into a
decision. Device and filter phase delays must be characterized on the intended
hardware. Learned neural lags are not a substitute for measuring clock offsets.

The controller accepts finite evidence only, requires a 0.5 correlation margin
and three consecutive confident windows before committing or switching, and
expires evidence after three seconds. Weak evidence returns to neutral. Scores
are correlations, not calibrated probabilities. Manual selection is explicit.
The simple baseline uses a switching margin and holds its selection; the quality
controller deliberately abstains instead. No probabilistic state model is added.

`AudioMixer` scales normalized full-band inputs with sample-level exponential
ramps and fixed headroom. It does not apply the EEG band to audible speech.
`AudioPlayback` uses a separate blocking audio worker and counts underruns.
Optional playback requires `sounddevice`, which is not installed automatically.
Install the optional audio dependencies with `pip install -e ".[audio]"`.

```python
from scripts.auditory.streamer import AuditoryReplayStreamer

streamer = AuditoryReplayStreamer(trial, model)
streamer.initialize()
streamer.stream()
```

The calling thread owns envelope processing and decoding. The audio thread owns
the controller and mixer. A one-result queue prevents stale results accumulating.
Worker errors reach the caller. Replay playback is not a live-brain experiment.

## Live EEG with timestamped candidate audio

Use `python -B -m scripts.auditory.live --trial candidates.npz --model decoder.npz
--stream OUTLET --output wav --out output/live` on one command line. WAV mode is
paced and tests software timing without sending audio to a device. A run with no
EEG decisions fails. Source labels/types/units are checked before acquisition;
extra channels are selected out by name, never truncated by column count.

`--output play --timing-profile measured.json` uses the sound device's DAC time,
mapped to the local LSL clock on each callback. `timing.json` records audio sample
positions, source-clock timestamps, block timing errors and estimated drift.
Envelope positions come from those records, not requested EEG sample counts.

The profile must match the one stored at calibration and contain exactly
`device` (output device name), `sample_rate`, `block_size`, and
`residual_offset_seconds` (measured remaining EEG/audio offset). The block size is
`round(sample_rate * 0.032)`. Residuals over 0.030 seconds fail startup. This is an
operator-supplied loopback measurement, not a measurement inferred from correlations.
Measure the full playback/EEG path with a loopback signal, record calibration
through the same path, and keep the device, rate and buffering fixed. The decoder's
lag window does not absorb arbitrary offsets. Hardware timing remains unverified
until that measurement is actually made.

Train with `--window 5` (alias for `--history`) and optionally `--base base.npz` for
a contract-checked ridge prior. Use `--timing-profile measured.json` only for
calibration recordings acquired through that measured path. The prior keeps the
base normalization; development provenance is carried into held-out checks.
Live `--window` must match the model. Old models require retraining because the
causal streaming/resampling contract has changed.

No upstream 72%/82% accuracy or short-calibration recommendation is adopted here:
those quantities describe different experiments, and the cited upstream figure
has no reproducible generator. Personalization needs a held-out comparison.

## Evaluation scope

Metrics currently describe commanded selection, not the exact acoustic attenuation
during a ramp. Durations exclude unknown ground truth. Reported switching delays
are relative to the first known label after an uncertainty interval, not the exact
keypress or neural transition time. Failed EEG runs are reported explicitly;
remaining offline audio can still be rendered with the controller's fallback. WAV
rendering reports audio underruns as null because no device was tested. Replayed
EEG cannot respond to changed audio, so this is not evidence of comprehension or
listening-effort improvement.

## Checks

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s scripts/auditory/tests -v
.venv/Scripts/python.exe -B -m unittest discover -s tests/streaming -v
.venv/Scripts/python.exe -B -m unittest discover -s scripts/dataproc/streaming/tests
```

Tests cover candidate ordering, neural lag direction, model persistence/contracts,
chunk invariance, PCM scaling, mixer headroom, invalid/stale evidence, group overlap,
bounded handoff and synthetic training-to-mixed-audio replay. Real AASD/KU Leuven
data, live amplifier transfer and acoustic hardware still require external checks.
