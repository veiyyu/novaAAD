# Audio v2 software verification - September 12 update

The supported path now uses `nova2026.streaming` for auditory training, replay
and live EEG. Old saved models must be retrained. The current streaming suite
passes 181 tests on Windows, including LSL transport and recording. The 38 auditory
tests and 32 legacy streaming tests also pass (251 total). Added auditory
regressions cover the report's latency, faults, confidence, timing, model contracts,
metrics, held-out grouping, worker error and shutdown findings. A real LSL test
publishes EEG with timestamps anchored to paced audio and verifies correct
synthetic candidate scores and fresh evidence.

```powershell
python -B -m unittest discover -s tests/streaming -q
python -B -m unittest discover -s scripts/auditory/tests -q
python -B -m unittest discover -s scripts/dataproc/streaming/tests -q
python -B -m scripts.auditory.demo --out tmp/auditory_validation
python -B -m scripts.auditory.evaluate --trial tmp/auditory_validation/test.npz --model tmp/auditory_validation/decoder.npz --seed 7 --out tmp/auditory_validation/evaluation.json
```

Evaluation now reports counts, abstentions and binomial intervals alongside
controller comparisons. Zero output abstains. Intervals over windows do not
describe independent human participants. No amplifier, microphone, headphone
loopback or human calibration was performed. Device playback requires a measured,
calibration-matched timing profile with residual offset <=30 ms; the software
cannot certify that the supplied measurement was actually performed.

See `documents/AUDIO_V2_ISSUE_RESOLUTION.md` for all 26 report issues and D1-D11.

## Historical audio-branch baseline (superseded)

The following describes the original implementation, not current guarantees.

Verified locally using the existing Windows project environment.

- 12 auditory tests passed, covering chunk invariance, envelope alias rejection,
  neural lag direction, candidate swapping, save/load compatibility, zero evidence,
  stale evidence, audio headroom/scaling, bounded result handoff, independent audio
  expiry, failed EEG runs, KU Leuven MATLAB conversion, AASD manifest mapping, and
  synthetic training-to-mixed-WAV execution.
- All 32 existing streaming tests passed. No production streaming file was changed.
- Ruff check and format checks passed for the new packages.
- Pyright reported zero errors and zero warnings for the new packages.
- The synthetic demonstration trained, saved a model and rendered mixed audio.
- The evaluation command completed 52 controller/fault comparisons.
- The zero-model CLI produced zero coverage, zero wrong-speaker suppression, and
  no selection changes; it did not manufacture successful decoding.
- `uv lock` resolved the declared SoXR dependency and optional sounddevice extra.

The synthetic signal's accuracy is not a human-data result. Default evidence expiry
can produce substantial neutral time because EEG resampling emits batches. This
is visible in coverage metrics and should be measured before changing thresholds.
The implementation has not established a controller advantage on human data.

The AASD adapter was tested using a mocked CNT reader and explicit manifest. The
KU Leuven adapter was tested with a generated MATLAB v5 structure and PCM WAVs.
Neither substitutes for running the real archives. Group repeated excerpts whose
filenames differ before scientific evaluation.

Audio-device calls were exercised with a fake output stream. No amplifier,
microphone, headphone latency, or human calibration was tested. The live EEG
integration adapter exists, but actual audio acquisition and clock mapping require
verified hardware metadata and measured delays. Recorded playback is explicitly
separate from a live participant experiment.

Generated fixtures and reports are under `output/auditory_demo`; they are not
required source files. No existing user outputs were removed and no commit or
push was made.
