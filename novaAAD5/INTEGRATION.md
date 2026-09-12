# novaAAD v5 — how the demo integrates with the bundled auditory engine

novaAAD no longer ships its own decoder/controller/streaming. It **bundles** the
NOVA2026 auditory engine (`nova2026.auditory` + `nova2026.streaming`, vendored from the
`audio_v2` branch — see `VENDORED.md`) under `src/`. This repo adds only the demo
front-end and the pitch.

## 1. Set up (one clone)

```bash
pip install -r requirements.txt      # numpy scipy soxr mne (+ mne-lsl, sounddevice for the live path)
export PYTHONPATH=src:.               # engine on src/, CLIs at scripts/
```

The bundled `nova2026.auditory` exposes `RidgeDecoder`, `AttentionController`,
`AuditoryConfig`, envelope alignment (`EnvelopeBuffer`, `TimestampedAudio`),
`AudioMixer`, and the `AuditoryProcessor` that runs the `nova2026.streaming` chain. CLIs
live in `scripts.auditory` (`convert`, `train`, `replay`, `evaluate`, `live`, `demo`).
No second repository is required; `VENDORED.md` says how to refresh the engine from
`audio_v2` when it advances.

## 2. Data → model → replay (all in the engine)

```bash
# convert KU Leuven MATLAB v5 to the engine's trial format
python -m scripts.auditory.convert --kind kuleuven --input data/S1.mat \
       --stimuli data/stimuli --metadata data/recording.json --out data/converted/S1

# train (regularization chosen on validation; groups/subjects kept disjoint)
python -m scripts.auditory.train  --train data/converted/S1_train*.npz \
       --validation data/converted/S1_val.npz --model models/aad.npz --history 5

# replay a held-out trial -> estimates.json + mixed.wav + metrics.json
python -m scripts.auditory.replay --trial data/converted/S1_test.npz \
       --model models/aad.npz --out output/replay
# honesty checks:
python -m scripts.auditory.replay --trial data/converted/S1_test.npz \
       --model models/aad.npz --out output/null --zero-model
python -m scripts.auditory.evaluate --trial output/replay/... --model models/aad.npz --out cmp.json
```

## 3. Build the browser demo (this repo)

`demo/build_demo.py` reads the engine's `estimates.json` (and, for the EEG trace, the
replayed trial `.npz`) and writes a **self-contained** `demo/demo.html`:

```bash
python demo/build_demo.py --estimates output/replay/estimates.json \
       --trial data/converted/S1_test.npz --out demo/demo.html
# no engine yet? a UI preview with fabricated data:
python demo/build_demo.py --synthetic --out demo/demo.html
```

Open `demo/demo.html` in any browser — live EEG trace, both talkers' meters, and the
attention needle, played back at 10 Hz. The **decisions come from the engine's
estimates**; the small display controller in `build_demo.py` only mirrors
`nova2026.auditory.AttentionController` (margin 0.5, 3-window dwell, 3 s expiry, 6 dB
duck) so the picture matches what the real controller does.

## Boundary

- Decoding, control, alignment, streaming, evaluation, saved-model format: **engine**
  (`nova2026.auditory`). Do not reintroduce a second copy here.
- Visualization, the product story, and the viability figures: **this repo**.
- Old novaAAD `.npz`/`.npy` weights are **not** loadable by the engine — its feature
  contract (causal rectified envelope, 1–9 Hz @ 64 Hz) differs from the old offline
  Hilbert/zero-phase processing. Retrain through `scripts.auditory.train`.
