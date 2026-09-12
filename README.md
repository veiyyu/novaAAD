# novaAAD v5 — self-contained neuro-steered hearing on the NOVA2026 auditory engine

novaAAD reads a listener's EEG to decide which of two talkers they're attending, and
amplifies that voice. **v5 is self-contained: it bundles the NOVA2026 auditory engine**
(`nova2026.auditory` + `nova2026.streaming`, vendored from the `audio_v2` branch) under
`src/`, so the whole system — data conversion, training, replay, live acquisition,
evaluation, and the browser demo — runs from this one repository.

- **Engine (bundled):** `src/nova2026/auditory` + `src/nova2026/streaming` — the
  `RidgeDecoder`, `AttentionController`, envelope alignment, DAC clock-mapping,
  streaming and evaluation. CLIs in `scripts/auditory`. Provenance and update path in
  `VENDORED.md`.
- **Front-end / story (this repo's own):** `demo/` browser demo + build script,
  `docs/` pitch and build-day guide, `results/` viability figures.

> **Why bundled.** The engine was validated independently (NOVA2026 `audio_v2` resolved
> the same code-review audit — confidence gate, timestamp alignment, real StreamSession
> acquisition, honest evaluation, DAC clock mapping, D1–D11 — with a passing test
> suite). Rather than re-implement it, novaAAD vendors it as a pinned snapshot so the
> repo is one-clone runnable and robust. `FIXLOG.md` maps every audit finding to the
> engine; `VENDORED.md` records the exact commit and the one local patch.

## Quick start (no second repo needed)

```bash
pip install -r requirements.txt         # numpy scipy soxr mne (+ mne-lsl, sounddevice for live)

# 1) end-to-end, no data and no hardware — train + replay synthetic, then build the demo:
PYTHONPATH=src:. python -m scripts.auditory.demo
PYTHONPATH=src:. python -m scripts.auditory.replay --trial output/auditory_demo/test.npz \
        --model output/auditory_demo/decoder.npz --out output/replay
PYTHONPATH=src:. python demo/build_demo.py --estimates output/replay/estimates.json \
        --trial output/auditory_demo/test.npz --out demo/demo.html
# open demo/demo.html in a browser

# 2) real KU Leuven data:
PYTHONPATH=src:. python -m scripts.auditory.convert --kind kuleuven --input data/S1.mat \
        --stimuli data/stimuli --metadata data/recording.json --out data/converted/S1
PYTHONPATH=src:. python -m scripts.auditory.train  --train data/converted/S1_train*.npz \
        --validation data/converted/S1_val.npz --model models/aad.npz --history 5
PYTHONPATH=src:. python -m scripts.auditory.replay --trial data/converted/S1_test.npz \
        --model models/aad.npz --out output/replay
```

Verify the bundled engine:
```bash
PYTHONPATH=src:. python -m pytest scripts/auditory/tests/test_auditory.py \
                                  scripts/auditory/tests/test_report_fixes.py -q   # 37 passed
```

See `INTEGRATION.md` for how the demo maps onto the engine, and `VENDORED.md` for the
engine's provenance and how to refresh it from `audio_v2`.

## Scope / honesty

The engine decodes attention between **two already-separated** speech streams (headphones
or clip-on mics) — it does **not** separate a room microphone into speakers. Human
hearing benefit and real-cap accuracy are not established by the software checks. The
figures in `results/` are from novaAAD's earlier offline pipeline (Hilbert/zero-phase),
a **different feature contract** from the engine's causal envelope — re-run through
`scripts.auditory` to report current-pipeline numbers. The physical EEG↔audio clock
loopback (sync) still requires the cap and a measured audio-timing profile.
