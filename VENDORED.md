# Vendored engine — provenance and how to update

novaAAD v5 is **self-contained**: it bundles the NOVA2026 auditory engine so the whole
system runs from this one repository.

## What is vendored

Copied verbatim from **NOVA2026 `audio_v2`** at commit
`8cbc5031a9d11bfa1003d2a348a6a6fb1b2aa54d` (see `src/nova2026/VENDOR_COMMIT.txt`):

- `src/nova2026/auditory/` — decoder, controller, alignment, timing, audio, evaluation,
  envelopes, pipeline, config (the AAD engine)
- `src/nova2026/streaming/` — Acquire, CircularBuffer, preprocess (repair, quality,
  filters, resample, units), preflight, recording, recovery, offload, spatial, stats,
  window, bootstrap (StreamSession)
- `src/nova2026/data/pipeline.py`, `src/nova2026/config.py`, `src/nova2026/__init__.py`
- `scripts/auditory/` — the CLIs (convert, train, replay, evaluate, live, demo) + tests

## Local patches (kept minimal, documented)

1. **`src/nova2026/data/__init__.py`** — trimmed to export only `Pipeline`/`DefaultPipe`
   from `.pipeline`. Upstream also re-exports `.eeg`, which imports **torch** for the
   unrelated EEGNet/PVT side; the auditory path never needs it. This avoids a heavy,
   irrelevant dependency. No behavioural change to the auditory pipeline.

Nothing else is modified. The engine's own logic is unchanged from upstream.

## Verified in this repo (offline, no hardware)

```bash
PYTHONPATH=src:. python -m scripts.auditory.demo            # trains + replays synthetic
PYTHONPATH=src:. python -m pytest scripts/auditory/tests/test_auditory.py \
                                  scripts/auditory/tests/test_report_fixes.py -q   # 37 passed
```

(`test_live_lsl.py` needs a live LSL loopback outlet and is not run here.)

## Updating the vendored engine

When `audio_v2` advances and you want the newer engine:

```bash
git clone https://github.com/KineticJetIce245/NOVA2026 /tmp/nova && cd /tmp/nova
git checkout audio_v2
# copy the same set back over this repo's src/nova2026 and scripts/auditory:
rsync -a --delete src/nova2026/auditory src/nova2026/streaming <this-repo>/src/nova2026/
cp src/nova2026/config.py src/nova2026/__init__.py <this-repo>/src/nova2026/
cp -r src/nova2026/data/pipeline.py <this-repo>/src/nova2026/data/
cp -r scripts/auditory <this-repo>/scripts/
git rev-parse HEAD > <this-repo>/src/nova2026/VENDOR_COMMIT.txt
# then re-apply patch #1 above (data/__init__.py trim) and re-run the tests.
```

Because this is a pinned snapshot, it can lag upstream — treat NOVA2026 `audio_v2` as
the source of truth for the engine and re-vendor deliberately, not silently.
