# Vendored engine — provenance and how to update

novaAAD v5 is **self-contained**: it bundles the NOVA2026 auditory engine so the whole
system runs from this one repository.

## What is vendored

Copied verbatim from **NOVA2026 `audio_v2`** at commit
`13b01f47c49908cb896e4d3dbc397b3a466ce6d3` (see `src/nova2026/VENDOR_COMMIT.txt`).
Previously pinned at `8cbc5031a9d11bfa1003d2a348a6a6fb1b2aa54d`; re-vendored 2026-09-12.

- `src/nova2026/auditory/` — decoder, controller, alignment, timing, audio, evaluation,
  envelopes, pipeline, config (the AAD engine)
- `src/nova2026/streaming/` — Acquire, CircularBuffer, preprocess (repair, quality,
  filters, resample, units), preflight, recording, recovery, offload, spatial, stats,
  window, bootstrap (StreamSession), and now `judges.py` + `preprocess/scope.py`
  (per-channel fault identity behind the new bad-channel policy)
- `src/nova2026/data/pipeline.py`, `src/nova2026/config.py`, `src/nova2026/__init__.py`
- `scripts/auditory/` — the CLIs (convert, train, replay, evaluate, live, demo) + tests
- `scripts/getlive/` — **newly vendored this round.** Cap-agnostic hardware bring-up:
  `probe` (metadata-only LSL outlet inspection — run this first) and the live
  acceptance test (`python -m scripts.getlive`). Resolves the cap contract at run
  time (`--cap auto|ca-208|declared`) instead of a baked-in table, so any cap/amplifier
  the outlet declares can be brought up, including a dry cap with known-dead
  electrodes (`--exclude-channels`, `--max-bad-channels`, `--no-channel-check`).
  Ships its own `README.md` with the rig prerequisites and step-by-step probe → live
  flow — read that before touching a real cap.

## What changed since the last vendor (`8cbc503` → `13b01f4`)

Four upstream commits, all from the same day as this re-vendor:

- `4506de5` — auditory audit fixes: the `evaluation.py` negative control now actually
  decorrelates (block-reordering instead of a 7s roll that was only a phase shift);
  `controller.py` no longer lets a stale failure report clear a decision that newer
  evidence just justified; `timing.py`/`alignment.py` alignment/empty-window edge
  cases; `live.py` gained `--record`/`--subject`/`--session` so a live run is actually
  recorded and closes with its real status instead of that being dead code.
- `909297e` — streaming: per-channel fault identity. A single dead dry electrode used
  to be able to stall an entire live run after ~15s because every quality check
  collapsed "did any channel fault" across all channels. Now `QualityMonitor` tracks
  *which* channel, and a run can tolerate declared-dead electrodes without losing the
  ability to catch a real fault elsewhere. This is what `scripts.auditory.live`'s new
  `--exclude-channels`/`--max-bad-channels`/`--no-channel-check` flags use.
- `8421d65` — **`scripts/getlive` added** (see above). Directly relevant to bringing up
  a physical cap for the first time today.
- `13b01f4` — a `relay.py` addition to `scripts/getlive`.

## Local patches (kept minimal, documented)

1. **`src/nova2026/data/__init__.py`** — trimmed to export only `Pipeline`/`DefaultPipe`
   from `.pipeline`. Upstream also re-exports `.eeg`, which imports **torch** for the
   unrelated EEGNet/PVT side; the auditory path never needs it. This avoids a heavy,
   irrelevant dependency. No behavioural change to the auditory pipeline. Re-applied
   after this re-vendor.

Nothing else is modified. The engine's own logic is unchanged from upstream.

## Verified in this repo (offline, no hardware)

```bash
PYTHONPATH=src:. python -m scripts.auditory.demo            # trains + replays synthetic
PYTHONPATH=src:. python -m pytest scripts/auditory/tests/test_auditory.py \
                                  scripts/auditory/tests/test_report_fixes.py -q   # 44 passed
PYTHONPATH=src:. python -m scripts.getlive.probe --help     # CLI imports and parses cleanly
```

(`test_live_lsl.py` needs a live LSL loopback outlet and is not run here.
`scripts/getlive`'s own test suite, `tests/streaming/test_getlive.py` upstream, is not
vendored — only the runnable package/CLI is. Only import + `--help` were smoke-checked
here; the real acceptance test needs the amplifier, per its own README.)

## Updating the vendored engine

When `audio_v2` advances and you want the newer engine:

```bash
git clone https://github.com/KineticJetIce245/NOVA2026 /tmp/nova && cd /tmp/nova
git checkout audio_v2
# copy the same set back over this repo's src/nova2026 and scripts/auditory + getlive:
rsync -a --delete src/nova2026/auditory src/nova2026/streaming <this-repo>/src/nova2026/
cp src/nova2026/config.py src/nova2026/__init__.py <this-repo>/src/nova2026/
cp -r src/nova2026/data/pipeline.py <this-repo>/src/nova2026/data/
rsync -a --delete scripts/auditory <this-repo>/scripts/
rsync -a --delete scripts/getlive <this-repo>/scripts/
git rev-parse HEAD > <this-repo>/src/nova2026/VENDOR_COMMIT.txt
# then re-apply patch #1 above (data/__init__.py trim) and re-run the tests.
```

Because this is a pinned snapshot, it can lag upstream — treat NOVA2026 `audio_v2` as
the source of truth for the engine and re-vendor deliberately, not silently.
