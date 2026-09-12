# FIXLOG — where the audit fixes live now

The `NOVA2026_Current_Issues.pdf` audit is resolved in **`nova2026.auditory`**, which
novaAAD v5 now **bundles** under `src/` (vendored from NOVA2026 `audio_v2` @ `8cbc503`;
see `VENDORED.md`). The fixes live in the engine, not in novaAAD's own code. In this
repo, `PYTHONPATH=src:. python -m pytest scripts/auditory/tests/test_auditory.py
scripts/auditory/tests/test_report_fixes.py` runs **37 passing** engine tests
(`test_live_lsl.py` needs a live LSL outlet). Mapping, for reference:

| Audit finding | Where it's fixed in the engine |
|---|---|
| ISS-01 / ISS-17 confidence gate | `auditory.config.MIN_MARGIN` (0.5) + `AttentionController` re-checks every result, 3-window dwell, neutral on weak evidence |
| ISS-02 / ISS-05 live path | `scripts.auditory.live` on `StreamSession`/`Acquire`; `auditory.streaming.AuditoryProcessor` uses the current streaming primitives |
| ISS-03 / ISS-04 / ISS-23 sync | `auditory.timing.TimestampedAudio` maps DAC block positions to LSL time; 30 ms residual profile check; monotonic only for replay scheduling |
| ISS-06 envelope provider / runnable | `TimestampedAudio.align`; runnable `scripts.auditory.live` and `replay` |
| ISS-07 / ISS-08 evaluation | randomizes real candidate columns pre-scoring; binomial intervals + abstention + window counts; no forced zero-tie; no fixed 0.58 |
| ISS-09 accuracy claims | unsupported calibration/generic claims excluded; no unreproduced figure used as evidence |
| ISS-10 / ISS-11 model IO | `.npz` only, `.npy` rejected; `RidgeDecoder.load` for `--base` with contract check |
| ISS-12 / ISS-14 / ISS-15 / ISS-18 contract | labels/count/types/units checked; channels by name; full preprocessing + weight-shape contract; incompatible contracts fail before audio starts |
| ISS-13 assembler | current `CircularBuffer` emits every completed hop; 700-row-chunk regression |
| ISS-16 gains time | `AttentionController` requires finite source time |
| ISS-19 tests | null tests span seeds; chunk tests exceed the hop |
| ISS-20 duplication | one maintained auditory package; novaAAD **removed its own duplicate engine** and now bundles the canonical one as a pinned snapshot (VENDORED.md), not a divergent fork |
| ISS-21 window | default 5 s; `--history`/`--window`; live must match the stored model |
| ISS-22 lag tail | `RidgeDecoder.design` excludes unavailable lag-tail rows for all scorers |
| ISS-24 float WAV | normalized after finite-value validation (`auditory.audio.read_audio`) |
| ISS-25 stimulus grouping | `evaluation.check_split` rejects shared stimulus groups (curated IDs still needed when filenames hide repeats) |
| ISS-26 / D1–D11 | current-pipeline port + the D-series fixes, per the engine's resolution doc |

## What this repo still owns
The browser demo (`demo/`), the viability figures (`results/`, earlier offline
contract — re-run through `scripts.auditory` for current-pipeline numbers), and the
pitch (`docs/`). Nothing here re-implements the engine.

## The one genuinely open item (both implementations)
The **physical EEG↔audio clock loopback** measurement (ISS-03/04/23 hardware half) can
only be done with the cap and a measured audio-timing profile — software cannot close it.
