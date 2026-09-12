# novaAAD — review fixes + NOVA2026 integration

This (1) fixes the concrete AAD bugs from the 9 Sept 2026 review with tests that
reproduce the review's exact failures, and (2) provides the layer that runs the
AAD decoder on **NOVA2026's current streaming API** (`main` @ `da4d4c8`, the
tube-based `Pipeline` — see below). The KU Leuven viability code is preserved;
behaviour-changing edits are limited to the reviewed bugs.

> **Targets NOVA2026 `main` (Sept 2026).** NOVA rebuilt its streaming layer: the
> old `Streamer` / `StreamConfig` / `EEGWindow` classes are gone, the montage is
> now **64 channels @ 128 Hz**, and real-time EEG flows through a tube `Pipeline`
> fed by an `AcquisitionQueue`. The integration below is written to that API. See
> `INTEGRATION_MAP.md` for the full component-by-component plan.

## 1. Bugs fixed (with test evidence)

Run: `PYTHONPATH=. python tests/test_fixes.py` → **8/8 pass**.

| Review finding | Fix | File |
|---|---|---|
| 5 — stereo PCM scaled wrong | normalize by original dtype *before* down-mix; centre unsigned PCM | `audio_sources.py::_to_mono_float` |
| 1 & 7 — no-evidence still ducks a talker; A-order bias | mixer starts **neutral** and **abstains** without a clear margin; demo randomizes A/B and scores the **raw** decision separately | `attention_mixer.py`, `live_demo.py`, `evaluate.py` |
| 6 — output could clip (peak 2.0) | peak limiter + hard clip → output always in [-1, 1] | `attention_mixer.py::process` |
| 3 — per-chunk resampling corrupted sample counts | one **stateful** `soxr` resampler across chunks | `eeg_sources.py::LSLEEG` |
| 4 — "no data yet" == end-of-stream; fabricated zeros | 0-row array = *waiting*, `None` = *ended*; nothing padded | `eeg_sources.py::LSLEEG` |
| 9 — zero-padded lag tail scored | reconstruction's last `MAX_LAG` samples excluded from scoring | `decoder.py`, `evaluate.py` |
| 12 — bare weights, no input contract | decoder saved as a **bundle** (channels/rate/band/lags/subject/**names**); `reconstruct` **validates channel count** and raises on mismatch | `decoder.py` |

**Honest metrics (finding 1).** The live demo no longer reports the biased
"correct_frac". `evaluate.py` is the real measurement: randomized A/B ordering +
**asserted null controls** — a zero decoder, time-shuffled EEG, and mismatched
audio must all land near chance or the run *fails*.
`python evaluate.py --data-dir /path/to/AAD` prints the report and raises if any
null control decodes above chance.

## 2. NOVA2026 integration (current tube API)

Run: `PYTHONPATH=. python nova/test_adapter.py` → **9/9 pass** (numpy/scipy only;
NOVA need not be installed).

- **`nova/aad_adapter.py`**
  - `AADTube` — a NOVA **tube**, `Callable[[window], (AADResult, window)]`, added
    to a `Pipeline` with `add_tube`. It reads an `AADWindow` (samples×channels,
    NOVA canonical order, 128 Hz, broadband), aligns channels to the decoder's
    trained set **by name**, re-bands to 1–9 Hz, resamples to 64 Hz, z-scores,
    reconstructs, and correlates against candidate talker envelopes sampled on
    the window's **own timestamps**. It abstains on invalid / insufficient /
    low-margin windows and on montage mismatch — and passes the window through
    unchanged so it composes with downstream tubes.
  - `ReliabilityController` — turns `AADResult`s into per-talker audio gains and
    **re-checks freshness at the moment of action**: a decision past its
    `expires_at` decays to a neutral (both-audible) mix even if no new result
    arrives (closes the review's "stale result still acts" gap).
  - `build_channel_selector` / `window_to_decoder_input` — the channel-name
    alignment and the training-time preprocessing transform, re-derived for
    NOVA's rate.
- **`nova/run_adapter_replay.py`** — wires the tube onto NOVA's live path:
  `DefaultStream` → `AcquisitionQueue.get()` → `WindowAssembler` (fixed-length
  windows from arbitrary chunks) → `Pipeline[AADTube]` → `ReliabilityController`,
  with the audio mix on an **independent thread** so slow/lost inference never
  stalls audio. NOVA imports are lazy; three deployment TODOs are marked in-file
  (decoder trained on the rig, live envelope provider, connected LSL stream).

## 3. The honest last mile (what still needs data / hardware)

This is integration plumbing, verified in logic — not a finished live system:

1. **A decoder trained on NOVA's montage.** NOVA now delivers **64 ch @ 128 Hz,
   broadband** — much closer to KU Leuven's 64-ch cap than the old 56-ch layout,
   so alignment is now a channel-**name** mapping (the tube does this) rather than
   a hard mismatch. But weights are not identical across caps/reference/band, so a
   working decoder still needs **two-talker AAD recorded on the NOVA rig**, then
   `fit_decoder` on that montage, saved **with channel names** so the tube can
   align. NOVA's existing data is PVT/rest cognitive-state (its new regression
   trainer predicts log reaction time) — there is no NOVA AAD training set yet.
2. **A live candidate-envelope provider.** `envelopes(t0, t1, n)` must return the
   two talkers' envelopes on the window's timestamp grid, from the audio actually
   presented to the listener.
3. **Human-loop validation.** The replay path tests plumbing and control policy,
   not whether steering the audio helps a listener — that needs the dual-talker
   switching study (hardware + participants).

## 4. Run everything

```bash
PYTHONPATH=. python tests/test_fixes.py          # 8/8  — the bug fixes
PYTHONPATH=. python nova/test_adapter.py         # 9/9  — the NOVA integration logic
python evaluate.py --data-dir /path/to/AAD       # honest accuracy + asserted null controls
```
