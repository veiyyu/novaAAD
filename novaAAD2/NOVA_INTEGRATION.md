# novaAAD — review fixes + NOVA2026 integration

This addresses the 9 Sept 2026 review. It (1) fixes the concrete AAD bugs with tests
that reproduce the review's exact failures and now pass, and (2) provides the layer
that runs the decoder on NOVA2026's real-time streaming. Your KU Leuven code is
untouched — this is a separate `novaAAD/` copy.

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
| 12 — bare weights, no input contract | decoder saved as a **bundle** (channels/rate/band/lags/subject); `reconstruct` **validates channel count** and raises on mismatch | `decoder.py` |

**Honest metrics (finding 1, the important one).** The live demo no longer reports the
biased "correct_frac" (which could read 100% with a dead decoder). `evaluate.py` is the
real measurement: randomized A/B ordering + **asserted null controls** — a zero decoder,
time-shuffled EEG, and mismatched audio must all land near chance or the run *fails*.
`python evaluate.py --data-dir /path/to/AAD` prints the report and raises if any null
control decodes above chance.

## 2. NOVA2026 integration

Run: `PYTHONPATH=. python nova/test_adapter.py` → **5/5 pass** (against NOVA's real
`EEGWindow` class).

- **`nova/aad_adapter.py`**
  - `AADPipeline(decoder, envelopes, in_fs)` — a NOVA `pipeline(window)->AADResult`.
    Reads `window.data` (samples×56, µV, 128 Hz, 1–45 Hz), re-bands to 1–9 Hz, resamples
    to 64 Hz, z-scores, reconstructs, and correlates against candidate envelopes sampled
    on the window's **own timestamps** (fixes the sample-count sync problem, findings 2/3).
    Invalid windows and errors **abstain** instead of guessing.
  - `ReliabilityController` — a NOVA `on_result` sink that produces audio gains and
    **re-checks freshness at the moment of action**: a decision past its expiry decays to
    a neutral mix even if no new result arrives (closes the review's "stale result still
    acts" gap). Neutral = balanced, both talkers audible.
- **`nova/run_adapter_replay.py`** — wires the adapter onto NOVA's `Streamer` + `PlayerLSL`
  replay, with the audio mix on an **independent thread** so slow/lost inference never
  stalls audio. Runnable where NOVA's deps (mne, mne-lsl+liblsl, soxr) are installed.

## 3. The honest last mile (what still needs data / hardware)

This is integration plumbing, verified in logic — not a finished live system. Two things
cannot be produced from what's available:

1. **A decoder trained on NOVA's montage.** NOVA delivers **56 ch @ 128 Hz, 1–45 Hz**; the
   KU Leuven decoder is **64 ch @ 64 Hz, 1–9 Hz**. Weights do **not** transfer across caps
   (the adapter raises if you try, by design). A working decoder needs **two-talker AAD
   recorded on the NOVA rig**, then `fit_decoder` on that montage. NOVA's existing data is
   PVT/rest cognitive-state, not AAD — so there is no NOVA AAD training set yet.
2. **Human-loop validation.** The replay path tests plumbing and control policy, not
   whether steering the audio actually helps a listener. That needs the controlled
   dual-talker switching study from the review (hardware + participants).

So: bugs are fixed and tested; the decoder runs correctly on NOVA's streaming interface;
the remaining work is collecting AAD data on the NOVA cap and the human study.

## 4. Run everything

```bash
PYTHONPATH=. python tests/test_fixes.py          # 8/8  — the bug fixes
PYTHONPATH=. python nova/test_adapter.py         # 5/5  — the NOVA integration logic
python evaluate.py --data-dir /path/to/AAD       # honest accuracy + asserted null controls
```
