# Current-pipeline results (2026-09-12 re-vendor)

`aad_results_revendor.csv` / `results_revendor.json` / `aad_accuracy_curve_revendor.png`
are a re-run of the viability analysis through the **current, re-vendored**
`nova2026.auditory` engine (`audio_v2@13b01f4`), on the real KU Leuven S1/S2 EEG
recordings -- not the old offline Hilbert/zero-phase pipeline that produced the
original `aad_results.csv`/`aad_accuracy_curve.png`.

**Scope, honestly:**
- **S1 and S2 only.** S3 was not re-run (stopped by choice, not by data availability --
  `data/converted/S3/` has all 20 trials ready to go; running it is just
  `python3 tools/loto_accuracy_curve.py` again, no code changes needed).
- **Methodology change from the old CSV:** cross-validation folds are held out by
  `AuditoryTrial.group` (the stimulus-pair filenames) rather than naive per-trial
  leave-one-out, because the current engine's `check_split` (an audit fix, ISS-25)
  rejects a split that shares a stimulus group across train/validation -- the same
  audio pair replayed with the opposite track attended is one group, and leaking it
  would let the decoder key off content instead of attention. This is a *more*
  rigorous methodology than the old numbers used, so the two CSVs are not directly
  comparable data points even where they're close.
- **`check_channels=False`** was used for this offline analysis (see
  `tools/replay_windows_tolerant.py`): the new per-channel bad-electrode policy from
  today's re-vendor is built for live hardware runs, and real recorded EEG has
  ordinary artifacts (blinks, etc.) the old pipeline never screened for either. This
  is the correct setting for offline analysis of already-recorded data, not a
  workaround for anything wrong with the recordings.
- Alpha fixed at 100.0 (per the old config's own note that accuracy is flat across
  low regularization values), not re-tuned per fold.

**Numbers (accuracy %, non-overlapping windows):**

| window (s) | S1 | S2 | mean |
|---|---|---|---|
| 1 | 51.6 | 53.9 | 52.7 |
| 2 | 53.8 | 55.8 | 54.8 |
| 5 | 58.0 | 59.4 | 58.7 |
| 10 | 62.2 | 62.0 | 62.1 |
| 20 | 66.1 | 71.4 | 68.8 |
| 30 | 63.0 | 70.5 | 66.8 |
| 60 | 69.4 | 77.8 | 73.6 |

Shape and magnitude track the old S1/S2 numbers closely (chance at short windows,
~70-80% by 60s), giving real confidence the re-vendored engine reproduces the
underlying effect on real human EEG, not just on synthetic data.

**Tooling added for this, not part of the vendored engine** (see each file's own
docstring for why): `tools/kuleuven_convert_lowmem.py` (memory-safe KU Leuven
conversion -- this box has 3.8GB RAM, the vendored converter needs ~5.5GB for a full
subject), `tools/replay_windows_tolerant.py` (forwards the channel-tolerance knobs
the offline path doesn't expose), `tools/loto_accuracy_curve.py` (the CV sweep
itself, fold-checkpointed since one full subject/window run can exceed a single
shell call's budget on this box).
