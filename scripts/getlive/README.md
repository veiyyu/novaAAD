# getlive: does the streaming package work with the real cap?

`nova2026.streaming` was validated against PlayerLSL and recorded data only.
This folder is the hardware counterpart. It answers two questions on the rig:

1. Is the amplifier publishing LSL, and what does the outlet declare?
2. Does the package's preprocessing chain run on that outlet for a whole
   session, and does every electrode of the cap deliver signal?

It is **not tied to one cap**. The channel contract is resolved at run time:
when the outlet publishes exactly the electrodes of a datasheet profile (the
ANT Neuro **waveguard original CA-208** today, 64 channels on an **EEgo EE-22x**
amplifier) that profile is used and its order, reference and ground are checked;
otherwise the contract is built from the montage the outlet declares, at any
electrode count, and the run says so. Known-dead electrodes, which a dry cap
always has, can be excluded from the fault verdict while still being recorded.

## Rig prerequisites

1. Amplifier connected over USB, cap mounted, control software open and
   recording.
2. **Application options -> Network Operation -> tick `Enable LSL EEG
   streaming`.** Without it nothing is published on the network and every
   script here fails at resolution. (Tick `Enable Network Events` only if you
   also need LSL *markers*; this folder does not.)
3. Sampling rate and gain are set in the control software and are **not**
   advertised over LSL. Note them; the rate goes into `--sfreq`.
4. Montage: whichever cap is on the head, with its reference and ground.
5. LSL is multicast: the host must be on the same network as the amplifier
   PC, and a firewall that blocks multicast blocks LSL. Run the probe first.

## Step 1: probe the outlet

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive.probe
.venv/Scripts/python.exe -B -m scripts.getlive.probe --json records/probe.json
.venv/Scripts/python.exe -B -m scripts.getlive.probe --all --cap declared
```

It lists every LSL outlet, then connects to the EEG-like one and prints the
identity, the declared channel labels, types and units, **the cap profile the
live run would use**, and how the declared montage compares with it. Connecting
is **metadata-only**: no sample is acquired, no thread is started and nothing is
recorded. `--no-connect` stays at the resolve level; `--name` / `--source-id` /
`--stream-type` inspect one outlet explicitly; `--all` inspects every outlet.

Read the declared `sfreq` and `ch_units` from here: they are the values Step 2
must be told.

## Step 2: the live acceptance test

```powershell
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30
```

`--sfreq` and `--source-units` have no defaults on purpose: they are operator
assertions about the amplifier, and the package's pre-flight check compares
them against what the outlet declares. A wrong setting must fail the run, not
pass against a guess. `Ctrl+C` is a normal way to end the run: the report is
still printed and written.

```powershell
# CA-208, all 64 electrodes, 30 s
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV --duration 30

# Another cap: trust what the outlet declares
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 1000 --source-units V `
    --cap declared --duration 30

# Dry cap with two known-dead electrodes
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --exclude-channels Fp1,F7 --duration 60

# Record the raw run for offline replay
.venv/Scripts/python.exe -B -m scripts.getlive --sfreq 500 --source-units uV `
    --record records --subject S01 --session dry --out records/acceptance.json
```

| Option | Meaning |
| --- | --- |
| `--sfreq`, `--source-units` | Required. Declared rate (Hz) and unit (`V`, `mV`, `uV`, `nV`). |
| `--duration` | Seconds to stream; default 30. |
| `--cap auto\|ca-208\|declared` | `auto` (default): the CA-208 contract when the declared labels cover it, else the declared montage. `ca-208`: demand the datasheet. `declared`: always trust the outlet. |
| `--channels cap\|model` | `cap` (default): every EEG electrode of the resolved cap. `model`: the electrodes the offline classifier uses that this cap also has. |
| `--eog eog\|drop` | `eog` (default): keep the cap's EOG electrode as auxiliary. `drop`: exclude it. |
| `--exclude-channels` | Comma-separated EEG labels known to be dead. They are still recorded; they never reject a window. |
| `--max-bad-channels N` | Faulting EEG channels tolerated per fault type and window before quality rejects it. Must be below the EEG electrode count. |
| `--no-channel-check` | Record channel faults but never let them reject a window. |
| `--expect-channels N` | With no declared stream type, treat an outlet with at least this many channels as the amplifier. |
| `--stream-name`, `--source-id`, `--stream-type` | Pin the outlet when several publish. |
| `--notch` | 60 Hz default; use `--notch 50` on 50 Hz mains. |
| `--resample-quality` | `LQ` default (lowest latency); `auto` measures the presets. |
| `--reference`, `--ground` | Provenance strings; default to what the resolved profile asserts. |
| `--record` with `--subject/--session/--run` | Write the raw run (SQLite + FIF) under that root. |
| `--out` | Write the acceptance report as JSON. |
| `--quiet` | No per-window lines. |

### Which cap contract is used

| `--cap` | Behaviour when the outlet declares ... |
| --- | --- |
| `auto` (default) | every electrode of the CA-208 datasheet: use it, and check order, reference and ground. Anything else: build the contract from the declared labels and types, and say so. |
| `ca-208` | anything short of the full CA-208 contract: fail with the missing electrodes named. |
| `declared` | always build the contract from the declared labels and types. |

A **declared** contract classifies each channel as EEG or auxiliary:

1. a name with an auxiliary prefix (`EOG`, `EKG`, `ECG`, `EMG`, `RESP`, `GSR`,
   `Temp`, `Status`, `TRIG`, `Marker`, `Misc`, ...) is auxiliary, even when the
   outlet declares it as EEG — control software routinely labels the EOG drop
   lead as EEG;
2. a channel declared `eog` is the EOG auxiliary;
3. everything else declared `eeg`, or with no usable type, is EEG;
4. any other declared type (ECG, EMG, stim, misc) is auxiliary but not EOG, so
   it is reported and dropped rather than contracted.

`--channels model` intersects the cap with the offline classifier's electrode
list, so on a cap other than CA-208 it usually covers only part of it; the
`model_channels` check reports exactly how much and names what is missing.

### What runs

Same chain as `scripts/streaming_demo.py`, in this order:

```
Repair (source units, short NaN/gap runs; excluded columns are held, never fatal)
  -> source units to uV
  -> QualityMonitor (observes raw uV: amplitude, saturation, flatline; names the channels)
  -> 60 Hz notch
  -> 1-45 Hz band-pass (3rd order)
  -> SoXR resample to 128 Hz
  -> 2 s windows every 0.5 s, 2 s warm-up, bounded recovery on damage
```

128 Hz / 2 s is the geometry the offline model uses
(`src/nova2026/config.py`); the run reports whether it matches.

### Bad channels on a dry cap

A dry cap has electrodes that are dead for the whole session. The package's
quality monitor detects them per electrode (`flatline` for a dead contact,
`saturation` for a railed one, `amplitude` for a pop or a drift), and by default
a single one of them rejects every window it touches and the run stops after
`persistent_fault_seconds`. That is the wrong default for a dry cap, so this
script offers three ways out:

* `--exclude-channels Fp1,F7` — the operator's assertion "this electrode is
  dead". Still detected and still printed, never counted, never fatal. The same
  list reaches the repair stage, so a column that never reports a value is held
  at its last finite level instead of exhausting the recovery budget.
* `--max-bad-channels N` — tolerate up to N faulting electrodes **per fault
  type**; the N+1-th still rejects the window. Keeps the "electrode fell off
  mid-run" guard alive.
* `--no-channel-check` — record and never reject. The blunt version.

Every faulting electrode is recorded either way: the per-window line carries
`bad=...`, and the report carries `bad_channel_windows` (label -> window count)
plus `held_rows`. That census is what decides the next session's exclusion list.

### The verdict

Three statuses; only `fail` blocks acceptance. Exit codes: `0` accepted,
`1` rejected or failed, `130` stopped with Ctrl+C.

| Check | `fail` when |
| --- | --- |
| `connection` | the run raised |
| `source_rate` | declared rate differs from `--sfreq` |
| `channel_contract` | the outlet publishes fewer channels than the contract needs |
| `data_flow` | under half of `--duration` x `--sfreq` samples arrived |
| `input_lag` | a consumed block was 3 s old (the package's own guard) |
| `windows` | no valid window was produced |
| `resampler` | startup delay above 3 s |
| `electrodes` | **every** electrode is flat |

Everything else that is worth knowing only warns: dropped extra channels,
timestamp gaps, a slow-but-working source, rejected windows (with the reason
histogram), chain restarts, repaired rows, the flattest and noisiest
electrodes, a geometry that differs from the offline model, the channel-quality
policy in force (`channel_scope`), the faulting electrodes that were seen
(`bad_channels`), and a contract that misses electrodes the offline model needs
(`model_channels`). A dead electrode is a gel/contact problem, not a streaming
problem, so it warns - and the report names the electrodes.

The per-electrode summary uses two prototype thresholds (`--flat-uv` 1 uV,
`--noisy-uv` 200 uV mean peak-to-peak). They are heuristics for a first look,
not amplifier specifications. `--out` writes the full per-channel table.

## When something fails

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No LSL outlet appeared` | LSL streaming not enabled, amplifier not connected, wrong network/firewall | Tick `Enable LSL EEG streaming`, re-run the probe |
| `2 outlets could be the amplifier` | two publishers with the same identity (a stale helper process, or a second instance of the control software) | stop the other publisher; LSL cannot tell them apart |
| `Source sampling rate (...) does not match` | amplifier rate differs from `--sfreq` | set the rate in the software, or pass the probed value |
| `units ... do not match` | outlet declares e.g. `-6` while `--source-units V` was passed | pass the units the probe printed |
| `Source channel types must identify EEG and EOG correctly` | the outlet declares the EOG electrode as `eeg` (or a status channel as `eog`) | `--eog drop` to test the EEG electrodes, or `--channels model` |
| `Source is missing required channels` | the montage is neither the CA-208 contract nor the declared profile | compare the probe labels with `cap.py`, or run `--cap declared` |
| `cap profile rejected: ... does not publish the CA-208 contract` | `--cap ca-208` was demanded on another cap | `--cap auto` or `--cap declared` |
| `--exclude-channels names channel(s) that are not EEG electrodes` | typo, or the label only exists on another cap | the error lists the contract's EEG electrodes |
| many windows rejected `amplitude`/`saturation` | wrong gain or wrong `--source-units`, or a saturated electrode | check gain/units; look at the named electrodes |
| windows rejected `flatline`, run stops after ~15 s | a dead electrode (a dry cap always has some) | `--exclude-channels` the one the report names, or `--max-bad-channels` |
| `Input samples are 3 s old` / high `input_lag` | source publishes in large bursts, or the host is busy | reduce the amplifier's LSL chunk size, close other load |
| `no valid window` with `data_flow` ok | warm-up plus resampler delay exceeded the run | stream longer than 10 s |
| flat electrodes | gel/contact, or an electrode not connected | fix contact, re-run; impedance lives in the control software, not in the LSL stream |

## What is still unverified

* The **outlet name, type and channel order are unconfirmed offline**: no eego
  software was available while writing this. The probe exists precisely because
  of that; run it before trusting any default here.
* Only the **CA-208 datasheet profile** is curated in `cap.py`. Any other cap
  runs through the declared profile, where electrode order and completeness
  cannot be checked against a second source - the probe output is the only
  evidence, and the report says which profile was used.
* **Gain and upstream processing** are not advertised over LSL. The software's
  filters/gain settings must be recorded by the operator; the run stores the
  `--reference` / `--ground` / `--upstream` strings as provenance, and they are
  assertions, not measurements.
* The **reference** and **ground** come from the cap profile; the package does
  not re-reference, so whatever the amplifier applied is what the windows
  contain.
* LSL adds network latency and jitter. The run reports both; it does not
  compensate them.
* **No impedance/contact measurement**: that is the control software's job.
* Acceptance thresholds are prototype values. They say "this run behaved", not
  "this amplifier is calibrated".
* Excluding a bad channel from the verdict does **not** remove it from
  `window.data`: the decoder still receives that column, because removing it
  would break the model's channel contract.

## Files

| File | Responsibility |
| --- | --- |
| `cap.py` | cap profiles (CA-208 datasheet, declared), profile selection, channel contracts |
| `outlets.py` | resolve, filter, select, connect and inspect LSL outlets |
| `checks.py` | pure acceptance rules, including the channel-quality policy |
| `electrodes.py` | per-electrode peak-to-peak statistics |
| `probe.py` | the metadata probe CLI |
| `live.py` | the live acceptance run and its channel-quality policy |
| `report.py` | console and JSON rendering |
| `__main__.py` | `python -m scripts.getlive` |

Checks (hardware-free, they cover contracts, selection and scoring):

```powershell
.venv/Scripts/python.exe -B -m unittest tests.streaming.test_getlive -v
```

## References

* `datasets/UDO-SM-0215rev09 CA-208 Datasheet 2020-12-14.pdf`, pages 1-3:
  cap description, layout and pinning (CPz reference, AFz ground, EOG lead).
* `datasets/UDO-SM-0120_ENrev11 eego amplifier EE-22x User Manual
  2025-07-01_02.pdf`, page 6: up to 64 referential channels, sampling rate and
  gain set in the control software.
* [ANT Neuro: Third-Party Integrations with eego Systems](http://academy.ant-neuro.com/blog/application-notes-2/third-party-integrations-with-eegotm-systems-86):
  `Enable LSL EEG streaming`, LSL best practices.
* [MNE-LSL StreamLSL API](https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.stream.StreamLSL.html)
