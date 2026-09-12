"""Grouped training splits, null controls and duration-based controller metrics."""

import copy

import numpy as np


def check_split(training, validation, testing=()):
    """Reject reused trials or shared stimulus groups across partitions."""
    partitions = [training, validation, testing]
    seen_trials = set()
    seen_groups = set()
    for partition in partitions:
        trials = set()
        groups = set()
        for trial in partition:
            trials.add((trial.subject, trial.trial_id))
            groups.update(trial.group.split("|"))
        if trials & seen_trials or groups & seen_groups:
            raise ValueError("Split reuses a trial or stimulus group.")
        seen_trials.update(trials)
        seen_groups.update(groups)


MISMATCH_BLOCK_SECONDS = 0.25
MISMATCH_SEED = 20260912


def _decorrelated_audio(audio, audio_rate, seed=MISMATCH_SEED):
    """Cut the audio into short blocks and reorder them per candidate.

    A circular roll cannot break the audio/EEG correspondence here: the
    synthetic candidates are narrowband, so rolling a 2.7 Hz envelope by 7 s is
    a phase shift, not a decorrelation (measured correlation 0.81 against a
    margin of 0.5, so the control never fires). Reordering short blocks keeps
    length, amplitude and the local spectrum while destroying the envelope
    timeline the decoder would otherwise match.
    """
    block = max(1, round(MISMATCH_BLOCK_SECONDS * audio_rate))
    count = len(audio) // block
    if count < 2:
        raise ValueError("Audio is too short to decorrelate.")
    random = np.random.default_rng(seed)
    columns = []
    for candidate in range(audio.shape[1]):
        order = random.permutation(count)
        pieces = [
            audio[index * block : (index + 1) * block, candidate] for index in order
        ]
        tail = audio[count * block :, candidate]
        columns.append(np.concatenate(pieces + ([tail] if len(tail) else [])))
    return np.column_stack(columns)


def inject_fault(trial, kind, start=8.0, duration=0.25):
    """Return a copy; original recordings are never changed.

    ``mismatched`` replaces the candidate audio with a block-reordered version
    of itself, so the trial keeps its own signals but loses the audio timeline
    the EEG was recorded against.
    """
    result = copy.deepcopy(trial)
    selected = (result.timestamps >= start) & (result.timestamps < start + duration)
    if kind == "dropout":
        result.eeg[selected] = np.nan
    elif kind == "artifact":
        result.eeg[selected] += 2000
    elif kind == "mismatched":
        result.audio = _decorrelated_audio(result.audio, result.audio_rate)
    elif kind != "none":
        raise ValueError("Unknown fault type.")
    return result


def assert_held_out(trial, model):
    """Library and command-line evaluation share the same leakage guard.

    Fails closed. A model that records no development provenance cannot be
    shown to be held out, so it is refused rather than silently accepted: a
    decoder fitted in-process on this very trial would otherwise pass a check
    advertised as holding for the library and the CLI alike.
    """
    provenanced = False
    for partition in ("training", "validation"):
        entries = list(model.training_info.get(partition, []))
        provenanced = provenanced or bool(entries)
        for subject, identity, group in entries:
            if ((subject, identity) == (trial.subject, trial.trial_id)
                    or set(str(group).split("|")) & set(trial.group.split("|"))):
                raise ValueError("Evaluation overlaps model development data.")
    if not provenanced:
        raise ValueError(
            "Model records no training/validation provenance, so held-out "
            "evaluation cannot be verified."
        )


def selection_metrics(times, choices, labels, *, end_time=None):
    """Durations use each sample's following interval; unknown truth is excluded.

    ``reported_switch_delays_seconds`` measures time to the start of the run
    that holds the new talker for the remainder of the label segment, so a
    one-block coincidence is not scored as an immediate successful switch;
    such a segment counts as ``missed_switches`` instead.
    """
    times = np.asarray(times)
    choices = np.asarray(choices)
    labels = np.asarray(labels)
    if (times.ndim != 1 or not len(times) or choices.shape != times.shape
            or labels.shape != times.shape or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Metrics require matching arrays and increasing finite times.")
    if end_time is None:
        if len(times) < 2:
            raise ValueError("A single block requires its explicit end_time.")
        end_time = times[-1] + times[-1] - times[-2]
    if not np.isfinite(end_time) or end_time < times[-1]:
        raise ValueError("Invalid final interval end.")
    duration = np.diff(times, append=end_time)
    known = np.isin(labels, [0, 1])
    neutral = choices == -1
    correct = known & (choices == labels)
    wrong = known & ~neutral & (choices != labels)
    total = float(duration[known].sum())
    changes = (choices[1:] != choices[:-1]) & (labels[1:] == labels[:-1])
    changes &= known[1:] & known[:-1]
    changes &= (choices[1:] != labels[1:]) & (choices[1:] != -1)
    switch_delays = []
    missed = 0
    switches = []
    previous_label = None
    for index, label in enumerate(labels):
        if label not in (0, 1):
            continue
        if previous_label is not None and label != previous_label:
            switches.append(index)
        previous_label = label
    for position, index in enumerate(switches):
        end = switches[position + 1] if position + 1 < len(switches) else len(times)
        segment = choices[index:end] == labels[index]
        # The delay is the first block from which the new talker is held for the
        # rest of the segment. A single matching block is a flicker, not a
        # switch, and reporting it would make the delay distribution optimistic.
        suffix = np.cumprod(segment[::-1].astype(np.int64))[::-1]
        sustained = np.flatnonzero(suffix)
        if len(sustained):
            switch_delays.append(float(times[index + sustained[0]] - times[index]))
        else:
            missed += 1
    return {
        "known_seconds": total,
        "correct_emphasis_seconds": float(duration[correct].sum()),
        "wrong_suppression_seconds": float(duration[wrong].sum()),
        "neutral_seconds": float(duration[known & neutral].sum()),
        "coverage": float(duration[known & ~neutral].sum() / total) if total else None,
        "false_selection_changes": int(changes.sum()),
        "reported_switch_delays_seconds": switch_delays,
        "missed_switches": missed,
    }
