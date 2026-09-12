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


def inject_fault(trial, kind, start=8.0, duration=0.25):
    """Return a copy; original recordings are never changed."""
    result = copy.deepcopy(trial)
    selected = (result.timestamps >= start) & (result.timestamps < start + duration)
    if kind == "dropout":
        result.eeg[selected] = np.nan
    elif kind == "artifact":
        result.eeg[selected] += 2000
    elif kind == "mismatched":
        result.audio = np.roll(result.audio, round(7 * result.audio_rate), axis=0)
    elif kind != "none":
        raise ValueError("Unknown fault type.")
    return result


def assert_held_out(trial, model):
    """Library and command-line evaluation share the same leakage guard."""
    for partition in ("training", "validation"):
        for subject, identity, group in model.training_info.get(partition, []):
            if ((subject, identity) == (trial.subject, trial.trial_id)
                    or set(str(group).split("|")) & set(trial.group.split("|"))):
                raise ValueError("Evaluation overlaps model development data.")


def selection_metrics(times, choices, labels, *, end_time=None):
    """Durations use each sample's following interval; unknown truth is excluded."""
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
        matches = np.flatnonzero(choices[index:end] == labels[index])
        if len(matches):
            switch_delays.append(float(times[index + matches[0]] - times[index]))
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
