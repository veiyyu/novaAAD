"""Dataset containers. Labels remain separate from inference windows."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


class AuditoryTrial:
    """A continuous recording with two stable candidate identities."""

    def __init__(
        self,
        eeg,
        timestamps,
        audio,
        audio_rate,
        labels,
        subject,
        trial_id,
        channel_names,
        reference,
        upstream_processing,
        group=None,
    ):
        self.eeg = np.asarray(eeg, dtype=float)
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.audio = np.asarray(audio, dtype=float)
        self.audio_rate = float(audio_rate)
        self.labels = np.asarray(labels, dtype=int)
        self.subject = str(subject)
        self.trial_id = str(trial_id)
        self.channel_names = tuple(channel_names)
        self.reference = reference
        self.upstream_processing = upstream_processing
        self.group = str(group if group is not None else trial_id)
        if self.eeg.ndim != 2 or self.eeg.shape[1] != len(self.channel_names):
            raise ValueError("EEG must have samples by named channels.")
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Channel names must be nonempty and unique.")
        if not all(np.all(np.isfinite(a)) for a in (self.eeg, self.timestamps, self.audio)):
            raise ValueError("Trial arrays must be finite; inject runtime faults after loading.")
        if np.any(np.abs(self.audio) > 1):
            raise ValueError("Trial audio must be normalized to [-1, 1].")
        if len(self.timestamps) != len(self.eeg) or len(self.labels) != len(self.eeg):
            raise ValueError("Each EEG row requires a timestamp and label.")
        if len(self.timestamps) < 2 or not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("EEG timestamps must increase.")
        if self.audio.ndim != 2 or self.audio.shape[1] != 2:
            raise ValueError("Audio must contain exactly two candidate columns.")
        if not np.all(np.isin(self.labels, [-1, 0, 1])):
            raise ValueError("Labels must be unknown (-1), A (0), or B (1).")
        if not np.isfinite(self.audio_rate) or self.audio_rate <= 40:
            raise ValueError("Audio rate must exceed 40 Hz.")

    @property
    def sample_rate(self):
        return 1.0 / np.median(np.diff(self.timestamps))


class AuditoryWindow:
    """Aligned inference data without ground-truth labels."""

    def __init__(
        self,
        eeg,
        envelopes,
        timestamps,
        available_at,
        valid=True,
        reasons=(),
        contract=None,
        segment=0,
    ):
        self.eeg = np.asarray(eeg)
        self.envelopes = np.asarray(envelopes)
        self.timestamps = np.asarray(timestamps)
        self.available_at = float(available_at)
        self.valid = valid
        self.reasons = tuple(reasons)
        self.contract = contract
        self.segment = segment


class AttentionEstimate:
    """Correlations are evidence scores, not probabilities."""

    def __init__(self, scores, evidence_end, emitted_at, valid=True, reasons=()):
        self.scores = None if scores is None else np.asarray(scores, dtype=float)
        self.evidence_end = float(evidence_end)
        self.emitted_at = float(emitted_at)
        self.valid = valid
        self.reasons = tuple(reasons)

    def to_dict(self):
        return {
            "scores": None if self.scores is None else self.scores.tolist(),
            "evidence_end": self.evidence_end,
            "emitted_at": self.emitted_at,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def load_kuleuven(path, stimuli, channel_names, reference, upstream_processing):
    """Read MATLAB v5 trials; require explicit recording metadata from the caller."""
    from .audio import read_audio

    path = Path(path)
    records = loadmat(path, squeeze_me=True, struct_as_record=False)["trials"]
    result = []
    for index, record in enumerate(np.atleast_1d(records)):
        eeg = np.asarray(record.RawData.EegData, dtype=float)
        rate = float(record.FileHeader.SampleRate)
        names = sorted(str(name) for name in np.atleast_1d(record.stimuli))
        if len(names) != 2:
            raise ValueError("Expected two stimulus files per trial.")
        tracks = []
        audio_rate = None
        for name in names:
            matches = list(Path(stimuli).rglob(name))
            if len(matches) != 1:
                raise ValueError(f"Expected one stimulus file for {name}.")
            samples, current_rate = read_audio(matches[0])
            if audio_rate is not None and current_rate != audio_rate:
                raise ValueError("Candidate audio rates differ.")
            audio_rate = current_rate
            tracks.append(samples)
        count = min(len(tracks[0]), len(tracks[1]))
        audio = np.column_stack([tracks[0][:count], tracks[1][:count]])
        target = f"track{int(record.attended_track)}_"
        attended = [i for i, name in enumerate(names) if target in name]
        if len(attended) != 1:
            raise ValueError("Cannot identify the attended track.")
        labels = np.full(len(eeg), attended[0])
        # Names are retained as grouping metadata; repeated excerpts need a curated
        # group manifest before a claim about unseen stories can be made.
        group = "|".join(names)
        trial = AuditoryTrial(
            eeg,
            np.arange(len(eeg)) / rate,
            audio,
            audio_rate,
            labels,
            path.stem,
            index,
            channel_names,
            reference,
            upstream_processing,
            group,
        )
        result.append(trial)
    return result


def load_trial(path):
    """Load the documented numeric interchange format, also used by AASD export."""
    import json

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        return AuditoryTrial(
            archive["eeg"],
            archive["timestamps"],
            archive["audio"],
            metadata["audio_rate"],
            archive["labels"],
            metadata["subject"],
            metadata["trial_id"],
            metadata["channel_names"],
            metadata["reference"],
            metadata["upstream_processing"],
            metadata["group"],
        )


def save_trial(trial, path):
    """Save arrays and JSON without executable objects."""
    import json

    metadata = {
        "audio_rate": trial.audio_rate,
        "subject": trial.subject,
        "trial_id": trial.trial_id,
        "channel_names": trial.channel_names,
        "reference": trial.reference,
        "upstream_processing": trial.upstream_processing,
        "group": trial.group,
    }
    np.savez_compressed(
        path,
        eeg=trial.eeg,
        timestamps=trial.timestamps,
        audio=trial.audio,
        labels=trial.labels,
        metadata=json.dumps(metadata),
    )
