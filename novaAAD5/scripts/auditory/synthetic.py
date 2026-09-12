"""Known synthetic EEG/audio relationships for software verification only."""

import numpy as np

from nova2026.auditory.data import AuditoryTrial


def synthetic_trial(identity="one", seconds=24):
    """Generate independent noise and phases for each named fixture."""
    seed = sum((index + 1) * value for index, value in enumerate(identity.encode()))
    random = np.random.default_rng(seed)
    audio_rate = 8000
    audio_time = np.arange(seconds * audio_rate) / audio_rate
    phase_a, phase_b = random.uniform(0, 2 * np.pi, size=2)
    envelope_a = 0.3 + 0.12 * np.sin(2 * np.pi * 2.7 * audio_time + phase_a)
    envelope_b = 0.3 + 0.12 * np.sin(2 * np.pi * 4.1 * audio_time + phase_b)
    audio_a = envelope_a * np.sin(2 * np.pi * 220 * audio_time)
    audio_b = envelope_b * np.sin(2 * np.pi * 340 * audio_time)
    timestamps = np.arange(seconds * 128) / 128
    response = 8 * np.sin(2 * np.pi * 2.7 * (timestamps - 0.15) + phase_a)
    eeg = np.column_stack([response, response * 0.7])
    eeg += random.normal(size=eeg.shape)
    return AuditoryTrial(
        eeg,
        timestamps,
        np.column_stack([audio_a, audio_b]),
        audio_rate,
        np.zeros(len(eeg), dtype=int),
        "synthetic",
        identity,
        ("F3", "F4"),
        "synthetic",
        "none",
        identity,
    )
