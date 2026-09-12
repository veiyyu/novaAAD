"""Bounded candidate-envelope history, aligned by source time."""

import numpy as np

from .data import AuditoryWindow


class EnvelopeBuffer:
    """Retain only recent envelopes and their actual availability times."""

    def __init__(self, sample_rate, seconds: float = 60):
        self.sample_rate = sample_rate
        self.capacity = round(sample_rate * seconds)
        self.reset()

    def reset(self):
        self.values = np.empty((0, 2))
        self.timestamps = np.empty(0)
        self.availability = np.empty(0)

    def feed(self, values, timestamps, available_at):
        if len(values) == 0:
            return
        values = np.asarray(values, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)
        if values.shape != (len(timestamps), 2):
            raise ValueError("Envelope dimensions do not match timestamps.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("Envelope data must be finite.")
        if not np.isfinite(available_at) or available_at < timestamps[-1]:
            raise ValueError("Envelope availability cannot precede its signal time.")
        if np.any(np.abs(np.diff(timestamps) - 1 / self.sample_rate) > 1e-6):
            raise ValueError("Envelope time grid is discontinuous.")
        if len(self.timestamps):
            interval = timestamps[0] - self.timestamps[-1]
            if abs(interval - 1 / self.sample_rate) > 1e-6:
                raise ValueError("Reset envelope history after an audio discontinuity.")
        self.values = np.concatenate([self.values, values])[-self.capacity :]
        self.timestamps = np.concatenate([self.timestamps, timestamps])[
            -self.capacity :
        ]
        availability = np.full(len(values), available_at)
        self.availability = np.concatenate([self.availability, availability])[
            -self.capacity :
        ]

    def align(self, window):
        if not len(window.timestamps):
            raise ValueError("Alignment requires a nonempty window.")
        reasons = list(window.reasons)
        envelopes = np.zeros((len(window.timestamps), 2))
        available_at = (window.available_at if window.available_at is not None
                        else float(window.timestamps[-1]))
        if len(self.timestamps) < 2 or (
            window.timestamps[0] < self.timestamps[0] - 1e-6
            or window.timestamps[-1] > self.timestamps[-1] + 1e-6
        ):
            reasons.append("audio_unavailable")
        else:
            for candidate in range(2):
                envelopes[:, candidate] = np.interp(
                    window.timestamps, self.timestamps, self.values[:, candidate]
                )
            indices = np.searchsorted(self.timestamps, window.timestamps)
            indices = np.minimum(indices, len(self.timestamps) - 1)
            available_at = max(available_at, float(self.availability[indices].max()))
        return AuditoryWindow(
            window.data,
            envelopes,
            window.timestamps,
            available_at,
            window.valid and not reasons,
            reasons,
            window.contract,
            window.segment,
        )
