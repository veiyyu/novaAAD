"""Ordinary Pipeline stages for aligned auditory inference."""

from nova2026.data.pipeline import Pipeline

from .data import AttentionEstimate


class AuditoryPipeline(Pipeline):
    """Return (AttentionEstimate, original window); invalid data skips decoding."""

    def __init__(self, model, clock):
        super().__init__()
        self.model = model
        self.clock = clock
        self.add_tube(self.predict)

    def predict(self, window):
        scores = None
        if window.valid:
            scores = self.model.score(window)
        lag = self.model.config.lag_samples
        end_index = len(window.timestamps) - lag - 1
        if end_index < 0:
            raise ValueError("Insufficient history for decoder lags.")
        estimate = AttentionEstimate(
            scores,
            window.timestamps[end_index],
            self.clock(),
            window.valid,
            window.reasons,
        )
        return estimate, window
