"""Settings shared by auditory training, inference and replay."""

import math

# Conservative for the default five-second windows. A margin is not a probability.
# Short-window null probes exceeded the report's suggested 0.22 starting value.
MIN_MARGIN = 0.5


class AuditoryConfig:
    """Keep feature settings explicit and independent of thread ownership."""

    def __init__(self, sample_rate=64, band=(1.0, 9.0), lag_seconds=0.4):
        self.sample_rate = sample_rate
        self.band = tuple(band)
        self.lag_seconds = lag_seconds
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if not 0 < self.band[0] < self.band[1] < sample_rate / 2:
            raise ValueError("Band must lie below Nyquist.")
        if not math.isfinite(lag_seconds) or lag_seconds < 0:
            raise ValueError("Lag must be finite and nonnegative.")
        self.lag_samples = round(lag_seconds * sample_rate)

    def to_dict(self):
        return {
            "sample_rate": self.sample_rate,
            "band": list(self.band),
            "lag_seconds": self.lag_seconds,
            "envelope_method": "rectify-causal-lowpass8-20-bandpass3-linear-grid-v1",
        }
