"""Map actual audio sample positions to the EEG source clock."""

from threading import Lock

import numpy as np

from .data import AuditoryWindow
from .envelopes import EnvelopeExtractor


def validate_audio_profile(profile, sample_rate, block_size):
    """Validate operator-measured loopback residual and frozen output settings.

    ``residual_offset_seconds`` is a tolerance gate, not a correction. The
    alignment in :class:`TimestampedAudio` never adds it: a run is accepted when
    the operator's measured residual is inside +-30 ms, and that residual is
    then carried in the diagnostics rather than compensated. Applying it would
    need a sign convention that only the loopback measurement itself can settle
    (subtracting the wrong sign doubles the error), so it is deliberately left
    to the operator and recorded, not silently applied.
    """
    required = {"device", "sample_rate", "block_size", "residual_offset_seconds"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("Audio timing profile requires device, sample_rate, block_size, residual_offset_seconds.")
    residual = profile["residual_offset_seconds"]
    if not isinstance(residual, (int, float)) or not np.isfinite(residual) or abs(residual) > .030:
        raise ValueError("Measured residual EEG/audio offset exceeds the 30 ms budget.")
    if profile["sample_rate"] != sample_rate or profile["block_size"] != block_size:
        raise ValueError("Output rate/buffer differs from the calibrated audio path.")
    if not isinstance(profile["device"], str) or not profile["device"].strip():
        raise ValueError("The calibrated output device must be identified by name.")
    return dict(profile)


class TimestampedAudio:
    """Prepared candidates plus observed DAC block timestamps in local LSL time.

    The provider never exposes unplayed samples. Device timestamps, rather than
    a connection epoch or number of requested EEG rows, locate each envelope.
    A missing block or timing discontinuity invalidates overlapping decisions.
    """

    def __init__(self, audio, audio_rate, config, *, tolerance=0.030):
        self.rate = audio_rate
        self.feature_rate = config.sample_rate
        self.values = EnvelopeExtractor(audio_rate, config).feed(audio)
        self.tolerance = tolerance
        self.blocks = []
        self.lock = Lock()
        self.max_error = 0.0

    def record(self, position, frames, audible_at):
        if position < 0 or frames <= 0 or not np.isfinite(audible_at):
            raise ValueError("Invalid audio timestamp record.")
        with self.lock:
            error = 0.0
            if self.blocks:
                previous, count, stamp, _ = self.blocks[-1]
                if position != previous + count or audible_at <= stamp:
                    raise ValueError("Audio sample counter or clock moved backwards/gapped.")
                error = abs(audible_at - stamp - count / self.rate)
            self.max_error = max(self.max_error, error)
            self.blocks.append((position, frames, float(audible_at), error))

    def align(self, window):
        if not len(window.timestamps):
            raise ValueError("Alignment requires a nonempty window.")
        with self.lock:
            blocks = list(self.blocks)
        reasons = list(window.reasons)
        envelopes = np.zeros((len(window.timestamps), 2))
        if not blocks:
            reasons.append("audio_unavailable")
        else:
            positions = np.array([b[0] for b in blocks], float)
            stamps = np.array([b[2] for b in blocks])
            last = blocks[-1]
            positions = np.append(positions, last[0] + last[1] - 1)
            stamps = np.append(stamps, last[2] + (last[1] - 1) / self.rate)
            if window.timestamps[0] < stamps[0] or window.timestamps[-1] > stamps[-1]:
                reasons.append("audio_unavailable")
            else:
                sample_positions = np.interp(window.timestamps, stamps, positions)
                grid = np.arange(len(self.values)) * self.rate / self.feature_rate
                if sample_positions[-1] > grid[-1]:
                    reasons.append("audio_unavailable")
                else:
                    for candidate in range(2):
                        envelopes[:, candidate] = np.interp(sample_positions, grid, self.values[:, candidate])
            if any(b[3] > self.tolerance and window.timestamps[0] - 1 <= b[2] <= window.timestamps[-1]
                   for b in blocks):
                reasons.append("audio_clock_discontinuity")
        # EEGWindow permits an unknown availability time; fall back to the last
        # signal timestamp exactly as EnvelopeBuffer.align does.
        available_at = (window.available_at if window.available_at is not None
                        else float(window.timestamps[-1]))
        return AuditoryWindow(window.data, envelopes, window.timestamps,
                              available_at, window.valid and not reasons,
                              reasons, window.contract, window.segment)

    def diagnostics(self):
        with self.lock:
            blocks = list(self.blocks)
        drift = None
        if len(blocks) > 1:
            elapsed = blocks[-1][2] - blocks[0][2]
            nominal = (blocks[-1][0] - blocks[0][0]) / self.rate
            drift = (elapsed / nominal - 1) * 1e6
        return {"clock_domain": "local_lsl", "max_block_timing_error_seconds": self.max_error,
                "estimated_audio_drift_ppm": drift, "blocks": blocks}
