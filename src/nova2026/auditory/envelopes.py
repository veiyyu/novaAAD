"""Continuous speech-envelope extraction with bounded retained state."""

from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt


class EnvelopeExtractor:
    """Rectify speech, smooth and band-limit it, then resample continuously."""

    def __init__(self, audio_rate, config):
        self.audio_rate = audio_rate
        self.config = config
        lowpass = butter(8, 20, fs=audio_rate, output="sos")
        bandpass = butter(3, config.band, btype="bandpass", fs=audio_rate, output="sos")
        self.sos = np.vstack([cast(np.ndarray, lowpass), cast(np.ndarray, bandpass)])
        self.reset()

    def reset(self):
        self.state = np.zeros((len(self.sos), 2, 2))
        self.input_count = 0
        self.output_count = 0
        self.previous = None

    def feed(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("Expected two audio candidate columns.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Audio samples must be finite.")
        if len(samples) == 0:
            return np.empty((0, 2))
        filtered, self.state = sosfilt(self.sos, np.abs(samples), axis=0, zi=self.state)
        # Band-limit before sampling the common grid. Keep the preceding sample
        # so interpolation across a chunk boundary uses the same two endpoints.
        positions = self.input_count + np.arange(len(filtered))
        if self.previous is not None:
            positions = np.concatenate([[self.input_count - 1], positions])
            filtered = np.vstack([self.previous, filtered])
        self.input_count += len(samples)
        last_position = self.input_count - 1
        output_end = (
            int(np.floor(last_position * self.config.sample_rate / self.audio_rate)) + 1
        )
        targets = (
            np.arange(self.output_count, output_end)
            * self.audio_rate
            / self.config.sample_rate
        )
        result = np.empty((len(targets), 2))
        for candidate in range(2):
            result[:, candidate] = np.interp(targets, positions, filtered[:, candidate])
        self.previous = filtered[-1].copy()
        self.output_count = output_end
        return result
