"""Full-band audio IO and per-sample gain ramps."""

from importlib import import_module
from queue import Empty, Full, Queue

import numpy as np
from scipy.io import wavfile


def read_audio(path):
    rate, samples = wavfile.read(path)
    dtype = samples.dtype
    samples = samples.astype(float)
    if np.issubdtype(dtype, np.unsignedinteger):
        midpoint = (np.iinfo(dtype).max + 1) / 2
        samples = (samples - midpoint) / midpoint
    elif np.issubdtype(dtype, np.signedinteger):
        samples = samples / (np.iinfo(dtype).max + 1)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("Invalid audio file.")
    if len(samples) and np.max(np.abs(samples)) > 1:
        samples /= np.max(np.abs(samples))
    return samples, rate


class AudioMixer:
    """Inputs must be normalized; fixed headroom keeps two full-scale tracks bounded."""

    def __init__(self, sample_rate, ramp_seconds=0.4):
        if sample_rate <= 0 or ramp_seconds <= 0:
            raise ValueError("Rate and ramp duration must be positive.")
        self.sample_rate = sample_rate
        self.ramp_seconds = ramp_seconds
        self.gain = np.ones(2)

    def process(self, samples, target):
        samples = np.asarray(samples, dtype=float)
        target = np.asarray(target, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2 or target.shape != (2,):
            raise ValueError("Expected two audio tracks and two gains.")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(target)):
            raise ValueError("Audio and gains must be finite.")
        if np.any(np.abs(samples) > 1) or np.any(target < 0) or np.any(target > 1):
            raise ValueError("Audio must be normalized and gains between zero and one.")
        if not len(samples):
            return np.empty(0, dtype=np.float32)
        time = (np.arange(len(samples)) + 1) / self.sample_rate
        decay = np.exp(-time / self.ramp_seconds)
        gains = target + (self.gain - target) * decay[:, None]
        self.gain = gains[-1].copy()
        return (0.49 * np.sum(samples * gains, axis=1)).astype(np.float32)


class LatestEstimate:
    """A single producer and single consumer exchange at most one pending result."""

    def __init__(self):
        self.queue = Queue(maxsize=1)

    def put(self, estimate):
        try:
            self.queue.put_nowait(estimate)
        except Full:
            self.get()
            self.queue.put_nowait(estimate)

    def get(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class AudioPlayback:
    """Blocking playback loop to run independently of the EEG processing thread."""

    def __init__(self, controller, handoff, sample_rate, clock):
        self.controller = controller
        self.handoff = handoff
        self.sample_rate = sample_rate
        self.clock = clock
        self.mixer = AudioMixer(sample_rate)
        self.underruns = 0

    def stream(self, blocks, stop_event):
        sd = import_module("sounddevice")

        with sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32"
        ) as output:
            for block in blocks:
                if stop_event.is_set():
                    break
                now = self.clock()
                estimate = self.handoff.get()
                if estimate is not None:
                    self.controller.update(estimate, now)
                samples = self.mixer.process(block, self.controller.gains(now))
                underflowed = output.write(samples)
                self.underruns += int(underflowed)
