"""The attention-driven mixer — the 'feedback' half of the system.

Given periodic attention decisions (which talker is being attended, and how
confidently), this holds a per-source gain that *glides* toward a target rather
than switching hard. The attended talker sits at 0 dB; the others are ducked by
ATTEN_DB. Gains ramp with a time constant so the mix never jumps, and a small
hysteresis margin stops it flip-flopping on a near-tie.

Audio flows through this continuously at block rate; decisions arrive slowly and
only move the *targets*. That is what decouples the (few-ms) audio latency from
the (few-second) decision latency.
"""
from __future__ import annotations
import numpy as np

from config import AUDIO_FS, BLOCK_S, ATTEN_DB, GAIN_RAMP_S, HYSTERESIS


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class AttentionMixer:
    def __init__(self, n_sources: int = 2, fs: int = AUDIO_FS, block_s: float = BLOCK_S,
                 atten_db: float = ATTEN_DB, ramp_s: float = GAIN_RAMP_S,
                 hysteresis: float = HYSTERESIS):
        self.n = n_sources
        self.duck = db_to_lin(-abs(atten_db))          # linear gain for ignored talkers
        self.gain = np.ones(n_sources)                 # current (ramping) gains
        self.target = np.ones(n_sources)
        self.attended = 0                              # index currently attended
        self.hysteresis = hysteresis
        # per-block smoothing coefficient for an exponential glide to target
        self.alpha = 1.0 - np.exp(-block_s / max(ramp_s, 1e-6))

    def set_decision(self, corrs) -> int:
        """Update the target gains from per-source correlations.

        ``corrs[i]`` is how well the EEG reconstruction matches source i. The best
        source becomes attended, but only if it beats the current one by more than
        the hysteresis margin. Returns the attended index.
        """
        corrs = np.asarray(corrs, dtype=float)
        best = int(np.argmax(corrs))
        if best != self.attended:
            if corrs[best] - corrs[self.attended] > self.hysteresis:
                self.attended = best                   # confident enough to switch
        self.target = np.full(self.n, self.duck)
        self.target[self.attended] = 1.0               # attended talker at 0 dB
        return self.attended

    def process(self, blocks) -> np.ndarray:
        """Mix one block from each source with the current (ramping) gains."""
        self.gain += self.alpha * (self.target - self.gain)
        out = np.zeros_like(np.asarray(blocks[0], dtype=np.float32))
        for i, b in enumerate(blocks):
            out += self.gain[i] * np.asarray(b, dtype=np.float32)
        return out

    @property
    def gains_db(self):
        return [20.0 * np.log10(max(g, 1e-6)) for g in self.gain]
