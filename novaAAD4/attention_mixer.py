"""The attention-driven mixer — the 'feedback' half of the system.

Given periodic attention decisions (which talker is attended, and how confidently),
this holds a per-source gain that *glides* toward a target rather than switching
hard. Audio flows through continuously at block rate; decisions only move the targets.

Review fixes applied here:
- **Abstention (finding 1 & 7):** with no valid evidence (invalid window, NaN, or a
  gap below hysteresis and no established choice) the mixer targets a NEUTRAL, balanced
  mix instead of confidently ducking a talker. `set_neutral()` / `set_decision(..., valid=False)`.
- **Minimum audibility (finding 7):** the ducked talker is attenuated, never muted.
- **Output limiting (finding 6):** `process()` applies headroom + a peak limiter and a
  final hard clip, so the returned audio is always within [-1, 1] — summing two
  full-scale sources can no longer exceed float full scale.
"""
from __future__ import annotations
import numpy as np

from config import AUDIO_FS, BLOCK_S, ATTEN_DB, GAIN_RAMP_S, HYSTERESIS


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class AttentionMixer:
    def __init__(self, n_sources: int = 2, fs: int = AUDIO_FS, block_s: float = BLOCK_S,
                 atten_db: float = ATTEN_DB, ramp_s: float = GAIN_RAMP_S,
                 hysteresis: float = HYSTERESIS, ceiling: float = 0.98):
        self.n = n_sources
        self.duck = db_to_lin(-abs(atten_db))          # linear gain for ignored talkers
        self.neutral_gain = 1.0 / n_sources            # balanced mix that cannot clip
        self.gain = np.full(n_sources, self.neutral_gain)   # start neutral, not A-favoring
        self.target = np.full(n_sources, self.neutral_gain)
        self.attended: int | None = None               # None == abstaining / no decision
        self.hysteresis = hysteresis
        self.alpha = 1.0 - np.exp(-block_s / max(ramp_s, 1e-6))
        # peak limiter state
        self.ceiling = ceiling
        self._lim = 1.0
        self._lim_rel = 1.0 - np.exp(-block_s / 0.20)  # ~200 ms release

    # ---- decisions -------------------------------------------------------------
    def set_neutral(self) -> None:
        """Abstain: glide toward a balanced, both-audible mix; forget the choice."""
        self.attended = None
        self.target = np.full(self.n, self.neutral_gain)

    def set_decision(self, corrs, valid: bool = True) -> int | None:
        """Update targets from per-source scores.

        Returns the attended index, or None if abstaining. With ``valid=False`` or
        non-finite/empty scores the mixer abstains (neutral) rather than guessing.
        """
        corrs = np.asarray(corrs, dtype=float)
        if (not valid) or corrs.size != self.n or not np.all(np.isfinite(corrs)):
            self.set_neutral()
            return None
        best = int(np.argmax(corrs))
        if self.attended is None:
            # need a clear margin over the runner-up to leave abstention
            order = np.sort(corrs)
            if order[-1] - order[-2] > self.hysteresis:
                self.attended = best
            else:
                self.set_neutral()
                return None
        elif best != self.attended and corrs[best] - corrs[self.attended] > self.hysteresis:
            self.attended = best                        # confident enough to switch
        self.target = np.full(self.n, self.duck)
        self.target[self.attended] = 1.0                # attended at 0 dB
        return self.attended

    # ---- audio -----------------------------------------------------------------
    def process(self, blocks) -> np.ndarray:
        """Mix one block per source with the current (ramping) gains, limited to [-1,1]."""
        self.gain += self.alpha * (self.target - self.gain)
        out = np.zeros_like(np.asarray(blocks[0], dtype=np.float32))
        for i, b in enumerate(blocks):
            out += self.gain[i] * np.asarray(b, dtype=np.float32)
        # peak limiter: instantaneous attack, slow release, hard-clip safety net
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak * self._lim > self.ceiling:
            self._lim = self.ceiling / max(peak, 1e-9)      # attack
        else:
            self._lim += self._lim_rel * (1.0 - self._lim)  # release toward unity
            self._lim = min(self._lim, 1.0)
        out = np.clip(out * self._lim, -1.0, 1.0)           # guaranteed bounded output
        return out

    @property
    def gains_db(self):
        return [20.0 * np.log10(max(g, 1e-6)) for g in self.gain]
