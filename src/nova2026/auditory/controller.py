"""Three-state control with evidence expiry independent of inference."""

import math

import numpy as np

from .config import MIN_MARGIN


class AttentionController:
    """Own this component on the audio thread; hand estimates to it via a queue."""

    def __init__(self, margin=MIN_MARGIN, max_age=3.0, attenuation_db=6.0, min_switch_windows=3):
        if not all(math.isfinite(value) for value in (margin, max_age, attenuation_db)):
            raise ValueError("Controller parameters must be finite.")
        if margin <= 0 or max_age <= 0 or not 0 <= attenuation_db <= 20:
            raise ValueError("Invalid controller thresholds.")
        self.margin = margin
        self.max_age = max_age
        self.duck = 10 ** (-attenuation_db / 20)
        self.selected = None
        self.evidence_end = -math.inf
        self.manual = None
        self.manual_enabled = False
        if isinstance(min_switch_windows, bool) or not isinstance(min_switch_windows, int) or min_switch_windows < 1:
            raise ValueError("min_switch_windows must be a positive integer.")
        self.min_switch_windows = min_switch_windows
        self.pending = None
        self.pending_count = 0

    def set_manual(self, enabled, candidate=None):
        if candidate not in (None, 0, 1):
            raise ValueError("Manual candidate must be A, B, or neutral.")
        self.manual_enabled = bool(enabled)
        self.manual = candidate

    def update(self, estimate, now):
        if not math.isfinite(now):
            raise ValueError("Controller requires finite source-clock time.")
        scores = estimate.scores
        # Stale evidence is ignored whether or not it is usable: an old failure
        # report must not clear a decision that newer evidence justified.
        # (A non-finite evidence_end compares false here and falls through to
        # the invalid branch below, which is where it belongs.)
        if estimate.evidence_end <= self.evidence_end:
            return
        if (
            not estimate.valid
            or scores is None
            or scores.shape != (2,)
            or not np.all(np.isfinite(scores))
            or not math.isfinite(estimate.evidence_end)
            or not math.isfinite(estimate.emitted_at)
        ):
            self.selected = None
            self.pending, self.pending_count = None, 0
            return
        age = now - estimate.evidence_end
        if age < 0 or age > self.max_age or estimate.emitted_at > now:
            self.selected = None
            self.pending, self.pending_count = None, 0
            return
        self.evidence_end = estimate.evidence_end
        difference = scores[0] - scores[1]
        if abs(difference) < self.margin:
            self.selected = None
            self.pending, self.pending_count = None, 0
        else:
            candidate = int(difference < 0)
            if self.selected == candidate:
                self.selected = candidate
                self.pending, self.pending_count = None, 0
            else:
                self.pending_count = self.pending_count + 1 if self.pending == candidate else 1
                self.pending = candidate
                if self.pending_count >= self.min_switch_windows:
                    self.selected = candidate
                    self.pending, self.pending_count = None, 0

    def choice(self, now):
        if not math.isfinite(now):
            raise ValueError("Controller requires finite source-clock time.")
        if self.manual_enabled:
            return self.manual
        if now < self.evidence_end or now - self.evidence_end > self.max_age:
            return None
        return self.selected

    def gains(self, now):
        gains = np.ones(2)
        selected = self.choice(now)
        if selected is not None:
            gains[1 - selected] = self.duck
        return gains
