"""Linear backward (stimulus-reconstruction) decoder for AAD.

The decoder learns ridge-regression weights that reconstruct the attended speech
envelope from time-lagged EEG. At test time it correlates the reconstruction with
each candidate talker's envelope over a sliding window and picks the better match.

Efficiency note: leave-one-trial-out CV is done by precomputing each trial's
covariance (Dáµ€D, Dáµ€y); a fold's training set is then the global sum minus the
held-out trial, so we never rebuild the design matrix per fold.
"""
from __future__ import annotations
import numpy as np

from config import FS, LAG_MIN, LAG_MAX, ALPHA, WINDOWS

LAGS = np.arange(int(round(LAG_MIN * FS)), int(round(LAG_MAX * FS)) + 1)


def design(eeg: np.ndarray) -> np.ndarray:
    """Build the lagged design matrix: (n_times, n_ch * n_lags)."""
    n, nch = eeg.shape
    cols = []
    for lag in LAGS:
        sh = np.roll(eeg, -lag, axis=0)
        if lag > 0:
            sh[-lag:] = 0
        cols.append(sh)
    return np.concatenate(cols, axis=1)


def _trial_cov(eeg: np.ndarray, env: np.ndarray):
    D = design(eeg)
    y = (env - env.mean()) / (env.std() + 1e-9)
    return D.T @ D, D.T @ y


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12
    return float((a * b).sum() / d)


def decode_accuracy(recon, env_a, env_b, win_s, step_s=None):
    """Slide a window; count how often recon matches env_a (the attended) better.

    Returns (n_correct, n_windows). env_a is always the attended talker, so a
    correct decision is one where recon correlates more with env_a than env_b.
    """
    w = int(win_s * FS)
    step = int((step_s if step_s else win_s) * FS)
    if w > len(recon):
        return 0, 0
    correct = total = 0
    for s in range(0, len(recon) - w + 1, step):
        sl = slice(s, s + w)
        correct += int(_corr(recon[sl], env_a[sl]) > _corr(recon[sl], env_b[sl]))
        total += 1
    return correct, total


def run_subject(trials: list[dict], alpha: float = ALPHA):
    """Leave-one-trial-out decoding for one subject.

    Returns (accuracy_by_window, n_windows_by_window), accuracy in [0, 1].
    """
    XtX, Xty = [], []
    for tr in trials:
        A, b = _trial_cov(tr["eeg"], tr["att"])
        XtX.append(A); Xty.append(b)
    XtX_sum = np.sum(XtX, axis=0)
    Xty_sum = np.sum(Xty, axis=0)
    ident = np.eye(XtX_sum.shape[0])

    counts = {w: [0, 0] for w in WINDOWS}
    for i, tr in enumerate(trials):
        w = np.linalg.solve(XtX_sum - XtX[i] + alpha * ident, Xty_sum - Xty[i])
        recon = design(tr["eeg"]) @ w
        for win in WINDOWS:
            c, t = decode_accuracy(recon, tr["att"], tr["unatt"], win)
            counts[win][0] += c; counts[win][1] += t

    acc = {w: (counts[w][0] / counts[w][1] if counts[w][1] else float("nan")) for w in WINDOWS}
    ntot = {w: counts[w][1] for w in WINDOWS}
    return acc, ntot


def fit_decoder(trials: list[dict], alpha: float = ALPHA) -> np.ndarray:
    """Train one decoder on the attended envelopes of all given trials. Returns w."""
    XtX = Xty = None
    for tr in trials:
        A, b = _trial_cov(tr["eeg"], tr["att"])
        XtX = A if XtX is None else XtX + A
        Xty = b if Xty is None else Xty + b
    return np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), Xty)


class RealtimeDecoder:
    """A pre-trained decoder that reconstructs an envelope from an EEG window."""

    def __init__(self, w: np.ndarray):
        self.w = w

    def reconstruct(self, eeg_window: np.ndarray) -> np.ndarray:
        """eeg_window: (n_times, 64) at FS -> reconstructed envelope (n_times,)."""
        return design(eeg_window) @ self.w
