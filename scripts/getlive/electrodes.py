"""Per-electrode statistics for one run: which electrode saw what.

The acceptance rules need one number per electrode, not the windows
themselves: that is what answers "did every electrode of the cap deliver
signal?". ``ChannelStats`` accumulates peak-to-peak and peak amplitude while
the run streams and reports the flattest and noisiest electrodes at the end.
"""

import math

import numpy as np


class ChannelStats:
    """Per-electrode peak-to-peak and peak-amplitude statistics.

    Fed with every valid window, it answers the bring-up question "which
    electrodes saw no signal, and which are so noisy that the contact is
    suspect?" without keeping the windows themselves.

    Args:
        channels: Channel names in column order, EEG first.
        flat_uv: Mean peak-to-peak below this value counts as flat (a probable
            bad or unconnected electrode).
        noisy_uv: Mean peak-to-peak above this value counts as noisy.

    Notes:
        Both thresholds are prototype heuristics for a first hardware check.
        A flat trace can also be a genuinely quiet electrode on a healthy
        subject, and 50/60 Hz pickup can be removed later by the notch.
    """

    def __init__(
        self,
        channels: tuple[str, ...],
        *,
        flat_uv: float = 1.0,
        noisy_uv: float = 200.0,
    ) -> None:
        """Validate the thresholds and start empty."""

        if not channels:
            raise ValueError("channels cannot be empty.")
        for label, value in (("flat_uv", flat_uv), ("noisy_uv", noisy_uv)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative.")
        if noisy_uv < flat_uv:
            raise ValueError("noisy_uv must not be below flat_uv.")
        self.channels = tuple(str(name) for name in channels)
        self.flat_uv = float(flat_uv)
        self.noisy_uv = float(noisy_uv)
        self.windows = 0
        self._ptp_sum = np.zeros(len(self.channels))
        self._ptp_max = np.zeros(len(self.channels))
        self._abs_max = np.zeros(len(self.channels))

    def update(self, eeg: np.ndarray, eog: np.ndarray | None = None) -> None:
        """Accumulate one window's EEG (and optional auxiliary) columns.

        Raises:
            ValueError: If the column count does not match ``channels``.
        """

        columns = [np.asarray(eeg, dtype=float)]
        if eog is not None and np.asarray(eog).size:
            columns.append(np.asarray(eog, dtype=float))
        block = np.concatenate(columns, axis=1)
        if block.ndim != 2 or block.shape[1] != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} channels, received shape "
                f"{block.shape}."
            )
        if block.shape[0] == 0:
            return
        # Non-finite rows are the Repair stage's business; ignore them here so
        # one broken sample cannot distort an electrode's statistics.
        if not np.isfinite(block).all():
            block = block[np.isfinite(block).all(axis=1)]
            if block.shape[0] == 0:
                return

        peak_to_peak = np.ptp(block, axis=0)
        self._ptp_sum += peak_to_peak
        self._ptp_max = np.maximum(self._ptp_max, peak_to_peak)
        self._abs_max = np.maximum(self._abs_max, np.max(np.abs(block), axis=0))
        self.windows += 1

    def summary(self) -> dict:
        """Return the per-electrode table and the flat/noisy verdicts."""

        if self.windows == 0:
            return {
                "windows": 0,
                "channels": [
                    {
                        "channel": name,
                        "mean_ptp_uv": None,
                        "max_ptp_uv": None,
                        "max_abs_uv": None,
                    }
                    for name in self.channels
                ],
                "flat": (),
                "noisy": (),
            }
        mean = self._ptp_sum / self.windows
        flat = tuple(
            name for name, value in zip(self.channels, mean) if value < self.flat_uv
        )
        noisy = tuple(
            name for name, value in zip(self.channels, mean) if value > self.noisy_uv
        )
        return {
            "windows": self.windows,
            "channels": [
                {
                    "channel": name,
                    "mean_ptp_uv": float(mean[index]),
                    "max_ptp_uv": float(self._ptp_max[index]),
                    "max_abs_uv": float(self._abs_max[index]),
                }
                for index, name in enumerate(self.channels)
            ],
            "flat": flat,
            "noisy": noisy,
        }

    def format_worst(self, count: int = 8) -> str:
        """Render the flattest and the noisiest electrodes."""

        if self.windows == 0:
            return "no valid window reached the per-electrode statistics"
        table = self.summary()["channels"]
        ranked = sorted(table, key=lambda row: row["mean_ptp_uv"])
        flattest = ranked[:count]
        noisiest = list(reversed(ranked[-count:]))
        lines = [f"flattest ({count})      noisiest ({count})"]
        for index in range(len(flattest)):
            low = flattest[index]
            high = noisiest[index]
            lines.append(
                f"  {low['channel']:<6} {low['mean_ptp_uv']:8.2f} uV   "
                f"{high['channel']:<6} {high['mean_ptp_uv']:8.2f} uV"
            )
        return "\n".join(lines)
