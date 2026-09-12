"""Causal SOS filtering: coefficient design and stateful application."""

import math

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


def design_bandpass(
    low: float, high: float, order: int, sfreq: float
) -> np.ndarray:
    """Design a Butterworth band-pass filter in SOS form."""

    # Cutoffs must stay inside the usable spectrum of the given rate.
    if not 0 < low < high < sfreq / 2:
        raise ValueError("Cutoffs must satisfy 0 < low < high < sfreq / 2.")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer.")

    # Butterworth in second-order-section form is numerically stable.
    return butter(order, (low, high), btype="bandpass", fs=sfreq, output="sos")


def design_notch(frequency: float, quality: float, sfreq: float) -> np.ndarray:
    """Design a notch filter in SOS form."""

    if not 0 < frequency < sfreq / 2:
        raise ValueError("Notch frequency must lie below the Nyquist rate.")
    if not math.isfinite(quality) or quality <= 0:
        raise ValueError("quality must be finite and positive.")

    # iirnotch returns (b, a); convert to SOS for the streaming filter.
    b, a = iirnotch(w0=frequency, Q=quality, fs=sfreq)
    return tf2sos(b, a)


class SosFilter:
    """Apply one causal SOS filter continuously across variable-size chunks.

    Args:
        sos: Second-order-section coefficients from ``design_bandpass`` or
            ``design_notch``.
        n_channels: Expected number of data columns.

    Notes:
        Filter state (``zi``) is carried between calls, so the output does not
        depend on how the stream is chopped into chunks, only on sample order.
        ``reset()`` restarts the filter with no history.
    """

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        """Validate geometry and start with no filter state."""

        sos = np.asarray(sos)
        # A second-order section always has six coefficients per row.
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError("SOS coefficients must have shape (n_sections, 6).")
        if (
            isinstance(n_channels, bool)
            or not isinstance(n_channels, int)
            or n_channels < 1
        ):
            raise ValueError("n_channels must be a positive integer.")

        self._sos = sos
        self._channels = n_channels
        self._zi: np.ndarray | None = None  # filter state, None = uninitialized

    def __call__(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter one chunk; timestamps pass through unchanged."""

        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")
        if data.shape[1] != self._channels:
            raise ValueError(
                f"Expected {self._channels} channels, got {data.shape[1]}."
            )
        if timestamps.ndim != 1 or len(timestamps) != len(data):
            raise ValueError("Each sample must have one timestamp.")
        if data.shape[0] == 0:
            return data.copy(), timestamps

        if self._zi is None:
            # Start on a steady-state step so the first chunk has no step
            # response transient.
            initial = sosfilt_zi(self._sos)[:, :, None]
            self._zi = initial * data[0][None, None, :]

        # sosfilt updates zi for the next call: state lives across chunks.
        filtered, self._zi = sosfilt(self._sos, data, axis=0, zi=self._zi)
        return filtered, timestamps

    def reset(self) -> None:
        """Clear the saved filter state."""

        self._zi = None
