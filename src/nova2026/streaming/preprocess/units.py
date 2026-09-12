"""Unit conversion: source units to microvolts."""

import numpy as np


def unit_scaler(source_exponent: int, desired_exponent: int = -6):
    """Return a stateless stage converting source units to ``desired_exponent``.

    Args:
        source_exponent: Power of ten of the source unit, e.g. 0 for volts,
            -6 for microvolts.
        desired_exponent: Power of ten of the output unit; defaults to -6 (uV).

    Returns:
        A stateless ``(data, timestamps) -> (data, timestamps)`` stage.
    """

    # Only the units the pipeline knows how to reason about are allowed.
    if source_exponent not in (0, -3, -6, -9):
        raise ValueError("Supported source exponents are 0, -3, -6, -9.")
    if desired_exponent not in (0, -3, -6, -9):
        raise ValueError("Supported desired exponents are 0, -3, -6, -9.")

    # 10^(source - desired): volts(0) to uV(-6) scales by 1e6, etc.
    factor = 10.0 ** (source_exponent - desired_exponent)

    # The returned stage is a plain scaling closure, applied per data block.
    def scale(data: np.ndarray, timestamps: np.ndarray):
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")
        return data * factor, timestamps

    return scale
