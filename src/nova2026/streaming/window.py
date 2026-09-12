"""EEGWindow: one processed window with its verdict, for consumers.

A window leaves the ring buffer as a raw tuple; ``EEGWindow`` packages it with
the gating verdict so the consumer never has to re-derive "may I use this and
why". It is a value object: arrays are copied on construction so the window
can safely travel to worker threads.
"""

import numpy as np


class EEGWindow:
    """One output window with its timing, channels and quality decision.

    Args:
        data: EEG samples by channels in microvolts, canonical EEG order.
        eog: Auxiliary samples by channels (zero columns when absent).
        timestamps: Source-time grid of the window.
        valid: Whether the window may enter the consumer.
        reasons: Rejection reasons; empty when valid.
        start_sample: First sample index within the current segment.
        segment: Processing segment; increases after a recovery later.
        channel_names: EEG channel labels matching the ``data`` columns.
        contract: Optional serializable window/processing contract, useful
            for model compatibility checks.
        available_at: Latest source time consumed for this window, when known.
        bad_channels: Labels of the EEG channels a judge found faulty in this
            window. Evidence, not a verdict: a window can carry bad channels
            and still be valid when the run tolerates them (see
            :class:`~.preprocess.QualityMonitor`).

    Notes:
        Arrays are copied, so mutating the inputs afterwards never changes the
        window, and the object is safe to hand to another thread.
    """

    def __init__(
        self,
        data: np.ndarray,
        eog: np.ndarray,
        timestamps: np.ndarray,
        valid: bool,
        reasons: tuple[str, ...],
        start_sample: int,
        segment: int = 0,
        artifact_id: str | None = None,
        channel_names: tuple[str, ...] = (),
        contract: dict | None = None,
        available_at: float | None = None,
        bad_channels: tuple[str, ...] = (),
    ) -> None:
        """Validate shapes and copy every array."""

        data = np.asarray(data)
        eog = np.asarray(eog)
        timestamps = np.asarray(timestamps)

        if data.ndim != 2:
            raise ValueError("data must be samples by channels.")
        if eog.ndim != 2 or eog.shape[0] != data.shape[0]:
            raise ValueError("eog must share the sample count with data.")
        if timestamps.shape != (data.shape[0],):
            raise ValueError("timestamps must match the sample count.")
        if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be a non-negative integer.")

        self.data = np.array(data, copy=True)
        self.eog = np.array(eog, copy=True)
        self.timestamps = np.array(timestamps, copy=True)
        self.valid = bool(valid)
        self.reasons = tuple(str(reason) for reason in reasons)
        self.start_sample = start_sample
        self.segment = int(segment)
        self.artifact_id = artifact_id
        self.channel_names = tuple(str(name) for name in channel_names)
        self.bad_channels = tuple(str(name) for name in bad_channels)
        self.contract = dict(contract) if contract else None
        self.available_at = (
            float(available_at) if available_at is not None else None
        )

    def __len__(self) -> int:
        """Number of samples in the window."""

        return len(self.data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EEGWindow(samples={len(self.data)}, "
            f"eeg={self.data.shape[1]}, eog={self.eog.shape[1]}, "
            f"valid={self.valid}, reasons={self.reasons}, "
            f"bad_channels={self.bad_channels}, "
            f"start={self.start_sample})"
        )
