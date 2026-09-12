"""EOG-guided spatial projection: fit, save/load, and per-window application.

Blinks and eye movements leak a shared spatial direction into frontal EEG
channels. ``fit_ssp`` finds that direction from marked calibration epochs
(EEG correlated with measured EOG) and returns a fixed projector
``P = I - U @ U.T`` that removes it. ``SpatialOperator`` stores the projector
with an explicit processing contract, verifies that contract when it is used,
and applies the correction to EEG columns only — EOG, timing and validity are
never touched.

This is EOG-guided SSP, not ICA or interpolation. It is also a linear
projection: neural activity sharing the removed direction is attenuated too,
and the data rank drops by one per removed component.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .window import EEGWindow


def processing_contract(
    *,
    eeg_channels: tuple[str, ...],
    eog_channels: tuple[str, ...] = (),
    out_sfreq: float,
    units: str = "uV",
    stamp: str = "",
) -> dict:
    """Build the JSON-safe description a projector was calibrated under.

    ``stamp`` is a free-form fingerprint of the preprocessing the calibration
    ran through (for example the notch/band-pass/resampling settings). Two
    runs whose contract dictionaries differ must not share an operator, so
    ``SpatialOperator.validate`` compares whole dictionaries.
    """

    values = {
        "eeg_channels": list(eeg_channels),
        "eog_channels": list(eog_channels),
        "out_sfreq": float(out_sfreq),
        "units": str(units),
        "stamp": str(stamp),
    }
    if not math.isfinite(values["out_sfreq"]) or values["out_sfreq"] <= 0:
        raise ValueError("out_sfreq must be finite and positive.")
    # Normalize through JSON so tuples/lists compare equal and nothing nested
    # can sneak in later.
    return json.loads(json.dumps(values, allow_nan=False))


def cut_epochs(
    eeg: np.ndarray,
    eog: np.ndarray,
    timestamps: np.ndarray,
    event_times,
    *,
    seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut calibration epochs out of already-processed, continuous data.

    This is the calibration data-input boundary for C: it accepts data that
    has ALREADY been processed and hands :func:`fit_ssp` the epochs it needs,
    so whoever produced the processed data (our own offline replay of a
    recorded run, or an external pipeline with an identical contract) can feed
    it in without re-running this package's chain.

    What this interface expects:

    - ``eeg`` / ``eog``: ALREADY-PROCESSED samples, 2-D arrays
      ``(samples, channels)``, float-convertible, columns in the same
      channel order as ``contract`` (EEG channels first for ``eeg``; EOG
      channels for ``eog``), in the same units the contract declares
      (microvolts for our chain). Every row is one instant shared by both
      arrays.
    - ``timestamps``: 1-D ``(samples,)`` source-clock times, one per row,
      strictly increasing. ``event_times`` must use the same clock.
    - ``event_times``: centers of the marked events (e.g. blink centers) to
      cut around, sorted ascending.
    - ``seconds``: length of each epoch. Epochs must fit entirely inside the
      data, otherwise the run is rejected — a silently shortened epoch set
      would corrupt the calibration.

    Returns:
        ``(eeg_epochs, eog_epochs)`` shaped
        ``(events, samples_per_epoch, channels)``, ready for
        :func:`fit_ssp`.

    Raises:
        ValueError: If the inputs are malformed, timestamps/events are not
            ordered or finite, or an epoch would fall outside the data.
    """

    eeg = np.asarray(eeg, dtype=np.float64)
    eog = np.asarray(eog, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if eeg.ndim != 2 or eog.ndim != 2:
        raise ValueError("EEG and EOG must be (samples, channels) arrays.")
    if eeg.shape[0] != eog.shape[0]:
        raise ValueError("EEG and EOG must share the same sample count.")
    if timestamps.ndim != 1 or len(timestamps) != eeg.shape[0]:
        raise ValueError("Each sample row needs one timestamp.")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive.")
    samples = eeg.shape[0]
    if samples < 2:
        raise ValueError("Need at least two samples to calibrate.")

    diffs = np.diff(timestamps)
    if not np.all(np.isfinite(timestamps)) or np.any(diffs <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")
    interval = float(np.median(diffs))
    rows = int(round(seconds / interval))
    if rows < 1:
        raise ValueError("seconds is shorter than one sample interval.")

    events = np.asarray(event_times, dtype=np.float64).ravel()
    if events.size == 0:
        raise ValueError("Provide at least one event time.")
    if not np.all(np.isfinite(events)):
        raise ValueError("Event times must be finite.")
    if np.any(np.diff(events) <= 0):
        raise ValueError("Event times must be strictly increasing.")

    eeg_epochs, eog_epochs = [], []
    for event in events:
        center = int(np.argmin(np.abs(timestamps - event)))
        start = center - rows // 2
        stop = start + rows
        if start < 0 or stop > samples:
            raise ValueError(
                f"An epoch of {seconds:g}s around {event:.3f}s falls outside "
                f"the data ({samples} samples); provide more margin."
            )
        eeg_epochs.append(eeg[start:stop])
        eog_epochs.append(eog[start:stop])

    return np.stack(eeg_epochs), np.stack(eog_epochs)


def fit_ssp(
    eeg: np.ndarray,
    eog: np.ndarray,
    contract: dict,
    n_components: int = 1,
    *,
    min_events: int = 6,
    min_seconds: float = 0.5,
    min_eog_std_uv: float = 1.0,
    max_heldout_coupling_ratio: float = 0.5,
) -> "SpatialOperator":
    """Fit an EOG-guided projector from marked, non-overlapping epochs.

    Args:
        eeg: Epochs by samples by EEG channels, in uV, after the same causal
            preprocessing the operator will later run under.
        eog: Matching epochs by samples by EOG channels, also in uV.
        contract: The processing contract these epochs were produced under
            (see :func:`processing_contract`).
        n_components: Number of EOG-associated spatial directions to remove.
        min_events: Minimum number of calibration events.
        min_seconds: Minimum epoch length in seconds.
        min_eog_std_uv: Required EOG movement in both training and held-out
            sets; a flat set proves nothing.
        max_heldout_coupling_ratio: Highest accepted EEG-EOG coupling after
            projection, relative to before.

    Returns:
        A ``SpatialOperator`` whose report includes the held-out coupling
        ratio, retained energy and the fitted spatial weights.

    Raises:
        ValueError: If the epochs are insufficient, flat, non-finite, or the
            held-out coupling is not reduced enough.
    """

    eeg = np.asarray(eeg, dtype=np.float64)
    eog = np.asarray(eog, dtype=np.float64)
    if eeg.ndim != 3 or eog.ndim != 3 or eeg.shape[:2] != eog.shape[:2]:
        raise ValueError("EEG and EOG need matching (events, samples, channels).")
    eeg_channels = tuple(contract["eeg_channels"])
    eog_channels = tuple(contract["eog_channels"])
    if eeg.shape[2] != len(eeg_channels) or eog.shape[2] != len(eog_channels):
        raise ValueError("Calibration channel counts do not match the contract.")
    minimum = min(len(eog_channels), len(eeg_channels) - 1)
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("n_components must be an integer.")
    if not 1 <= n_components <= minimum:
        raise ValueError(f"n_components must be between 1 and {minimum}.")
    required_samples = round(float(contract["out_sfreq"]) * min_seconds)
    if len(eeg) < min_events or eeg.shape[1] < required_samples:
        raise ValueError(
            f"Provide at least {min_events} non-overlapping events, each at "
            f"least {min_seconds:g} seconds long."
        )
    if not np.all(np.isfinite(eeg)) or not np.all(np.isfinite(eog)):
        raise ValueError("Calibration epochs must be finite.")

    # Mean-subtract each epoch, then keep whole events together: the last
    # events are held out and never used for fitting.
    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    eog = eog - eog.mean(axis=1, keepdims=True)
    split = len(eeg) - max(2, len(eeg) // 3)
    train_eeg = eeg[:split].reshape(-1, eeg.shape[2])
    train_eog = eog[:split].reshape(-1, eog.shape[2])
    test_eeg = eeg[split:].reshape(-1, eeg.shape[2])
    test_eog = eog[split:].reshape(-1, eog.shape[2])

    if np.any(np.std(train_eog, axis=0) < min_eog_std_uv) or np.any(
        np.std(test_eog, axis=0) < min_eog_std_uv
    ):
        raise ValueError(
            "Training and held-out EOG must contain representative movement."
        )

    # EEG directions coupled to EOG: the cross-covariance's left singular
    # vectors. The projector removes the top n_components of them.
    cross = train_eeg.T @ train_eog / len(train_eeg)
    directions, singular_values, _ = np.linalg.svd(cross, full_matrices=False)
    if singular_values[n_components - 1] < 1e-6:
        raise ValueError("No measurable EOG-associated EEG direction was found.")
    basis = directions[:, :n_components]
    matrix = np.eye(eeg.shape[2]) - basis @ basis.T

    corrected = test_eeg @ matrix
    before = float(np.linalg.norm(test_eeg.T @ test_eog))
    after = float(np.linalg.norm(corrected.T @ test_eog))
    ratio = after / before if before > 1e-6 else float("inf")
    if ratio > max_heldout_coupling_ratio:
        raise ValueError(
            "The fitted operator did not sufficiently reduce held-out EOG "
            f"coupling (ratio {ratio:.3f})."
        )

    total_energy = float(np.sum(test_eeg**2))
    report = {
        "method": "EOG-guided SSP via EEG/EOG cross-covariance",
        "training_events": int(split),
        "heldout_events": int(len(eeg) - split),
        "samples_per_event": int(eeg.shape[1]),
        "n_components": int(n_components),
        "heldout_coupling_ratio": ratio,
        "heldout_energy_retained": (
            float(np.sum(corrected**2)) / total_energy if total_energy else 0.0
        ),
        "directions": basis.tolist(),
        "singular_values": singular_values.tolist(),
        "review_note": "Coupling reduction does not prove neural preservation.",
    }
    return SpatialOperator(matrix, contract, report)


class SpatialOperator:
    """One fixed, EEG-only orthogonal projector with its processing contract.

    Args:
        matrix: Square projector over the EEG channels, in canonical order.
        contract: The processing contract (see :func:`processing_contract`).
        report: Calibration measurements kept with the matrix.

    Notes:
        EOG is never projected. Rank reduction is intentional; a future
        covariance classifier must regularize after projection.
    """

    def __init__(self, matrix: np.ndarray, contract: dict, report: dict) -> None:
        """Validate the projector and fix its content identifier."""

        matrix = np.asarray(matrix, dtype=np.float64).copy()
        count = len(contract["eeg_channels"])
        if matrix.shape != (count, count) or not np.all(np.isfinite(matrix)):
            raise ValueError("Projector shape and values do not match EEG channels.")
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("An orthogonal projector must be symmetric.")
        if not np.allclose(matrix @ matrix, matrix, atol=1e-8):
            raise ValueError("The matrix is not an idempotent projector.")
        rank = np.linalg.matrix_rank(matrix, tol=1e-7)
        if not 0 < rank < count:
            raise ValueError("The projector must remove some, but not all, EEG directions.")

        self.matrix = matrix
        self.matrix.setflags(write=False)
        self.contract = json.loads(json.dumps(contract, allow_nan=False))
        self.report = json.loads(json.dumps(report, allow_nan=False))
        encoded = json.dumps(self.contract, sort_keys=True).encode()
        self.artifact_id = hashlib.sha256(matrix.tobytes() + encoded).hexdigest()

    def validate(self, contract: dict) -> None:
        """Reject a run whose preprocessing contract differs from the operator's."""

        expected = json.loads(json.dumps(contract, allow_nan=False))
        if expected != self.contract:
            raise ValueError("Artifact operator and run contracts differ.")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Project samples by EEG channels without changing the input."""

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.matrix.shape[0]:
            raise ValueError("Expected samples by the saved EEG channel count.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Do not project non-finite data.")
        return data @ self.matrix

    def apply_window(self, window: EEGWindow) -> EEGWindow:
        """Project one window exactly once; preserve everything else."""

        if window.artifact_id is not None:
            raise ValueError("This window already has a spatial correction.")
        if window.data.shape[1] != self.matrix.shape[0]:
            raise ValueError("Window EEG columns do not match the operator.")
        names = window.channel_names
        if names and tuple(names) != tuple(self.contract["eeg_channels"]):
            raise ValueError("Window channel order does not match the operator.")
        window.data = self.apply(window.data)
        window.artifact_id = self.artifact_id
        return window

    def save(self, path: str | Path) -> None:
        """Save matrix and metadata together; never overwrite an operator."""

        metadata = json.dumps(
            {
                "schema": 1,
                "contract": self.contract,
                "report": self.report,
                "artifact_id": self.artifact_id,
            },
            allow_nan=False,
        )
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream, matrix=self.matrix, metadata=np.array(metadata)
            )

    @classmethod
    def load(cls, path: str | Path) -> "SpatialOperator":
        """Load an operator without pickle and verify its content identifier."""

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("schema") != 1:
                raise ValueError("Unsupported artifact format.")
            operator = cls(
                data["matrix"], metadata["contract"], metadata["report"]
            )
        if operator.artifact_id != metadata["artifact_id"]:
            raise ValueError("The saved artifact identifier does not match its content.")
        return operator
