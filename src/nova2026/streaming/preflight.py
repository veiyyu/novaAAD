"""Pre-flight source contract: what the outlet must be, and how to reorder it.

Everything that happens *before a single sample is used* lives here: the
canonical channel contract (which columns the chain expects, in which order)
and the metadata checks that verify a connected LSL outlet really matches
that contract. The module answers two questions:

1. ``ChannelContract`` — "given the labels a source declares, how do I map
   its columns into my canonical order?" (validated once, then applied to
   every data block through ``reorder``).
2. ``validate_source`` — "does this connected outlet deliver what I
   declared?" (identity, rate, units, channel types, untouched state).

``prepare`` is the one-call entry point: it runs the metadata checks and
returns the ``ChannelContract`` the caller then uses for the whole run.
"""

import math
from time import monotonic, sleep

import numpy as np


class ChannelContract:
    """Map source data columns into a canonical ``expected_channels`` order.

    The contract is built once, right after the inlet is connected and before
    any data is acquired. It raises at construction when a required channel is
    missing or labels are duplicated, so a misconfigured source fails loudly
    before a run starts instead of silently corrupting recordings.

    Args:
        source_channels: Labels of the connected source, in its column order.
        expected_channels: Required labels, EEG first then auxiliary (e.g.
            EOG), in the exact order the preprocessing chain expects.

    Notes:
        Extra source channels are dropped. ``reorder`` is a column permutation
        and returns the input unchanged when the source already matches the
        expected columns exactly; it never edits the source array.

    Attributes:
        source_channels: Labels of the connected source.
        expected_channels: Canonical labels after ``reorder``.
        selected_channels: Labels kept from the source, in canonical order.
        dropped_channels: Source labels that are not part of the contract.
    """

    def __init__(
        self,
        source_channels: tuple[str, ...],
        expected_channels: tuple[str, ...],
    ) -> None:
        """Validate labels and build the source-to-canonical permutation."""

        source = tuple(str(name) for name in source_channels)
        expected = tuple(str(name) for name in expected_channels)

        # Reject empty label sets before anything else.
        if not source or any(not name for name in source):
            raise ValueError("source_channels must be non-empty strings.")
        if not expected:
            raise ValueError("expected_channels cannot be empty.")

        # Duplicate labels make any name-to-column mapping ambiguous.
        if len(set(source)) != len(source):
            raise ValueError(f"Source channel labels are duplicated: {source!r}")
        if len(set(expected)) != len(expected):
            raise ValueError(
                f"Expected channel labels are duplicated: {expected!r}"
            )

        # Every required channel must exist on the source; fail at setup.
        lookup = {name: index for index, name in enumerate(source)}
        missing = [name for name in expected if name not in lookup]
        if missing:
            raise ValueError(
                f"Source is missing required channels: {missing!r} "
                f"(source has {source!r})."
            )

        # Map each expected channel to its source column. When the source is
        # already exactly the expected columns, no permutation is needed.
        permutation = tuple(lookup[name] for name in expected)
        identity = permutation == tuple(range(len(permutation))) and len(
            source
        ) == len(permutation)

        self.source_channels = source
        self.expected_channels = expected
        self.selected_channels = expected
        self.dropped_channels = tuple(
            name for name in source if name not in set(expected)
        )
        # None means "copy nothing, pass the array through as-is".
        self._indices: tuple[int, ...] | None = None if identity else permutation

    def reorder(self, data: np.ndarray) -> np.ndarray:
        """Return data restricted to and ordered as ``expected_channels``.

        Args:
            data: Samples by channels matching ``source_channels``.

        Returns:
            The same array when no permutation is needed, otherwise a new
            array with the contract columns. Never modifies the input.
        """

        # Column count must match the source the contract was built for.
        if data.ndim != 2:
            raise ValueError("Expected data shaped (samples, channels).")
        if data.shape[1] != len(self.source_channels):
            raise ValueError(
                f"Expected {len(self.source_channels)} source channels, "
                f"received {data.shape[1]}."
            )

        # Identity shortcut: avoid an unnecessary copy on the hot path.
        if self._indices is None:
            return data
        # Take the contract columns in canonical order (drops the extras).
        return np.take(data, self._indices, axis=1)


# Accepted spellings of the source's declared voltage unit, mapped to the
# power of ten of the unit (0 = volts, -3 = mV, -6 = uV, -9 = nV).
_UNIT_ALIASES = {
    "0": 0,
    "v": 0,
    "volt": 0,
    "volts": 0,
    "-3": -3,
    "mv": -3,
    "millivolt": -3,
    "millivolts": -3,
    "-6": -6,
    "uv": -6,
    "microvolt": -6,
    "microvolts": -6,
    "-9": -9,
    "nv": -9,
    "nanovolt": -9,
    "nanovolts": -9,
}


def validate_source(
    stream,
    *,
    sfreq: float,
    channels: tuple[str, ...],
    source_unit_exponent: int = 0,
    n_eeg: int | None = None,
    stream_name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
) -> None:
    """Check a connected inlet's metadata against the run contract.

    Raises:
        RuntimeError: If any source fact does not match the run contract, or
            the outlet already has filters, callbacks or unread samples.
        ValueError: If the expectation arguments themselves are invalid.
    """

    if not getattr(stream, "connected", False) or stream.sinfo is None:
        raise RuntimeError("The source did not connect.")

    if stream_name is not None and stream.name != stream_name:
        raise RuntimeError("The source name does not match the configuration.")
    if source_id is not None and stream.source_id != source_id:
        raise RuntimeError("The source ID does not match the configuration.")
    if stream_type is not None and stream.stype != stream_type:
        raise RuntimeError("The source type does not match the configuration.")

    if not math.isfinite(float(sfreq)) or float(sfreq) <= 0:
        raise ValueError("sfreq must be finite and positive.")
    actual_sfreq = float(stream.info["sfreq"])
    if not math.isclose(actual_sfreq, float(sfreq), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"The source sampling rate ({actual_sfreq:g} Hz) does not match the "
            f"configured rate ({sfreq:g} Hz)."
        )

    dtype = getattr(stream, "dtype", None)
    if dtype is None or not np.issubdtype(np.dtype(dtype), np.number):
        raise RuntimeError("EEG samples must be numeric.")

    if stream.filters or stream.callbacks or getattr(stream, "n_new_samples", 0):
        raise RuntimeError("Source processing or acquisition started before setup.")

    names = tuple(stream.ch_names)
    # Duplicate, empty or missing labels fail here with a RuntimeError, using
    # ChannelContract's single source of truth for the label rules.
    try:
        ChannelContract(names, channels)
    except ValueError as error:
        raise RuntimeError(str(error)) from None

    n_eeg = len(channels) if n_eeg is None else int(n_eeg)
    if not 1 <= n_eeg <= len(channels):
        raise ValueError("n_eeg must be between 1 and the channel count.")
    expected_types = ("eeg",) * n_eeg + ("eog",) * (len(channels) - n_eeg)
    actual_types = tuple(stream.get_channel_types(picks=list(channels)))
    if actual_types != expected_types:
        raise RuntimeError(
            "Source channel types must identify EEG and EOG correctly "
            f"(expected {expected_types}, got {actual_types})."
        )

    units = stream.sinfo.get_channel_units()
    if units is None or len(units) != len(names):
        raise RuntimeError("The source must declare voltage units per channel.")
    for name in channels:
        raw = units[names.index(name)]
        label = str(raw).strip().lower()
        if label not in _UNIT_ALIASES:
            raise RuntimeError(
                f"Missing or unsupported voltage units for {name}: {raw!r}."
            )
        if _UNIT_ALIASES[label] != source_unit_exponent:
            raise RuntimeError(
                f"Source voltage units for {name} do not match the configured "
                f"exponent ({source_unit_exponent})."
            )


def resolve_outlet(
    *,
    name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    resolver=None,
) -> None:
    """Confirm a matching outlet exists BEFORE connecting (B1).

    The full metadata checks (B2, :func:`validate_source`) need a connected
    inlet, but outlet identity is visible without one. Resolving first fails
    fast — "the amplifier is not publishing" — instead of blocking inside
    ``connect()`` on the wrong or absent stream. It checks the outlet's
    ``name`` / ``source_id`` / ``stype`` only; rates, units and channel types
    are still verified after connecting.

    Args:
        name: Expected outlet name, checked when given.
        source_id: Expected outlet source ID, checked when given.
        stream_type: Expected outlet type, checked when given.
        timeout: How long to keep looking before giving up.
        poll_interval: Time between resolution attempts.
        resolver: Optional callable returning the list of available outlets
            (defaults to ``mne_lsl.lsl.resolve_streams``, imported lazily so
            this module stays importable without MNE-LSL). Tests inject a fake.

    Raises:
        RuntimeError: If no matching outlet appears within ``timeout``.
        ValueError: If the expectations are invalid.
    """

    if not name and not source_id:
        raise ValueError("Specify a stream name or source ID.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive.")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be finite and positive.")

    if resolver is None:
        # Lazy import: preflight stays lightweight when only labels are used.
        from mne_lsl.lsl import resolve_streams

        resolver = resolve_streams

    deadline = monotonic() + float(timeout)
    while True:
        for stream in resolver():
            if name is not None and stream.name != name:
                continue
            if source_id is not None and stream.source_id != source_id:
                continue
            if stream_type is not None and stream.stype != stream_type:
                continue
            return  # a matching outlet is publishing
        if monotonic() >= deadline:
            wanted = name or source_id
            raise RuntimeError(
                f"No outlet matching {wanted!r} was found within "
                f"{timeout:g}s."
            )
        sleep(float(poll_interval))


def prepare(
    stream,
    *,
    sfreq: float,
    channels: tuple[str, ...],
    source_unit_exponent: int = 0,
    n_eeg: int | None = None,
    stream_name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
) -> ChannelContract:
    """Run the full pre-flight check and return the run's channel contract.

    This is the one-call start-up step: after ``connect()`` it validates every
    metadata fact (see :func:`validate_source`) and returns the
    ``ChannelContract`` the caller (or ``StreamSession``) uses to reorder
    every incoming block for the rest of the run.

    Returns:
        A contract mapping the connected outlet's columns into ``channels``.

    Raises:
        RuntimeError: On any source mismatch (see :func:`validate_source`).
        ValueError: If the expectation arguments are invalid.
    """

    validate_source(
        stream,
        sfreq=sfreq,
        channels=channels,
        source_unit_exponent=source_unit_exponent,
        n_eeg=n_eeg,
        stream_name=stream_name,
        source_id=source_id,
        stream_type=stream_type,
    )
    # Label rules were just verified, so this construction cannot fail.
    return ChannelContract(tuple(stream.ch_names), tuple(channels))
