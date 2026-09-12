"""Channel contracts for EEG caps: a datasheet profile, or the outlet's own.

There are two ways to know what a cap's electrodes are, and this module offers
both because they answer different questions:

* A **datasheet profile** is curated: labels, electrode order, reference and
  ground for one cap (the ANT Neuro waveguard original CA-208 today). It can
  check order and completeness, which is what a bring-up test wants.
* A **declared profile** is built from the labels and channel types the outlet
  itself publishes. It works for any cap the amplifier can drive, at any
  electrode count, and cannot check anything beyond self-consistency: there is
  no second opinion to compare against.

``select_profile`` picks between them. ``--cap auto`` uses the datasheet profile
when the outlet's labels cover it and falls back to the declared profile
otherwise; ``--cap ca-208`` demands the datasheet; ``--cap declared`` always
trusts the outlet.

Source of the CA-208 numbers: ``datasets/UDO-SM-0215rev09 CA-208 Datasheet
2020-12-14.pdf``.

* Page 1-2: "waveguard original cap, 64 channels, 10/10, shielded, Tyco 68",
  compatible with the ``EE-22x`` eego amplifier, 64 channels plus CPz
  reference and AFz ground, including one ring electrode as a drop lead for
  the EOG electrode, sintered Ag/AgCl pins.
* Page 2 (cap layout) and page 3 (pinning specification) give the electrode
  order used below: connector 1 carries channel 1..32 (Fp1 .. EOG) and
  connector 2 carries the remaining 32 (AF7 .. Oz). CPz is the reference and
  is not a signal channel; GND (AFz) is the ground.
* Page 2, notes 1 and 2: "Reference at CPz labelled CPz", "Ground at AFz
  labelled GND".

The LSL outlet is published by the eego software, not by this repository, so
the labels it declares are still an operator-verified assumption: the probe in
this folder prints exactly what the outlet declares.
"""

from dataclasses import dataclass

# Reference and ground of the CA-208 cap: assertions, not LSL metadata.
CAP_REFERENCE = "CPz"
CAP_GROUND = "AFz"

# The single auxiliary electrode of the cap (ring electrode drop lead).
CAP_EOG_CHANNEL = "EOG"

# 63 EEG electrodes in cap order: connector 1 (channels 1-31), then the
# electrodes that follow EOG on connector 1 and connector 2 (channels 33-64).
CAP_EEG_CHANNELS = (
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "O2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CP4",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO5",
    "PO3",
    "PO4",
    "PO6",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "Oz",
)

# All 64 electrodes in pin order; EOG is connector 1 channel 32.
CAP_CHANNELS = CAP_EEG_CHANNELS[:31] + (CAP_EOG_CHANNEL,) + CAP_EEG_CHANNELS[31:]

# Electrodes the offline model contract does not use (see the legacy prototype
# in scripts/dataproc/streaming/config.py, which shares CA-208 and COG-BCI).
MODEL_EXCLUDED_EEG_CHANNELS = ("Fpz", "M1", "Cz", "M2", "PO5", "PO6")

# 57 EEG electrodes of the classifier contract, still in cap order.
MODEL_EEG_CHANNELS = tuple(
    name for name in CAP_EEG_CHANNELS if name not in MODEL_EXCLUDED_EEG_CHANNELS
)

# Name prefixes that identify a channel which is not EEG. They are the fallback
# when an outlet declares nothing useful in its channel types, and they also
# override a declared "eeg" type: control software routinely labels the EOG
# drop lead as EEG, while a channel called EOG is an EOG.
AUXILIARY_PREFIXES = (
    "EOG",
    "EKG",
    "ECG",
    "EMG",
    "RESP",
    "GSR",
    "TEMP",
    "STATUS",
    "TRIG",
    "STI",
    "MARKER",
    "MISC",
    "SCLK",
    "SDAT",
)

# Channel types that mean "auxiliary, but not an EOG".
NON_EOG_AUXILIARY_TYPES = (
    "ecg",
    "emg",
    "resp",
    "gsr",
    "temperature",
    "bio",
    "misc",
    "stim",
    "exg",
    "eyetrack",
)

CAP_MODES = ("auto", "ca-208", "declared")
CHANNEL_MODES = ("cap", "model")
EOG_MODES = ("eog", "drop")


def _normalise(label) -> str:
    """Uppercase a label and drop every non-alphanumeric character."""

    return "".join(
        character for character in str(label).upper() if character.isalnum()
    )


def _looks_auxiliary(label) -> bool:
    """Whether a channel name identifies a non-EEG electrode."""

    normalised = _normalise(label)
    return any(normalised.startswith(prefix) for prefix in AUXILIARY_PREFIXES)


def _is_eog(label, declared_type: str | None) -> bool:
    """Whether a channel is the EOG auxiliary the contract keeps."""

    return declared_type == "eog" or _normalise(label).startswith("EOG")


@dataclass(frozen=True)
class CapProfile:
    """One cap's electrode contract, from a datasheet or from the outlet.

    Args:
        name: Profile label for reports, e.g. ``"CA-208"`` or ``"declared"``.
        eeg_channels: EEG labels in the order the cap or the outlet presents
            them.
        eog_channels: Auxiliary labels the contract keeps as EOG channels.
        reference: Amplifier reference when the profile knows it; ``None``
            means the profile asserts nothing and the run records the
            operator's ``--reference`` string instead.
        ground: Amplifier ground, same convention as ``reference``.
        source: Where the profile came from (``"datasheet"`` or ``"outlet"``).
        other_auxiliary: Declared channels that are neither EEG nor EOG (a
            status channel, a trigger line). They are reported and dropped as
            extras; they are never part of the contract.
        detail: Free-text provenance, e.g. the datasheet reference.

    Raises:
        ValueError: On a missing name or an empty EEG list, on duplicated
            labels, or on a label that is both EEG and auxiliary.
    """

    name: str
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...] = ()
    reference: str | None = None
    ground: str | None = None
    source: str = "datasheet"
    other_auxiliary: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        """Validate the contract: labels exist, are unique and do not overlap."""

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A cap profile needs a name.")
        if not self.eeg_channels:
            raise ValueError("A cap profile needs at least one EEG electrode.")
        labels = tuple(str(label) for label in self.eeg_channels)
        auxiliary = tuple(str(label) for label in self.eog_channels)
        for label in labels + auxiliary:
            if not label:
                raise ValueError("Channel labels cannot be empty.")
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Duplicated EEG labels: {duplicates!r}.")
        overlap = sorted(set(labels) & set(auxiliary))
        if overlap:
            raise ValueError(f"Labels cannot be both EEG and EOG: {overlap!r}.")
        object.__setattr__(self, "eeg_channels", labels)
        object.__setattr__(self, "eog_channels", auxiliary)
        object.__setattr__(
            self, "other_auxiliary", tuple(str(label) for label in self.other_auxiliary)
        )

    @property
    def channels(self) -> tuple[str, ...]:
        """Every contracted channel: EEG first, then auxiliary.

        This is the run's contract order, not the connector's pin order: the
        package expects EEG columns first, so the auxiliary channels move to
        the end even when the datasheet interleaves them (CA-208 publishes its
        EOG lead as connector-1 pin 32).
        """

        return self.eeg_channels + self.eog_channels

    @property
    def n_eeg(self) -> int:
        """Number of EEG electrodes in the contract."""

        return len(self.eeg_channels)

    def matches(self, declared_channels) -> bool:
        """Whether the declared labels cover this profile completely.

        Coverage is what ``--cap auto`` tests: an outlet that publishes every
        electrode of the profile (and may publish extras) is this cap.
        """

        present = {str(name) for name in declared_channels}
        return all(name in present for name in self.channels)

    def model_subset(self, model_channels=MODEL_EEG_CHANNELS) -> tuple[str, ...]:
        """The offline model's electrodes, in model order, that this cap has."""

        available = set(self.eeg_channels)
        return tuple(name for name in model_channels if name in available)

    def match(self, declared_channels) -> dict:
        """Compare declared labels against this profile.

        Returns:
            A plain dictionary: expected/present/missing EEG and EOG counts and
            labels, declared channels outside the profile, whether the declared
            EEG labels follow the profile's order, and the profile identity.
        """

        declared = tuple(str(name) for name in declared_channels)
        present = set(declared)
        expected = set(self.eeg_channels)
        matched = [name for name in declared if name in expected]
        in_order = [name for name in self.eeg_channels if name in present]

        return {
            "profile": self.name,
            "source": self.source,
            "reference": self.reference,
            "ground": self.ground,
            "eeg_expected": len(self.eeg_channels),
            "eeg_present": len(matched),
            "eeg_missing": tuple(
                name for name in self.eeg_channels if name not in present
            ),
            "eeg_order": matched == in_order,
            "eog_expected": len(self.eog_channels),
            "eog_present": sum(1 for name in self.eog_channels if name in present),
            "eog_missing": tuple(
                name for name in self.eog_channels if name not in present
            ),
            "extra": tuple(name for name in declared if name not in set(self.channels)),
        }


def _classify_channels(labels, types):
    """Split declared labels into EEG, EOG and the rest.

    The rules, in order: an auxiliary-looking name is auxiliary even when the
    outlet declares it as EEG; a declared ``eog`` is the EOG auxiliary; anything
    else declared ``eeg`` or with no usable type is EEG; any other declared type
    stays auxiliary but not EOG.
    """

    eeg: list[str] = []
    eog: list[str] = []
    other: list[str] = []
    for label, kind in zip(labels, types):
        if _looks_auxiliary(label):
            (eog if _is_eog(label, kind) else other).append(label)
        elif _is_eog(label, kind):
            eog.append(label)
        elif kind in NON_EOG_AUXILIARY_TYPES:
            other.append(label)
        else:
            eeg.append(label)

    return eeg, eog, other


def declared_profile(
    channel_names,
    channel_types=None,
    *,
    name: str = "declared",
    reference: str | None = None,
    ground: str | None = None,
) -> CapProfile:
    """Build a profile from what the outlet declares.

    Classification, in order:

    1. a name that looks auxiliary (an ``EOG``/``EKG``/``Status``/... prefix)
       is auxiliary, even when the outlet declares it as EEG;
    2. a channel declared ``eog`` is the EOG auxiliary;
    3. everything else declared ``eeg``, or with no usable type, is EEG;
    4. any other declared type is auxiliary but not EOG, so it is reported and
       dropped rather than contracted.

    Args:
        channel_names: Labels in declared order.
        channel_types: Optional per-channel types in the same order.
        name: Profile label for reports.
        reference: Reference to record, when the operator stated one.
        ground: Ground to record, when the operator stated one.

    Raises:
        ValueError: On an empty or duplicated label list, on a length mismatch
            between names and types, or when no EEG channel can be identified.
    """

    labels = tuple(str(label) for label in channel_names)
    if not labels:
        raise ValueError("A declared profile needs at least one channel.")
    if len(set(labels)) != len(labels):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"Duplicated declared labels: {duplicates!r}.")
    if channel_types is None:
        types: tuple[str | None, ...] = (None,) * len(labels)
    else:
        types = tuple(str(kind).lower() if kind else None for kind in channel_types)
        if len(types) != len(labels):
            raise ValueError("channel_types must match channel_names.")

    eeg, eog, other = _classify_channels(labels, types)
    if not eeg:
        raise ValueError(
            "No EEG channel could be identified in the declared labels; use "
            "--cap ca-208, or publish a montage whose EEG channels carry the "
            "'eeg' type."
        )

    return CapProfile(
        name=name,
        eeg_channels=tuple(eeg),
        eog_channels=tuple(eog),
        reference=reference,
        ground=ground,
        source="outlet",
        other_auxiliary=tuple(other),
        detail="built from the labels and types the outlet declared",
    )


# The one datasheet profile this repository ships.
CA208 = CapProfile(
    name="CA-208",
    eeg_channels=CAP_EEG_CHANNELS,
    eog_channels=(CAP_EOG_CHANNEL,),
    reference=CAP_REFERENCE,
    ground=CAP_GROUND,
    source="datasheet",
    detail="waveguard original CA-208, 64 channels, 10/10, EE-22x amplifier",
)


def select_profile(
    mode: str,
    declared_channels,
    channel_types=None,
    *,
    reference: str | None = None,
    ground: str | None = None,
) -> CapProfile:
    """Choose the cap profile for one outlet.

    Args:
        mode: ``"auto"`` (the CA-208 profile when the declared labels cover it,
            the declared profile otherwise), ``"ca-208"`` (demand the datasheet
            profile) or ``"declared"`` (always trust the outlet).
        declared_channels: Labels the outlet publishes.
        channel_types: Optional per-channel types, same order.
        reference: Reference the operator asserted, for declared profiles.
        ground: Ground the operator asserted, for declared profiles.

    Raises:
        ValueError: On an unknown mode, or when ``"ca-208"`` was demanded and
            the declared labels do not cover the profile.
    """

    if mode not in CAP_MODES:
        raise ValueError(f"cap mode must be one of {CAP_MODES}, not {mode!r}.")
    if mode == "declared":
        return declared_profile(
            declared_channels, channel_types, reference=reference, ground=ground
        )
    if not CA208.matches(declared_channels):
        if mode == "ca-208":
            match = CA208.match(declared_channels)
            missing = ", ".join(match["eeg_missing"][:8])
            suffix = "..." if len(match["eeg_missing"]) > 8 else ""
            raise ValueError(
                "The outlet does not publish the CA-208 contract: "
                f"{match['eeg_present']}/{match['eeg_expected']} EEG electrodes "
                f"present, missing {missing}{suffix}. Use --cap declared to "
                "trust the declared montage instead."
            )
        return declared_profile(
            declared_channels, channel_types, reference=reference, ground=ground
        )

    return CA208


def resolve_channels(
    profile: CapProfile,
    mode: str = "cap",
    eog: str = "eog",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the ``(eeg_channels, eog_channels)`` contract for a run.

    Args:
        profile: The cap profile selected for this outlet.
        mode: ``"cap"`` for every EEG electrode the profile has, ``"model"``
            for the offline classifier's electrodes that this cap also has. On
            a cap other than CA-208 the model set is an intersection, so ask
            :meth:`CapProfile.model_subset` how many electrodes that covers.
        eog: ``"eog"`` to keep the profile's EOG auxiliary in the contract,
            ``"drop"`` to leave it out entirely.

    Returns:
        EEG channel names first, then the auxiliary EOG names.

    Raises:
        ValueError: If ``mode`` or ``eog`` is not one of the documented values,
            or when ``"model"`` leaves no usable electrode.
        TypeError: When ``profile`` is not a :class:`CapProfile`.
    """

    if not isinstance(profile, CapProfile):
        raise TypeError("profile must be a CapProfile.")
    if mode == "cap":
        eeg = profile.eeg_channels
    elif mode == "model":
        eeg = profile.model_subset()
        if not eeg:
            raise ValueError(
                f"The {profile.name} profile shares no electrode with the "
                "offline model contract; use --channels cap."
            )
    else:
        raise ValueError(f"mode must be one of {CHANNEL_MODES}, not {mode!r}.")

    if eog == "eog":
        auxiliary = profile.eog_channels
    elif eog == "drop":
        auxiliary = ()
    else:
        raise ValueError(f"eog must be one of {EOG_MODES}, not {eog!r}.")

    return tuple(eeg), tuple(auxiliary)
