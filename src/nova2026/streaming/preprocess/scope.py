"""Channel scope: which EEG columns a stage is allowed to judge.

A dry cap arrives with electrodes that are dead for the whole session, and a
labelled montage makes "this column is not signal" an operator assertion rather
than something a detector should rediscover every window. One object carries
that assertion to every stage that would otherwise stop a run because of it:
:class:`~.quality.QualityMonitor` for the window verdict and
:class:`~.repair.Repair` for the repair decisions.

Scope is *evidence preserving*. An out-of-scope column keeps its data and is
still reported; it simply cannot fail a check. Nothing here removes columns, so
a decoder's channel contract stays valid.
"""

import numpy as np


class ChannelScope:
    """Resolve excluded labels to column positions and answer scope queries.

    Args:
        channel_names: Optional labels in column order. Required as soon as
            ``exclude_channels`` is non-empty, because a label is the stable
            identity across runs while a column index is not.
        exclude_channels: Labels of columns that are known dead (for example
            the electrodes a session was labelled with).

    Raises:
        TypeError: When a label collection is passed as a bare string.
        ValueError: When exclusions are given without labels, or when a label
            is not part of ``channel_names``. An unknown label must fail loudly:
            silently excluding nothing would leave the run stopping for an
            electrode the operator believes is already handled.

    Attributes:
        channel_names: Validated labels, or ``None`` when none were given.
        exclude_channels: Validated, de-duplicated excluded labels.
        excluded_indices: Column positions of the excluded labels.
    """

    def __init__(
        self,
        channel_names: tuple[str, ...] | None = None,
        exclude_channels: tuple[str, ...] = (),
    ) -> None:
        """Validate labels and resolve the exclusions."""

        if isinstance(channel_names, str):
            raise TypeError("channel_names must be an iterable of labels, not a string.")
        self.channel_names = (
            None
            if channel_names is None
            else tuple(str(name) for name in channel_names)
        )
        if isinstance(exclude_channels, str):
            raise TypeError(
                "exclude_channels must be an iterable of labels, not a string."
            )
        labels = tuple(dict.fromkeys(str(name) for name in exclude_channels))
        if labels and self.channel_names is None:
            raise ValueError("exclude_channels needs channel_names to be meaningful.")
        if labels:
            unknown = [name for name in labels if name not in self.channel_names]
            if unknown:
                raise ValueError(f"Unknown exclude_channels labels: {unknown!r}.")
        self.exclude_channels = labels
        self.excluded_indices = frozenset(
            self.channel_names.index(name) for name in labels
        )

    def __bool__(self) -> bool:
        """Whether any column is excluded."""

        return bool(self.excluded_indices)

    def contains(self, index: int) -> bool:
        """Whether one column position is excluded."""

        return index in self.excluded_indices

    def mask(self, channels: int) -> np.ndarray:
        """Return a boolean in-scope mask over ``channels`` columns.

        Positions outside ``channel_names`` are in scope: auxiliary columns are
        not part of the excluded EEG montage, so they are judged as before.
        """

        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer.")
        in_scope = np.ones(channels, dtype=bool)
        for index in self.excluded_indices:
            if index < channels:
                in_scope[index] = False

        return in_scope

    def excluded_mask(self, channels: int) -> np.ndarray:
        """Return the complement of :meth:`mask`."""

        return ~self.mask(channels)
