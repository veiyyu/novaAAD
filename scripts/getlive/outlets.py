"""Resolve, select and inspect LSL outlets before any sample is acquired.

Two facts shape this module:

* A resolved outlet exposes its identity (name, type, source id, channel
  count, nominal rate) but **not** its channel labels, types or units until an
  inlet is connected: the description only arrives with the connection.
* The amplifier is a shared lab device, so selection must never guess between
  two plausible outlets; it either picks the single candidate or fails with
  the list of what is on the network.

Nothing here acquires samples, starts a thread or writes a file. Cap contracts
live in :mod:`scripts.getlive.cap`; this module only deals with LSL.
"""

from time import monotonic


def _default_resolver(timeout: float):
    """Import MNE-LSL lazily so this module stays importable without it."""

    from mne_lsl.lsl import resolve_streams

    return resolve_streams(timeout)


def outlet_rows(infos) -> list[dict]:
    """Describe each resolved outlet as a plain, printable dictionary.

    Labels are reported when the outlet advertised them; LSL only guarantees
    that after connecting, which ``channel_facts`` reads instead.
    """

    rows = []
    for info in infos:
        labels = info.get_channel_names()
        rows.append(
            {
                "name": info.name,
                "stype": info.stype,
                "source_id": info.source_id,
                "n_channels": int(info.n_channels),
                "sfreq": float(info.sfreq),
                "hostname": info.hostname,
                "uid": str(info.uid),
                "labels_advertised": None if labels is None else tuple(labels),
            }
        )
    return rows


def find_outlets(
    *,
    timeout: float = 10.0,
    interval: float = 1.0,
    resolver=None,
) -> list[dict]:
    """Keep resolving until an outlet appears or ``timeout`` expires.

    Args:
        timeout: Total seconds to keep looking.
        interval: Seconds per resolution attempt; a single attempt can only
            see outlets that are already publishing, so a fresh amplifier
            needs more than one.
        resolver: ``(timeout) -> infos``; injectable for tests, defaults to
            ``mne_lsl.lsl.resolve_streams``.

    Returns:
        One row per outlet, empty when none appeared in time.
    """

    if timeout <= 0 or interval <= 0:
        raise ValueError("timeout and interval must be positive.")
    resolve = _default_resolver if resolver is None else resolver
    deadline = monotonic() + float(timeout)

    while True:
        remaining = deadline - monotonic()
        # One attempt can only see outlets that already publish, so retry.
        rows = outlet_rows(resolve(min(float(interval), max(remaining, 0.1))))
        if rows:
            return rows
        if monotonic() >= deadline:
            return []


def filter_rows(
    rows: list[dict],
    *,
    name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
) -> list[dict]:
    """Keep the outlets matching every filter that was given.

    Names and source ids are LSL identifiers and must match exactly; the type
    is compared case-insensitively because outlets commonly publish ``eeg``
    while an operator types ``EEG``.
    """

    kept = []
    for row in rows:
        if name is not None and row["name"] != name:
            continue
        if source_id is not None and row["source_id"] != source_id:
            continue
        if stream_type is not None and str(row["stype"]).lower() != stream_type.lower():
            continue
        kept.append(row)
    return kept


def is_eeg_like(row: dict, *, expected_channels: int | None = None) -> bool:
    """Whether an outlet could be the EEG amplifier.

    A known EEG type is enough. Outlets frequently publish an empty type (a
    PlayerLSL outlet with mixed channel types does, and the eego software is
    not guaranteed to fill it in either), so the channel count is the fallback:
    the outlet must then publish at least the electrodes the run needs.
    """

    if "eeg" in str(row["stype"]).lower():
        return True
    if expected_channels is None:
        return False
    return int(row["n_channels"]) >= int(expected_channels)


def describe_rows(rows: list[dict]) -> str:
    """Render outlets as one line each, for error messages."""

    if not rows:
        return "    (no LSL outlet found)"
    return "\n".join(
        f"    name={row['name']!r} type={row['stype']!r} "
        f"channels={row['n_channels']} sfreq={row['sfreq']:g} "
        f"source_id={row['source_id']!r} host={row['hostname']!r}"
        for row in rows
    )


def select_outlet(
    rows: list[dict],
    *,
    name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
    expected_channels: int | None = None,
) -> dict:
    """Choose exactly one outlet, or explain why the choice is not unique.

    With explicit filters, every matching outlet must be one and the same
    outlet. Without filters, the single EEG-like outlet is chosen; anything
    ambiguous fails, because silently streaming from the wrong device is
    worse than not streaming at all.

    Raises:
        RuntimeError: On no match, on several matches, or when nothing on the
            network looks like the amplifier.
    """

    if not rows:
        raise RuntimeError("No LSL outlet is publishing on this network.")

    if name is not None or source_id is not None or stream_type is not None:
        matched = filter_rows(
            rows, name=name, source_id=source_id, stream_type=stream_type
        )
        if not matched:
            raise RuntimeError(
                "No outlet matches the requested identity.\n"
                f"Available:\n{describe_rows(rows)}"
            )
        if len(matched) > 1:
            raise RuntimeError(
                f"{len(matched)} outlets match the requested identity; "
                "narrow it with --stream-name or --source-id.\n"
                f"Matched:\n{describe_rows(matched)}"
            )
        return matched[0]

    candidates = [row for row in rows if is_eeg_like(row, expected_channels=expected_channels)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No outlet looks like an EEG amplifier (no EEG type and no "
            "matching channel count); inspect the network with probe.py and "
            "pass --stream-name.\n"
            f"Available:\n{describe_rows(rows)}"
        )
    raise RuntimeError(
        f"{len(candidates)} outlets could be the amplifier; "
        "pass --stream-name or --source-id.\n"
        f"Candidates:\n{describe_rows(candidates)}"
    )


def open_inlet(
    row: dict,
    *,
    bufsize: float,
    connect_timeout: float = 5.0,
):
    """Connect an inlet pinned to exactly one resolved outlet.

    All identity fields the outlet published are passed back, so LSL resolves
    the same outlet again instead of a namesake. Acquisition is manual: the
    caller decides when samples are pulled, and merely connecting never
    consumes data.

    Raises:
        RuntimeError: If the outlet carries no usable identity, or if LSL
            resolves several outlets for the given identity.
    """

    name = row.get("name") or None
    source_id = row.get("source_id") or None
    stype = row.get("stype") or None
    if name is None and source_id is None:
        raise RuntimeError(
            "The outlet publishes neither a name nor a source id, so it cannot "
            "be identified for connection."
        )

    # Imported here so the rest of the module works without MNE-LSL.
    from mne_lsl.stream import StreamLSL

    stream = StreamLSL(
        bufsize=bufsize,
        name=name,
        stype=stype,
        source_id=source_id,
    )
    try:
        stream.connect(
            acquisition_delay=None,
            processing_flags=["clocksync"],
            timeout=connect_timeout,
        )
    except BaseException:
        # Never leave a half-open inlet behind on a failed connect.
        if stream.connected:
            stream.disconnect()
        raise
    return stream


def channel_facts(stream) -> dict:
    """Read the full declared metadata of a connected inlet.

    The values are reported exactly as declared: nothing is converted, and a
    unit string is never rewritten to another unit.
    """

    names = tuple(str(name) for name in stream.ch_names)
    units = stream.sinfo.get_channel_units()
    return {
        "name": stream.name,
        "stype": stream.stype,
        "source_id": stream.source_id,
        "sfreq": float(stream.info["sfreq"]),
        "n_channels": len(names),
        "channels": names,
        "types": tuple(stream.get_channel_types(picks=list(names))),
        "units": None if units is None else tuple(str(unit) for unit in units),
        "dtype": str(stream.dtype),
    }


def format_outlets(rows: list[dict]) -> str:
    """Render the resolved-outlet table."""

    if not rows:
        return "No LSL outlet found."
    header = ("name", "type", "channels", "sfreq", "source_id", "host")
    lines = [header]
    for row in rows:
        # An outlet may publish no type at all; show that instead of a blank.
        lines.append(
            (
                str(row["name"]) or "(no name)",
                str(row["stype"]) or "(none)",
                str(row["n_channels"]),
                f"{row['sfreq']:g}",
                str(row["source_id"]) or "(none)",
                str(row["hostname"]),
            )
        )
    widths = [max(len(line[column]) for line in lines) for column in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(line))
        for line in lines
    )


def wait_for_outlet(
    *,
    name: str | None = None,
    source_id: str | None = None,
    stream_type: str | None = None,
    timeout: float = 10.0,
    interval: float = 1.0,
    expected_channels: int | None = None,
    resolver=None,
) -> dict:
    """Resolve and select one outlet, retrying until ``timeout`` expires."""

    rows = find_outlets(timeout=timeout, interval=interval, resolver=resolver)
    if not rows:
        raise RuntimeError(
            f"No LSL outlet appeared within {timeout:g}s. Check that the eego "
            "software has 'Enable LSL EEG streaming' ticked and that the "
            "amplifier is connected, then run probe.py."
        )
    return select_outlet(
        rows,
        name=name,
        source_id=source_id,
        stream_type=stream_type,
        expected_channels=expected_channels,
    )
