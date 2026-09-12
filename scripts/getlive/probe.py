"""Report the LSL outlets on this network and what the EEG outlet declares.

Run from the repository root:

    .venv/Scripts/python.exe -B -m scripts.getlive.probe

Use it before anything else on the rig: it answers "is the amplifier
publishing, under which identity, with which channels, units and rate?".

A resolved outlet only exposes its identity, so channel labels, types and
units are read by connecting to it. Connecting is metadata-only: no sample is
acquired, no thread is started and nothing is recorded. Pass ``--no-connect``
to stay at the resolve level.

Whatever the cap, the probe also reports which profile the live run would use
(``--cap auto`` by default) and how the declared montage compares with it, so a
new cap can be brought up without editing any contract first.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .cap import CA208, select_profile
from .outlets import (
    channel_facts,
    filter_rows,
    find_outlets,
    format_outlets,
    is_eeg_like,
    open_inlet,
)


def uniform_or_summary(values: tuple[str, ...]) -> str:
    """Describe a per-channel attribute, collapsing it when it is uniform."""

    if not values:
        return "not declared"
    counts = Counter(values)
    if len(counts) == 1:
        return f"{values[0]} (all {len(values)} channels)"
    return ", ".join(f"{value} x{count}" for value, count in counts.most_common())


def wrap_names(names: tuple[str, ...], width: int = 90) -> str:
    """Wrap a channel-name list into lines that fit a terminal."""

    lines: list[str] = []
    current = ""
    for name in names:
        candidate = name if not current else f"{current}, {name}"
        if len(candidate) > width:
            lines.append(current)
            current = name
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(f"    {line}" for line in lines)


def format_cap_match(match: dict) -> str:
    """Render a profile comparison as a short, readable block."""

    lines = [
        f"profile                : {match['profile']} ({match['source']})",
        f"EEG electrodes present : {match['eeg_present']}"
        f"/{match['eeg_expected']}",
        f"EOG electrodes present : {match['eog_present']}"
        f"/{match['eog_expected']}",
        f"electrode order kept   : {match['eeg_order']}",
    ]
    if match.get("reference") or match.get("ground"):
        lines.append(
            f"reference / ground     : {match.get('reference') or 'not asserted'} / "
            f"{match.get('ground') or 'not asserted'}"
        )
    if match["eeg_missing"]:
        lines.append(f"missing electrodes     : {', '.join(match['eeg_missing'])}")
    if match["eog_missing"]:
        lines.append(f"missing auxiliary      : {', '.join(match['eog_missing'])}")
    if match["extra"]:
        lines.append(f"channels outside cap   : {', '.join(match['extra'])}")
    return "\n".join(lines)


def format_profile(profile) -> str:
    """Render one profile's identity and split, for any cap."""

    lines = [
        f"  profile     : {profile.name} ({profile.source})",
        f"  provenance  : {profile.detail}",
        f"  EEG         : {profile.n_eeg} electrode(s), in "
        f"{'contract' if profile.source == 'datasheet' else 'declared'} order",
    ]
    if profile.eog_channels:
        lines.append(f"  EOG         : {', '.join(profile.eog_channels)}")
    if profile.other_auxiliary:
        lines.append(
            f"  other aux   : {', '.join(profile.other_auxiliary)} (reported, dropped)"
        )
    return "\n".join(lines)


def format_facts(facts: dict, profile, match: dict) -> str:
    """Render one connected outlet's declared metadata and cap classification."""

    channels = facts["channels"]
    blocks = [
        f"  name        : {facts['name']!r}",
        f"  type        : {facts['stype']!r}",
        f"  source_id   : {facts['source_id']!r}",
        f"  channels    : {facts['n_channels']}",
        f"  sfreq       : {facts['sfreq']:g} Hz",
        f"  dtype       : {facts['dtype']}",
        f"  ch_types    : {uniform_or_summary(tuple(facts['types']))}",
        f"  ch_units    : {uniform_or_summary(tuple(facts['units'] or ()))}",
        "  cap         :",
        format_profile(profile),
        "  cap match   :",
        "\n".join(f"    {line}" for line in format_cap_match(match).splitlines()),
    ]
    if channels:
        blocks.append(f"  labels      : {len(channels)} declared")
        blocks.append(wrap_names(channels))
    return "\n".join(blocks)


def _add_outlet_args(parser: argparse.ArgumentParser) -> None:
    """Register how to find the outlets to inspect."""

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds to keep looking for an outlet",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds per resolution attempt",
    )
    parser.add_argument("--name", help="only this LSL outlet name")
    parser.add_argument("--source-id", help="only this LSL source id")
    parser.add_argument("--stream-type", help="only this LSL stream type")


def _add_cap_args(parser: argparse.ArgumentParser) -> None:
    """Register the cap classification options."""

    parser.add_argument(
        "--cap",
        choices=("auto", "ca-208", "declared"),
        default="auto",
        help="cap profile to classify against: auto (datasheet when it fits, "
        "else the declared montage), ca-208 (demand the datasheet), declared "
        "(always trust the outlet)",
    )
    parser.add_argument(
        "--reference",
        help="reference the amplifier applies, recorded for declared profiles",
    )
    parser.add_argument(
        "--ground",
        help="ground the amplifier uses, recorded for declared profiles",
    )


def _add_inspection_args(parser: argparse.ArgumentParser) -> None:
    """Register which outlets to inspect and where to write the report."""

    parser.add_argument(
        "--expect-channels",
        type=int,
        help="with no declared stream type, treat an outlet with at least this "
        "many channels as the EEG amplifier",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="inspect every resolved outlet, not only the EEG-like ones",
    )
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="resolve identities only; do not read channel metadata",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=5.0,
        help="seconds allowed for one metadata connection",
    )
    parser.add_argument("--json", type=Path, help="also write this report as JSON")


def build_parser() -> argparse.ArgumentParser:
    """Build the probe's command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    _add_outlet_args(parser)
    _add_cap_args(parser)
    _add_inspection_args(parser)
    return parser


def _print_classification(facts: dict, profile, match: dict, cap_mode: str) -> None:
    """Print the declared metadata, the cap profile and how they compare."""

    print(format_facts(facts, profile, match))
    if cap_mode == "auto" and profile is not CA208:
        print(
            "  note        : the declared montage does not cover the CA-208 "
            "datasheet; the run will trust these labels (--cap declared "
            "behaviour)"
        )
    if not profile.eog_channels:
        print(
            "  note        : no EOG-like channel declared; run with --eog "
            "drop, or check the montage"
        )


def main(argv: list[str] | None = None) -> int:
    """Print the outlet inventory and return the process exit code."""

    args = build_parser().parse_args(argv)
    rows = find_outlets(timeout=args.timeout, interval=args.interval)

    print("LSL outlets publishing on this network:")
    print(format_outlets(rows))
    report: dict = {"outlets": rows, "cap_mode": args.cap, "inspected": []}

    if not rows:
        print(
            "\nNo outlet found. Check that the eego software has "
            "'Enable LSL EEG streaming' ticked (Application options -> Network "
            "Operation), that the amplifier is connected, and that this host is "
            "on the same network."
        )
        _write_json(args.json, report)
        return 1

    if args.no_connect:
        print("\n--no-connect: channel labels, types and units were not read.")
        _write_json(args.json, report)
        return 0

    targets = _targets(rows, args)
    if not targets:
        print(
            "\nNo outlet looks like an EEG amplifier. Pass --name, --source-id "
            "or --stream-type to inspect one explicitly, --expect-channels N if "
            "it declares no stream type, or --all to inspect everything."
        )
        _write_json(args.json, report)
        return 1

    attempted, failed = _inspect(targets, args, report)
    _write_json(args.json, report)
    return 1 if failed == attempted else 0


def _targets(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Pick the outlets worth connecting to."""

    if args.name or args.source_id or args.stream_type:
        return filter_rows(
            rows,
            name=args.name,
            source_id=args.source_id,
            stream_type=args.stream_type,
        )
    if args.all:
        return list(rows)
    return [
        row for row in rows if is_eeg_like(row, expected_channels=args.expect_channels)
    ]


def _inspect_one(row: dict, args: argparse.Namespace, report: dict) -> bool:
    """Connect to one outlet, print its classification and fill the report.

    Returns:
        ``True`` when the outlet was inspected, ``False`` when it could not be
        reached or classified, so one bad outlet cannot hide the others.
    """

    print(f"\ninspecting name={row['name']!r} source_id={row['source_id']!r}:")
    try:
        stream = open_inlet(row, bufsize=1.0, connect_timeout=args.connect_timeout)
    except Exception as error:  # noqa: BLE001 - reported, other outlets continue
        print(f"  connection failed: {error}")
        return False
    try:
        facts = channel_facts(stream)
    finally:
        if stream.connected:
            stream.disconnect()

    try:
        profile = select_profile(
            args.cap,
            facts["channels"],
            facts["types"],
            reference=args.reference,
            ground=args.ground,
        )
    except ValueError as error:
        print(f"  cap classification failed: {error}")
        print(
            "  hint: --cap declared trusts the outlet; --cap ca-208 demands "
            "the datasheet"
        )
        return False

    match = profile.match(facts["channels"])
    _print_classification(facts, profile, match, args.cap)
    report["inspected"].append(
        {
            "outlet": row,
            "facts": facts,
            "profile": {
                "name": profile.name,
                "source": profile.source,
                "n_eeg": profile.n_eeg,
                "eeg_channels": list(profile.eeg_channels),
                "eog_channels": list(profile.eog_channels),
                "other_auxiliary": list(profile.other_auxiliary),
                "reference": profile.reference,
                "ground": profile.ground,
            },
            "cap_match": match,
            "model_subset": list(profile.model_subset()),
        }
    )

    return True


def _inspect(targets: list[dict], args: argparse.Namespace, report: dict) -> tuple[int, int]:
    """Inspect each target and count what could not be reached.

    Returns:
        ``(attempted, failed)`` so the caller can fail only when nothing could
        be inspected at all: one unreachable outlet must not hide the others.
    """

    attempted = failed = 0
    for row in targets:
        attempted += 1
        if not _inspect_one(row, args, report):
            failed += 1

    return attempted, failed


def _write_json(path: Path | None, report: dict) -> None:
    """Write the machine-readable report when a path was requested."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
