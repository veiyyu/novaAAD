"""Republish an LSL outlet with the channel metadata the package requires.

Some control software publishes EEG over LSL without a channel description
that MNE-LSL can read: the inlet then falls back to positional labels
(``0, 1, 2 ...``) and reports ``ch_units`` as *not declared*. The package's
pre-flight check (:func:`nova2026.streaming.preflight.validate_source`) refuses
such a source on purpose, because a stream whose units are unknown cannot be
scaled to uV and a stream whose labels are positional cannot be contracted to
a montage.

This relay is the bridge. It subscribes to the outlet as it is, keeps the
columns that carry electrodes, and republishes them under a new name with
proper labels, types and units. Nothing is filtered, resampled or rescaled:
the samples and their timestamps are forwarded untouched, so the relay adds
metadata and latency only.

Run it in its own terminal, leave it running, then point the live test at the
republished outlet:

    python -B -m scripts.getlive.relay --source-name UnicornRecorderRawDataLSLStream \
        --labels Fz,C3,Cz,C4,Pz,PO7,Oz,PO8 --keep 0-7

    python -B -m scripts.getlive --stream-name NOVA_Relay --cap declared \
        --sfreq 250 --source-units uV --eog drop --duration 30

``--labels`` is an operator assertion about the montage, exactly like
``--sfreq`` and ``--source-units`` are for the live test: LSL is not telling
us the electrode names, so someone has to, and the run records what was
asserted.
"""

import argparse
import sys
from time import sleep

import numpy as np

from mne_lsl.lsl import StreamInfo, StreamInlet, StreamOutlet, resolve_streams

# What the package accepts as a per-channel unit string.
UNIT_CHOICES = ("volts", "millivolts", "microvolts", "nanovolts")


def parse_keep(text: str, n_channels: int) -> tuple[int, ...]:
    """Turn ``0-7`` or ``0,1,2`` into column indices, validated against the source."""

    if not text.strip():
        return tuple(range(n_channels))
    kept: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, stop = part.partition("-")
            kept.extend(range(int(start), int(stop) + 1))
        else:
            kept.append(int(part))
    for index in kept:
        if not 0 <= index < n_channels:
            raise ValueError(
                f"--keep names column {index}, but the source publishes "
                f"{n_channels} channels (0..{n_channels - 1})."
            )
    if len(set(kept)) != len(kept):
        raise ValueError("--keep names the same column twice.")
    return tuple(kept)


def build_parser() -> argparse.ArgumentParser:
    """Build the relay's command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("source outlet")
    source.add_argument(
        "--source-name",
        required=True,
        help="LSL name of the outlet to republish, as probe.py prints it",
    )
    source.add_argument(
        "--source-id", help="LSL source id, when two outlets share a name"
    )
    source.add_argument(
        "--resolve-timeout",
        type=float,
        default=10.0,
        help="seconds to keep looking for the source outlet",
    )

    contract = parser.add_argument_group("metadata to publish")
    contract.add_argument(
        "--labels",
        required=True,
        help="comma-separated electrode names for the kept columns, in order",
    )
    contract.add_argument(
        "--keep",
        default="",
        help="source columns to forward, e.g. 0-7 or 0,1,2 (default: all)",
    )
    contract.add_argument(
        "--types",
        default="eeg",
        help="comma-separated channel types, or one type for every channel",
    )
    contract.add_argument(
        "--units",
        default="microvolts",
        choices=UNIT_CHOICES,
        help="unit the source samples are already in; declared, never applied",
    )

    out = parser.add_argument_group("republished outlet")
    out.add_argument("--out-name", default="NOVA_Relay", help="name to publish under")
    out.add_argument("--out-type", default="EEG", help="LSL stream type to publish")
    out.add_argument(
        "--out-source-id",
        default="nova-relay",
        help="LSL source id to publish; keep it stable across restarts",
    )
    out.add_argument(
        "--chunk",
        type=int,
        default=32,
        help="samples pulled and pushed per iteration",
    )
    out.add_argument("--quiet", action="store_true", help="no periodic throughput line")
    return parser


def resolve_source(args: argparse.Namespace):
    """Find exactly one source outlet, or say what is on the network instead."""

    infos = resolve_streams(args.resolve_timeout)
    matched = [info for info in infos if info.name == args.source_name]
    if args.source_id is not None:
        matched = [info for info in matched if info.source_id == args.source_id]
    if not matched:
        available = "\n".join(
            f"    name={info.name!r} type={info.stype!r} "
            f"channels={info.n_channels} sfreq={info.sfreq:g} "
            f"source_id={info.source_id!r}"
            for info in infos
        )
        raise RuntimeError(
            f"No outlet named {args.source_name!r} is publishing.\n"
            f"Available:\n{available or '    (none)'}"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"{len(matched)} outlets are named {args.source_name!r}; "
            "pass --source-id to pick one."
        )
    return matched[0]


def build_outlet_info(args: argparse.Namespace, labels: tuple[str, ...], sfreq: float, dtype) -> StreamInfo:
    """Describe the republished outlet: the metadata the source never declared."""

    types = [name.strip() for name in args.types.split(",") if name.strip()]
    if len(types) == 1:
        types = types * len(labels)
    if len(types) != len(labels):
        raise ValueError(
            f"--types gives {len(types)} entries for {len(labels)} channels; "
            "pass one type, or one per channel."
        )

    sinfo = StreamInfo(
        name=args.out_name,
        stype=args.out_type,
        n_channels=len(labels),
        sfreq=sfreq,
        dtype=dtype,
        source_id=args.out_source_id,
    )
    sinfo.set_channel_names(list(labels))
    sinfo.set_channel_types(types)
    sinfo.set_channel_units([args.units] * len(labels))
    return sinfo


def main(argv: list[str] | None = None) -> int:
    """Forward one outlet to another, adding the metadata, until interrupted."""

    args = build_parser().parse_args(argv)
    labels = tuple(name.strip() for name in args.labels.split(",") if name.strip())
    if not labels:
        raise SystemExit("--labels must name at least one electrode.")
    if args.chunk < 1:
        raise SystemExit("--chunk must be a positive number of samples.")

    try:
        info = resolve_source(args)
    except RuntimeError as error:
        print(f"error: {error}")
        return 1

    try:
        keep = parse_keep(args.keep, int(info.n_channels))
    except ValueError as error:
        print(f"error: {error}")
        return 1
    if len(keep) != len(labels):
        print(
            f"error: --keep forwards {len(keep)} column(s) but --labels names "
            f"{len(labels)}; they must agree one to one."
        )
        return 1

    print(
        f"source    : name={info.name!r} type={info.stype!r} "
        f"channels={info.n_channels} sfreq={info.sfreq:g} Hz "
        f"source_id={info.source_id!r}"
    )
    print(f"forwarding: columns {list(keep)} as {', '.join(labels)}")

    inlet = StreamInlet(info, max_buffered=10, processing_flags=["clocksync"])
    inlet.open_stream(timeout=10.0)
    try:
        dtype = inlet.dtype
        sinfo = build_outlet_info(args, labels, float(info.sfreq), dtype)
        outlet = StreamOutlet(sinfo, chunk_size=args.chunk)
        print(
            f"publishing: name={args.out_name!r} type={args.out_type!r} "
            f"channels={len(labels)} sfreq={info.sfreq:g} Hz "
            f"units={args.units} source_id={args.out_source_id!r}"
        )
        print("Ctrl+C to stop.\n")

        columns = np.asarray(keep, dtype=int)
        forwarded = 0
        since_report = 0
        while True:
            data, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=args.chunk)
            if timestamps.size == 0:
                # No samples this turn: yield instead of spinning on the inlet.
                sleep(0.002)
                continue
            outlet.push_chunk(np.ascontiguousarray(data[:, columns]), timestamps)
            forwarded += int(timestamps.size)
            since_report += int(timestamps.size)
            if not args.quiet and since_report >= int(info.sfreq) * 5:
                print(f"forwarded {forwarded} samples")
                since_report = 0
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130
    finally:
        inlet.close_stream()

    return 0


if __name__ == "__main__":
    sys.exit(main())
