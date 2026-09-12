"""Reusable command-line argument getter for streaming scripts.

Every live script needs the same geometry knobs (rates, windows,
preprocessing, recording identity, consumer mode). Instead of re-declaring
them with argparse in every script, build the parser once here and let the
script add its own options on top.
"""

import argparse
from collections.abc import Callable
from pathlib import Path


def make_parser(
    description: str = "",
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser preloaded with the common live-stream options.

    Args:
        description: Shown as the parser's help text.
        extra: Optional callback that receives the parser so the caller can
            register its own arguments before ``parse_args``.
    """

    parser = argparse.ArgumentParser(description=description)

    timing = parser.add_argument_group("source and timing")
    timing.add_argument("--duration", type=float, default=8.0,
                        help="seconds to stream")
    timing.add_argument("--sfreq", type=float, default=500.0,
                        help="source rate in Hz")
    timing.add_argument("--out-sfreq", type=float, default=128.0,
                        help="target rate in Hz after resampling")
    timing.add_argument("--block", type=int, default=50,
                        help="samples per Acquire.read()")
    timing.add_argument("--chunk-size", type=int, default=37,
                        help="source chunk size (uneven on purpose)")

    windows = parser.add_argument_group("windows (defined at the output rate)")
    windows.add_argument("--window", type=float, default=2.0,
                         help="window length in seconds")
    windows.add_argument("--hop", type=float, default=0.5,
                         help="window step in seconds")
    windows.add_argument("--capacity", type=float, default=6.0,
                         help="ring capacity in seconds")

    chain = parser.add_argument_group("preprocessing")
    chain.add_argument("--notch", type=float, default=60.0,
                       help="notch frequency in Hz")
    chain.add_argument("--notch-q", type=float, default=30.0,
                       help="notch quality")
    chain.add_argument("--lpass", type=float, default=1.0,
                       help="band-pass low cutoff")
    chain.add_argument("--hpass", type=float, default=45.0,
                       help="band-pass high cutoff")
    chain.add_argument("--order", type=int, default=3,
                       help="band-pass order")
    chain.add_argument("--warmup", type=float, default=2.0,
                       help="warm-up seconds before windows are trusted")
    chain.add_argument("--resample-quality",
                       choices=("auto", "LQ", "MQ", "HQ", "VHQ"), default="LQ",
                       help="SoXR quality preset; 'auto' measures each preset "
                            "and picks the cleanest one within the delay budget")

    recording = parser.add_argument_group("run recording")
    recording.add_argument("--record", type=Path, default=None,
                           help="recording root, e.g. records/ (off when omitted)")
    recording.add_argument("--subject", type=str, default="demo")
    recording.add_argument("--session", type=str, default="synthetic")
    recording.add_argument("--run", type=str, default=None,
                           help="run name; defaults to a timestamp")

    consumer = parser.add_argument_group("consumer mode")
    consumer.add_argument("--workers", type=int, default=0,
                          help="0 keeps analysis in the loop; N offloads")
    consumer.add_argument("--queue", type=int, default=8,
                          help="offload queue capacity")
    consumer.add_argument("--compute", type=float, default=0.0,
                          help="simulated analysis seconds per window")

    if extra is not None:
        extra(parser)

    return parser


def parse_args(
    description: str = "",
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the common live-stream options (plus any caller extras)."""

    return make_parser(description, extra).parse_args(argv)
