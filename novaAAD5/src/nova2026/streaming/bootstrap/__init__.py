"""One-time start-up helpers for live streaming scripts.

Scripts usually differ only in their consumer: the parsing and the
"build every run-once component" steps are the same. This package moves both
out of the script so a new script only writes its own loop.
"""

from .args import make_parser, parse_args
from .session import StreamSession

__all__ = ["StreamSession", "make_parser", "parse_args"]
