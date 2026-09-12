"""The live entry point: ``python -m scripts.getlive ...``.

This dispatches to :mod:`scripts.getlive.live`, so the hardware test is one
command away while the pieces stay separately importable (``cap``, ``outlets``,
``checks``, ``electrodes``, ``report``) and the metadata probe stays its own
module (``python -m scripts.getlive.probe``).
"""

from .live import main

if __name__ == "__main__":
    raise SystemExit(main())
