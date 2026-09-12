"""getlive: bring an EEG cap live and prove the streaming package drives it.

``nova2026.streaming`` was validated against PlayerLSL and recorded data only.
This package is the hardware counterpart: it talks to a real amplifier over
LSL, and it answers two questions on the rig:

1. Is the amplifier publishing LSL, and what does the outlet declare?
2. Does the package's preprocessing chain run on that outlet for a whole
   session, and does every electrode deliver signal?

The cap contract is resolved at run time, not baked in: a datasheet profile is
used when the outlet publishes exactly that cap (CA-208 today), and the declared
montage is used otherwise, so a different cap needs no code change. Known-dead
electrodes (dry caps) can be excluded from the fault verdict while still being
recorded.

Entry points:
    python -m scripts.getlive          the live acceptance run (``live.py``)
    python -m scripts.getlive.probe    metadata-only outlet inspection

Modules:
    cap: cap profiles (datasheet and declared), selection and channel contracts.
    outlets: resolve, select and inspect LSL outlets.
    checks: pure acceptance rules and per-electrode statistics.
    electrodes: per-electrode peak-to-peak statistics.
    probe: report what the network advertises, with and without connecting.
    live: the live run and its channel-quality policy.
    report: text and JSON rendering for one run.
"""
