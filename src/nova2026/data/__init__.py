"""Vendored trim: expose only the tube Pipeline the auditory path uses.

Upstream `nova2026.data` also re-exports `.eeg` (Loader/load_eeg/save_chkpt), which
imports torch for the EEGNet/PVT side — irrelevant to auditory decoding and a heavy
dependency. The auditory pipeline only needs `Pipeline`/`DefaultPipe` from
`.pipeline`, so this vendored copy imports just those. See VENDORED.md.
"""
from .pipeline import DefaultPipe, Pipeline

__all__ = ["DefaultPipe", "Pipeline"]
