"""Composable preprocessing stages for streamed EEG.

Transform stages share the contract ``stage(data, timestamps) ->
(data, timestamps)`` and keep their state across variable-size chunks.
``QualityMonitor`` and ``BadChannelJudge`` are judges, not transforms: feed the
monitor raw samples and ask about a window later; the judge only packages that
answer for a window. Nothing here touches LSL, threads or the buffer; only
``Resampler`` requires the optional ``soxr`` dependency.
"""

# Re-export every public preprocessing piece from one import location.
from .filters import SosFilter, design_bandpass, design_notch
from .quality import BadChannelJudge, QualityMonitor
from .repair import Repair, UnrepairableError
from .resample import Resampler, ResamplerQualityWarning, select_quality
from .scope import ChannelScope
from .units import unit_scaler

__all__ = [
    "BadChannelJudge",
    "ChannelScope",
    "QualityMonitor",
    "Repair",
    "Resampler",
    "ResamplerQualityWarning",
    "SosFilter",
    "UnrepairableError",
    "design_bandpass",
    "design_notch",
    "select_quality",
    "unit_scaler",
]
