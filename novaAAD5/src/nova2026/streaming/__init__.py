"""Real-time EEG streaming: decoupled acquisition, preprocessing and output."""

# Public entry points, re-exported so callers use one import line.
from .acquire import Acquire             # fixed-size blocks from the LSL inlet
from .circular_buffer import CircularBuffer   # ring storage -> overlapping windows
from .offload import TaskOffloader       # run per-window analysis on workers
from .preflight import (
    ChannelContract,  # validate + reorder source channels
    prepare,  # pre-flight check -> the run's channel contract
    resolve_outlet,  # confirm an outlet exists before connecting (B1)
    validate_source,  # check a connected inlet's metadata
)
from .preprocess.repair import UnrepairableError  # damage Repair cannot fix
from .preprocess.resample import (  # stateful 500->128 Hz (SoXR)
    Resampler,
    ResamplerQualityWarning,
    select_quality,
)
from .recording import RunRecorder, RunSpec  # per-run SQLite recording + identity
from .recovery import Recovery          # bounded recovery: reset or stop (A2)
from .spatial import (
    SpatialOperator,
    cut_epochs,
    fit_ssp,
    processing_contract,
)
from .stats import StreamStats          # run counters, persisted at close (E)
from .window import EEGWindow            # one window + verdict, for consumers

__all__ = [
    "Acquire",
    "ChannelContract",
    "CircularBuffer",
    "TaskOffloader",
    "Resampler",
    "ResamplerQualityWarning",
    "RunRecorder",
    "RunSpec",
    "SpatialOperator",
    "Recovery",
    "StreamStats",
    "UnrepairableError",
    "cut_epochs",
    "EEGWindow",
    "fit_ssp",
    "prepare",
    "processing_contract",
    "resolve_outlet",
    "select_quality",
    "validate_source",
]
