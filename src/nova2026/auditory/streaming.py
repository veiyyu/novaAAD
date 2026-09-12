"""One causal EEG chain for auditory training, replay, and live acquisition."""

import math
from types import SimpleNamespace

from nova2026.streaming import Recovery, UnrepairableError
from nova2026.streaming.circular_buffer import CircularBuffer
from nova2026.streaming.judges import collect_verdict
from nova2026.streaming.preflight import ChannelContract
from nova2026.streaming.preprocess import (
    QualityMonitor, Repair, Resampler, SosFilter, design_bandpass,
)
from nova2026.streaming.window import EEGWindow


def stream_config(
    trial,
    config,
    history=5.0,
    step=1.0,
    *,
    check_channels=True,
    max_bad_channels=0,
    exclude_channels=(),
):
    """Build the chain settings for one trial and decoder configuration.

    The channel options are run policy, not part of the processing contract:
    ``AuditoryProcessor.contract`` is compared against a trained model's
    contract, so adding keys there would invalidate every existing decoder.
    """

    for name, value in (("history", history), ("step", step)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if history * config.sample_rate <= config.lag_samples + 2:
        raise ValueError("History must include the decoder lags and usable samples.")
    if not isinstance(check_channels, bool):
        raise TypeError("check_channels must be a bool.")
    if isinstance(exclude_channels, str):
        raise TypeError("exclude_channels must be an iterable of labels, not a string.")
    return SimpleNamespace(
        input_sfreq=round(trial.sample_rate, 6), output_sfreq=config.sample_rate,
        eeg_channels=tuple(trial.channel_names), bandpass=config.band,
        input_reference=trial.reference, upstream_processing=trial.upstream_processing,
        window_seconds=history, step_seconds=step, warmup_seconds=2.0,
        persistent_fault_seconds=max(15.0, 2 * history + 2),
        check_channels=check_channels, max_bad_channels=max_bad_channels,
        exclude_channels=tuple(exclude_channels),
    )


class AuditoryProcessor:
    """Use the current streaming primitives; retain every hop even in large chunks.

    Inputs are explicitly in microvolts. Live callers scale source units before
    feeding this chain. Channel selection occurs once, by name, at construction.
    """

    def __init__(self, settings, source_channels=None, judges=None):
        self.settings = settings
        names = settings.eeg_channels
        self.channels = ChannelContract(source_channels or names, names)
        n = len(names)
        # Dead-electrode policy. Both stages that can stop a run for one
        # channel take the same list, so no path is left unguarded.
        check_channels = bool(getattr(settings, "check_channels", True))
        max_bad_channels = int(getattr(settings, "max_bad_channels", 0))
        exclude_channels = tuple(getattr(settings, "exclude_channels", ()) or ())
        self.repair = Repair(
            settings.input_sfreq, source_unit_exponent=-6,
            channel_names=names, exclude_channels=exclude_channels,
        )
        self.quality = QualityMonitor(
            n_eeg=n, sfreq=settings.input_sfreq, channel_names=names,
            check_channels=check_channels, max_bad_channels=max_bad_channels,
            exclude_channels=exclude_channels,
        )
        self.bandpass = SosFilter(design_bandpass(*settings.bandpass, 3, settings.input_sfreq), n)
        self.resampler = Resampler(
            settings.input_sfreq, settings.output_sfreq, n, quality="auto",
            max_age_seconds=3.0, reserve_seconds=1.0, allow_qq=True, strict=True,
        )
        self.buffer = CircularBuffer(
            round(settings.window_seconds * settings.output_sfreq),
            round(settings.step_seconds * settings.output_sfreq),
            round((settings.window_seconds + 4) * settings.output_sfreq),
            settings.output_sfreq, n,
        )
        self.recovery = Recovery(
            resettable=(self.repair, self.quality, self.bandpass, self.resampler, self.buffer),
            persistent_fault_seconds=settings.persistent_fault_seconds,
        )
        # Judges are injected, so a script can add its own census rule (see
        # BadChannelJudge) without this class knowing about it.
        self.judges = self._resolve_judges(judges)
        self.contract = {
            "eeg_channels": list(names), "output_sfreq": settings.output_sfreq,
            "input_sfreq": settings.input_sfreq, "units": "uV",
            "input_reference": settings.input_reference,
            "upstream_processing": settings.upstream_processing,
            "bandpass": list(settings.bandpass), "filter_order": 3,
            "resample_quality": self.resampler.quality,
            "stage": "auditory-current-streaming-v2",
            "window_seconds": settings.window_seconds, "step_seconds": settings.step_seconds,
        }

    def _resolve_judges(self, judges):
        """Validate the injected judges; default to the monitor and the repairer."""

        if judges is None:
            return (self.quality, self.repair)
        if isinstance(judges, (str, bytes)):
            raise TypeError("judges must be an iterable of window judges.")
        resolved = tuple(judges)
        for judge in resolved:
            if not callable(getattr(judge, "reasons", None)):
                raise TypeError(
                    "Every judge must implement reasons(start, end); "
                    f"{judge!r} does not."
                )
        return resolved

    def feed(self, chunk):
        data, timestamps = chunk
        data = self.channels.reorder(data)
        if not len(timestamps):
            return []
        available = float(timestamps[-1])
        try:
            data, timestamps = self.repair(data, timestamps)
            self.quality.feed(data, timestamps)
            data, timestamps = self.bandpass(data, timestamps)
            data, timestamps = self.resampler(data, timestamps)
        except UnrepairableError as error:
            self.recovery.handle(error)
            return []
        result = []
        for rows, times, start in self.buffer.push(data, timestamps):
            reasons, bad_channels = collect_verdict(self.judges, times[0], times[-1])
            warm = start < round(self.settings.warmup_seconds * self.settings.output_sfreq)
            window = EEGWindow(
                rows, rows[:, :0], times, not reasons and not warm, reasons, start,
                segment=self.recovery.segment, channel_names=self.settings.eeg_channels,
                contract=self.contract, available_at=available,
                bad_channels=bad_channels,
            )
            self.recovery.watch(window)
            result.append(window)
        return result
