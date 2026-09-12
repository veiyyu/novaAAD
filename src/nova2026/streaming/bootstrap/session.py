"""Build every run-once component of a live session around a stream.

``StreamSession`` assembles the boilerplate start-up (channel contract,
optional recorder, acquire handle, window buffer) and owns the window gate.
Preprocessing does NOT live here: the script defines its own ordered tuple of
stages, each ``stage(data, timestamps) -> (data, timestamps)``, and runs it in
its own loop. The session only packages windows: warm-up is judged from the
sample counter, and quality/repair verdicts come from the ``judges`` the
script registered.
"""

from datetime import datetime

import numpy as np

from ..acquire import Acquire
from ..circular_buffer import CircularBuffer
from ..preflight import ChannelContract
from ..recording import RunRecorder, RunSpec
from ..window import EEGWindow


class StreamSession:
    """Assemble one live session's components around a connected stream.

    Args:
        stream: Connected ``StreamLSL`` (manual acquisition). Channel labels
            are read from it to build the contract.
        args: Parsed namespace (geometry + recording identity + consumer
            knobs). See :func:`~.args.parse_args` for the expected fields.
        channels: Canonical channel names, EEG first then auxiliary.
        contract: Optional pre-built :class:`~..preflight.ChannelContract`
            (normally the return value of :func:`~..preflight.prepare`). When
            omitted the session builds one from ``stream.ch_names`` itself.
        judges: Verdict providers queried by ``wrap()``; each must expose
            ``reasons(start, end) -> tuple[str, ...]`` (for example a
            :class:`~..preprocess.QualityMonitor`). Their reasons are unioned,
            so one rejected reason marks the whole window invalid.
        out_sfreq: Output rate in Hz that drives the window geometry; falls
            back to ``args.out_sfreq`` when omitted.
        source_unit_exponent: Power of ten of the source unit (0 = volts),
            stored by the optional recorder.
        role: Run role recorded in the metadata.
        ch_types: Optional per-channel MNE types; defaults to all EEG.
        n_eeg: Number of leading EEG columns; auxiliary columns are kept
            separately in each :class:`~..window.EEGWindow`.
        recorder_config: Optional serializable run description (rates, chain,
            geometry) stored in the run metadata for later replay/audit.
        recorder_files: Optional ``{role: path}`` provenance inputs copied and
            hashed into the run folder at open.
        recorder_track_windows: Whether the recorder logs every delivered
            window (see :meth:`~..recording.RunRecorder.log_window`).

    Notes:
        Preprocessing is fully owned by the script: no stage is built or run
        here, so the caller may chain, reorder and replace stages freely. The
        session does NOT connect the stream, start threads, process data or
        consume windows; it only builds the run-once pieces and exposes the
        loop helpers ``ingest`` / ``wrap``.

    Attributes:
        contract, recorder, acquire, buffer: The built components.
        judges: Verdict providers registered for ``wrap()``.
        eeg_count, n_channels, out_sfreq, warmup_samples: Geometry used here.
    """

    def __init__(
        self,
        stream,
        args,
        channels: tuple[str, ...],
        *,
        judges: tuple = (),
        out_sfreq: float | None = None,
        source_unit_exponent: int = 0,
        role: str = "run",
        ch_types: tuple[str, ...] | None = None,
        n_eeg: int | None = None,
        contract=None,
        recorder_config: dict | None = None,
        recorder_files: dict | None = None,
        recorder_track_windows: bool = False,
    ) -> None:
        """Build the contract, recorder, acquire, buffer and window gate."""

        self.args = args
        self.channels = tuple(str(name) for name in channels)
        self.n_channels = len(self.channels)
        # EEG columns come first; the rest are auxiliary (EOG) channels.
        self.eeg_count = self.n_channels if n_eeg is None else int(n_eeg)
        if not 1 <= self.eeg_count <= self.n_channels:
            raise ValueError("n_eeg must be between 1 and the channel count.")
        if ch_types is None:
            ch_types = ("eeg",) * self.n_channels
        self.ch_types = tuple(ch_types)

        sfreq = float(getattr(args, "sfreq", 500.0))

        # Channel contract: reuse the caller's pre-flight contract when one
        # was prepared; otherwise build it from the connected labels. Either
        # way a missing required label fails before any data is handled.
        if contract is not None:
            if tuple(contract.expected_channels) != self.channels:
                raise ValueError("The provided contract does not match channels.")
            if tuple(contract.source_channels) != tuple(stream.ch_names):
                # A contract built for another outlet maps columns by name and
                # would silently reorder this stream's data. Reject it instead:
                # contracts must come from preflight.prepare() on THIS stream.
                raise ValueError(
                    "The provided contract was built for a different source "
                    f"({tuple(contract.source_channels)!r} != "
                    f"{tuple(stream.ch_names)!r}); build it with "
                    "preflight.prepare() for the connected outlet."
                )
            self.contract = contract
        else:
            self.contract = ChannelContract(stream.ch_names, self.channels)

        # Optional per-run recorder (raw volts, before any transformation).
        self.recorder = None
        record_root = getattr(args, "record", None)
        if record_root is not None:
            run_name = getattr(args, "run", None) or datetime.now().strftime(
                "run-%Y%m%d-%H%M%S"
            )
            self.recorder = RunRecorder(
                record_root,
                RunSpec(
                    getattr(args, "subject", "demo"),
                    getattr(args, "session", "synthetic"),
                    run_name,
                    role=role,
                ),
                self.channels,
                sfreq,
                ch_types=self.ch_types,
                unit_exponent=source_unit_exponent,
                config=recorder_config,
                files=recorder_files,
                track_windows=recorder_track_windows,
            )

        # Verdict providers; the session only asks them, never orders them.
        self.judges = tuple(judges)

        # The window geometry lives at the OUTPUT rate of the preprocessing
        # chain. The script knows that rate (e.g. its resampler's), so it may
        # hand it in; without one we trust --out-sfreq.
        self.out_sfreq = float(
            out_sfreq if out_sfreq is not None else getattr(args, "out_sfreq", sfreq)
        )
        self.window_samples = round(
            float(getattr(args, "window", 2.0)) * self.out_sfreq
        )
        self.hop_samples = round(float(getattr(args, "hop", 0.5)) * self.out_sfreq)
        self.capacity_samples = round(
            float(getattr(args, "capacity", 6.0)) * self.out_sfreq
        )
        self.warmup_samples = round(
            float(getattr(args, "warmup", 2.0)) * self.out_sfreq
        )
        if self.capacity_samples < self.window_samples:
            raise ValueError("capacity must hold at least one window.")

        # Acquisition handle and window storage.
        self.acquire = Acquire(
            stream,
            block_samples=int(getattr(args, "block", 50)),
            sfreq=sfreq,
            no_data_timeout=5.0,
            max_lag_seconds=3.0,
        )
        self.buffer = CircularBuffer(
            self.window_samples,
            self.hop_samples,
            self.capacity_samples,
            sfreq=self.out_sfreq,
            n_channels=self.n_channels,
        )

    def ingest(self, data: np.ndarray, timestamps: np.ndarray):
        """Reorder to canonical columns and save the raw block if recording.

        Returns the reordered block with its timestamps, ready for the chain.
        """

        data = self.contract.reorder(data)
        if self.recorder is not None:
            self.recorder.write(data, timestamps)
        return data, timestamps

    def wrap(
        self,
        window: np.ndarray,
        window_times: np.ndarray,
        start_sample: int,
    ) -> EEGWindow:
        """Package one finished window into an :class:`EEGWindow`.

        This replaces hand-written gating: the returned object carries its own
        verdict (``valid``/``reasons``) and splits EEG from auxiliary columns.
        Preprocessing is not run here; ``ingest()`` output must already be
        processed and pushed by the script before a window is wrapped.

        Returns:
            An EEGWindow whose ``valid`` is False for warm-up windows or
            windows rejected by any registered judge.
        """

        start = float(window_times[0])
        end = float(window_times[-1])
        # Union every judge's verdict over this window's time span.
        reasons = sorted(
            {reason for judge in self.judges for reason in judge.reasons(start, end)}
        )
        valid = start_sample >= self.warmup_samples and not reasons

        return EEGWindow(
            data=window[:, : self.eeg_count],
            eog=window[:, self.eeg_count :],
            timestamps=window_times,
            valid=valid,
            reasons=reasons,
            start_sample=start_sample,
            segment=0,
            channel_names=self.channels[: self.eeg_count],
            contract=None,
            available_at=float(window_times[-1]),
        )

    def close(
        self,
        status: str = "completed",
        error: str | None = None,
        stats: dict | None = None,
    ) -> None:
        """Finalize the recorder (locks the run, exports the FIF, keeps stats).

        Args:
            status: ``"completed"`` or ``"failed"``.
            error: Optional human-readable error for failed runs.
            stats: Optional serializable counters (e.g.
                :meth:`~..stats.StreamStats.to_dict`) stored in the metadata.
        """

        if self.recorder is not None:
            self.recorder.close(status=status, error=error, stats=stats)
