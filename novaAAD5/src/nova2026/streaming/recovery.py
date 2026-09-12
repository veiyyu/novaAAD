"""Bounded recovery: restart the chain on repairable damage, else stop.

``Repair`` (see :mod:`~.preprocess.repair`) fixes what it can and raises
:class:`~.preprocess.repair.UnrepairableError` for what it cannot. This module
decides what happens next. A live stream may stall or glitch once in a while;
killing the whole run for a single bad chunk is too brittle. ``Recovery``
gives the run a *bounded* number of restarts: on an unrepairable chunk it
resets every stateful stage (filters, resampler, ring buffer, quality and
repair history), increments the segment number and keeps going. Beyond the
limit, or when a single fault is too large, it stops loudly — a live system
must never silently ship garbage.

The guard is deliberately NOT part of ``StreamSession``: preprocessing lives
in the script's own stage tuple, so the script tells the guard which of its
components are resettable.
"""

import math

from .preprocess.repair import UnrepairableError


class Recovery:
    """Restart the chain on recoverable damage; stop beyond configured limits.

    Args:
        resettable: Stateful components to reset on a recovery, in order.
            Each must expose ``reset()`` (filters, resampler, ring buffer,
            quality monitor, repair stage). Scalers and other stateless
            stages must not be included.
        max_events: Most recoveries allowed before the run stops.
        max_gap_seconds: Largest timestamp gap that still counts as a
            recoverable fault; anything larger stops the run immediately.
        persistent_fault_seconds: Longest run of consecutive judge-rejected
            windows before the run stops. Keep it comfortably above the window
            length plus whatever settling the judges append
            (``QualityMonitor.warmup_seconds``, ``Repair.settle_seconds``):
            one transient sample invalidates every window that overlaps it, so
            with a 5 s window and a 2 s quality settling a single spike can
            look like about nine seconds of faults and end the run. Roughly
            twice the window length is the documented margin.
        recorder: Optional ``RunRecorder``; every recovery is then written as
            a ``"recovery:<kind>"`` event for the audit trail.

    Attributes:
        segment: Current processing segment; increases by one per recovery.
        recoveries: Recoveries performed so far.
        events: Recovery decisions as ``{kind, recovery, segment, gap}``.
    """

    def __init__(
        self,
        resettable: tuple = (),
        *,
        max_events: int = 5,
        max_gap_seconds: float = 0.5,
        persistent_fault_seconds: float = 5.0,
        recorder=None,
    ) -> None:
        """Validate limits and start fresh."""

        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer.")
        for name, value in (
            ("max_gap_seconds", max_gap_seconds),
            ("persistent_fault_seconds", persistent_fault_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if isinstance(resettable, str):
            raise TypeError("resettable must be an iterable of components.")
        self._resettable = tuple(resettable)
        self._recorder = recorder
        self.max_events = int(max_events)
        self.max_gap_seconds = float(max_gap_seconds)
        self.persistent_fault_seconds = float(persistent_fault_seconds)
        self.reset()

    def reset(self) -> None:
        """Start a fresh run: no history, segment 0."""

        self.segment = 0
        self.recoveries = 0
        self.events: list[dict] = []
        self._bad_since: float | None = None

    def handle(self, error: UnrepairableError) -> None:
        """Process one unrepairable chunk and decide the run's fate.

        A gap larger than ``max_gap_seconds``, or a recovery count beyond
        ``max_events``, raises ``RuntimeError`` (the run stops). Otherwise
        every resettable component is reset, the segment advances, and the
        decision is recorded.

        Raises:
            RuntimeError: When the fault is fatal or the recovery budget is
                exhausted.
            TypeError: When ``error`` is not an ``UnrepairableError``.
        """

        if not isinstance(error, UnrepairableError):
            raise TypeError("Recovery handles UnrepairableError only.")

        if error.kind == "large_gap" and error.gap_seconds is not None:
            if error.gap_seconds > self.max_gap_seconds:
                raise RuntimeError(
                    "A source gap exceeds the recovery limit "
                    f"({error.gap_seconds:.3f}s > {self.max_gap_seconds:g}s); "
                    "the run requires operator attention."
                ) from error

        self.recoveries += 1
        if self.recoveries > self.max_events:
            raise RuntimeError(
                "Too many data faults; the run requires operator attention."
            ) from error

        # Restart the whole chain: every stateful stage forgets its history.
        for component in self._resettable:
            reset = getattr(component, "reset", None)
            if reset is None:
                raise RuntimeError(
                    f"Recovery components must implement reset(); "
                    f"{component!r} does not."
                )
            reset()
        self.segment += 1
        self._bad_since = None
        self.events.append(
            {
                "kind": error.kind,
                "recovery": self.recoveries,
                "segment": self.segment,
                "gap": error.gap_seconds,
            }
        )
        if self._recorder is not None:
            self._recorder.mark(f"recovery:{error.kind}")

    def watch(self, window) -> None:
        """Watch consecutive judge-rejected windows (persistent faults).

        Call once per finished window, after ``wrap()``. Windows rejected only
        by warm-up carry no reasons and are ignored; windows rejected by a
        judge (quality fault, interpolation) accumulate. When the bad run
        lasts longer than ``persistent_fault_seconds`` the run stops.

        Raises:
            RuntimeError: When quality faults persisted beyond the limit, or
                when a window carries no finite time grid so persistence
                cannot be measured (silently ignoring those would disable the
                guard for the rest of the run).
        """

        start = float(window.timestamps[0])
        end = float(window.timestamps[-1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise RuntimeError(
                "Cannot watch fault persistence: the window has no finite "
                "timestamps (its ring buffer never anchored to real time)."
            )
        if window.reasons:
            if self._bad_since is None:
                self._bad_since = start
            elif end - self._bad_since > self.persistent_fault_seconds:
                raise RuntimeError(
                    "EEG quality faults persisted beyond the allowed "
                    f"duration ({self.persistent_fault_seconds:g}s)."
                )
        else:
            # A clean window resets the clock; only uninterrupted badness counts.
            self._bad_since = None
