"""Recorded-EEG streaming demonstration with an independent audio worker."""

from threading import Event, Thread
from time import monotonic

from nova2026.auditory.audio import AudioPlayback, LatestEstimate
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.pipeline import AuditoryPipeline
from nova2026.auditory.data import AttentionEstimate
from nova2026.auditory.evaluation import assert_held_out
from nova2026.auditory.streaming import AuditoryProcessor, stream_config

from .runner import replay_windows


class AuditoryReplayStreamer:
    """Keep initialize()/stream() entry points; never describe replay as live EEG.

    The calling thread extracts features and decodes. A separate audio worker
    plays the matching tracks and expires decisions even if decoding stalls.
    Virtual recording time is mapped to a monotonic replay clock. Actual acoustic
    latency still requires loopback measurement before a live participant study.
    """

    def __init__(self, trial, model, on_result=None):
        self.trial = trial
        self.model = model
        self.on_result = on_result
        self.initialized = False
        self.worker = None
        self.stop_event = Event()
        self.error = None

    def initialize(self):
        assert_held_out(self.trial, self.model)
        processor = AuditoryProcessor(stream_config(
            self.trial, self.model.config, self.model.training_info["history"]))
        if processor.contract != self.model.contract:
            raise ValueError("Playback source contract differs from decoder; retrain first.")
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("Stop the active run before initializing again.")
        self.handoff = LatestEstimate()
        self.controller = AttentionController()
        self.stop_event.clear()
        self.error = None
        self.initialized = True

    def clock(self):
        return self.trial.timestamps[0] + monotonic() - self.started_at

    def blocks(self):
        block_size = round(self.trial.audio_rate * 0.032)
        for position in range(0, len(self.trial.audio), block_size):
            yield self.trial.audio[position : position + block_size]

    def play(self):
        try:
            self.playback.stream(self.blocks(), self.stop_event)
        except Exception as error:  # noqa: BLE001 - transport worker errors to the caller
            self.error = error
        finally:
            self.stop_event.set()

    def stream(self):
        if not self.initialized:
            raise RuntimeError("Call initialize() before stream().")
        self.initialized = False
        self.started_at = monotonic()
        self.playback = AudioPlayback(
            self.controller, self.handoff, self.trial.audio_rate, self.clock
        )
        pipeline = AuditoryPipeline(self.model, self.clock)
        self.worker = Thread(target=self.play, name="auditory-audio", daemon=True)
        self.worker.start()
        processing_error = None
        deadline = self.started_at + len(self.trial.audio) / self.trial.audio_rate + 3.0
        try:
            history = self.model.training_info["history"]
            for window in replay_windows(self.trial, self.model.config, history):
                wait = max(0, window.available_at - self.clock())
                if self.stop_event.wait(wait):
                    break
                estimate, original = pipeline.rundown(window)
                self.handoff.put(estimate)
                if self.on_result is not None:
                    self.on_result((estimate, original))
        except Exception as error:
            processing_error = error
            now = self.clock()
            self.handoff.put(AttentionEstimate(None, now, now, False, ("processing_failed",)))
        finally:
            while (self.worker.is_alive() and not self.stop_event.is_set()
                   and monotonic() < deadline):
                self.stop_event.wait(min(0.05, max(0, deadline - monotonic())))
            self.stop()
            self.worker.join(timeout=3)
        if self.worker.is_alive():
            raise RuntimeError("Audio device did not stop within the timeout.")
        if self.error is not None:
            raise self.error from processing_error
        if processing_error is not None:
            raise processing_error

    def stop(self):
        self.stop_event.set()
