"""Real loopback LSL transport plus the paced WAV audio clock, without hardware."""

import time
import unittest
from threading import Event, Thread
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from nova2026.auditory.timing import TimestampedAudio
from scripts.auditory.live import run
from scripts.auditory.synthetic import synthetic_trial
from scripts.auditory.train import train


class LiveAudioTests(unittest.TestCase):
    def test_timestamped_lsl_to_paced_audio(self):
        from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock
        from mne_lsl.stream import StreamLSL
        model = train([synthetic_trial("train")], [synthetic_trial("validation")], alphas=(10.,))[0]
        trial = synthetic_trial("test", seconds=12)
        name = "audio-fix-" + uuid4().hex[:8]
        info = StreamInfo(name, "EEG", 2, 128., "float64", name)
        info.set_channel_names(["F3", "F4"])
        info.set_channel_types(["eeg", "eeg"])
        info.set_channel_units(["volts", "volts"])
        outlet = StreamOutlet(info)
        ready, stopped = Event(), Event()
        epoch = []

        class ObservedAudio(TimestampedAudio):
            def record(self, position, frames, audible_at):
                super().record(position, frames, audible_at)
                if not epoch:
                    epoch.append(audible_at)
                    ready.set()

        def publish():
            if not ready.wait(10):
                return
            for p in range(0, len(trial.eeg), 4):
                times = epoch[0] + trial.timestamps[p:p+4]
                if stopped.wait(max(0., times[-1] - local_clock())):
                    return
                outlet.push_chunk(trial.eeg[p:p+4] * 1e-6, timestamp=times)

        worker = Thread(target=publish, daemon=True)
        stream = StreamLSL(bufsize=4., name=name)
        stream.connect(acquisition_delay=None, processing_flags=["clocksync"], timeout=10)
        worker.start()
        started = time.monotonic()
        try:
            with patch("scripts.auditory.live.TimestampedAudio", ObservedAudio):
                audio, report = run(trial, model, stream, output="wav")
        finally:
            stopped.set()
            ready.set()
            worker.join(2)
            stream.disconnect()
            del outlet
        self.assertGreater(time.monotonic() - started, 11)
        self.assertEqual(len(audio), len(trial.audio))
        valid = [e for e in report["estimates"] if e["valid"]]
        self.assertGreaterEqual(len(valid), 3)
        self.assertTrue(all(np.argmax(e["scores"]) == 0 for e in valid))
        self.assertTrue(all(e["emitted_at"] - e["evidence_end"] < 3 for e in valid))
        self.assertEqual(report["clock_domain"], "local_lsl")


if __name__ == "__main__":
    unittest.main()
