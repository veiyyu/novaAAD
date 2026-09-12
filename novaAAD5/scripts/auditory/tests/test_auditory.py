"""Check temporal alignment, null evidence, persistence and controller behavior."""

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import savemat, wavfile

from nova2026.auditory.audio import (
    AudioMixer,
    AudioPlayback,
    LatestEstimate,
    read_audio,
)
from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import (
    AttentionEstimate,
    AuditoryWindow,
    load_kuleuven,
    load_trial,
    save_trial,
)
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.envelopes import EnvelopeExtractor
from nova2026.auditory.evaluation import check_split
from scripts.auditory.replay import replay
from scripts.auditory.runner import replay_windows
from scripts.auditory.synthetic import synthetic_trial
from scripts.auditory.train import train


class AuditoryTests(unittest.TestCase):
    def test_kuleuven_import_keeps_source_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for candidate in (1, 2):
                wavfile.write(
                    root / f"part1_track{candidate}_dry.wav",
                    8000,
                    np.full(8000, 1000 * candidate, dtype=np.int16),
                )
            record = {
                "RawData": {"EegData": np.ones((128, 2))},
                "FileHeader": {"SampleRate": 128},
                "stimuli": np.array(
                    ["part1_track2_dry.wav", "part1_track1_dry.wav"], dtype=object
                ),
                "attended_track": 2,
            }
            path = root / "S1.mat"
            savemat(path, {"trials": np.array([record], dtype=object)})
            trial = load_kuleuven(path, root, ("F3", "F4"), "reference", "release")[0]
            np.testing.assert_equal(trial.labels, 1)
            self.assertLess(trial.audio[0, 0], trial.audio[0, 1])
            save_trial(trial, root / "trial.npz")
            restored = load_trial(root / "trial.npz")
            np.testing.assert_equal(trial.audio, restored.audio)

    def test_aasd_explicit_crop_and_switch_mapping(self):
        from scripts.auditory.convert import load_aasd_manifest

        class Raw:
            def __init__(self):
                self.info = {"sfreq": 128}

            def get_data(self, picks, start, stop):
                self.crop = (start, stop)
                return np.full((len(picks), stop - start), 1e-6)

            def close(self):
                pass

        raw = Raw()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.wav", "b.wav"):
                wavfile.write(root / name, 8000, np.zeros(8000, dtype=np.float32))
            row = {
                "eeg_file": "source.cnt",
                "start_seconds": 2,
                "stop_seconds": 3,
                "audio_files": ["a.wav", "b.wav"],
                "channel_names": ["F3", "F4"],
                "reference": "verified",
                "upstream_processing": "none",
                "subject": "S1",
                "trial_id": "one",
                "group": "story",
                "attention_events": [
                    {"seconds": 0, "candidate": 0},
                    {"seconds": 0.5, "candidate": 1},
                ],
                "switch_uncertainty_seconds": 0.05,
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"trials": [row]}))
            with patch("mne.io.read_raw_cnt", return_value=raw):
                trial = load_aasd_manifest(manifest)[0]
            np.testing.assert_equal(trial.eeg, 1)
            self.assertEqual(raw.crop, (256, 384))
            self.assertEqual(trial.labels[32], 0)
            self.assertEqual(trial.labels[64], -1)
            self.assertEqual(trial.labels[100], 1)

    def test_audio_expiry_without_new_inference(self):
        controller = AttentionController(max_age=0.2)
        handoff = LatestEstimate()
        handoff.put(AttentionEstimate([0.2, 0], 0, 0))
        times = iter([0.0, 0.1, 0.3, 0.4])
        written = []

        class Output:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def write(self, samples):
                written.append(samples)
                return False

        playback = AudioPlayback(controller, handoff, 8000, lambda: next(times))
        blocks = [np.full((80, 2), 0.5) for _ in range(4)]
        with patch.dict(
            "sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Output)}
        ):
            playback.stream(blocks, Event())
        self.assertEqual(len(written), 4)
        self.assertIsNone(controller.choice(0.4))
        self.assertEqual(playback.underruns, 0)

    def test_envelope_alias_rejection(self):
        rate = 8000
        times = np.arange(rate * 5) / rate
        amplitudes = []
        for modulation in (4, 60):
            speech = 0.4 + 0.2 * np.sin(2 * np.pi * modulation * times)
            speech *= np.sin(2 * np.pi * 1000 * times)
            values = EnvelopeExtractor(rate, AuditoryConfig()).feed(
                np.column_stack([speech, speech])
            )
            amplitudes.append(values[128:, 0].std())
        self.assertLess(amplitudes[1], amplitudes[0] * 0.01)

    def test_fault_is_reported_and_audio_continues(self):
        from nova2026.auditory.evaluation import inject_fault

        model, _ = train(
            [synthetic_trial("train")], [synthetic_trial("validation")], alphas=(10,)
        )
        damaged = inject_fault(synthetic_trial("test"), "dropout")
        audio, estimates, metrics = replay(damaged, model)
        self.assertIsNone(metrics["processing_failure"])
        self.assertGreater(len(audio), damaged.audio_rate * 20)
        self.assertTrue(
            any(not estimate["valid"] for estimate in estimates)
        )

    def test_envelope_chunk_invariance(self):
        trial = synthetic_trial(seconds=6)
        config = AuditoryConfig()
        whole = EnvelopeExtractor(trial.audio_rate, config).feed(trial.audio)
        extractor = EnvelopeExtractor(trial.audio_rate, config)
        parts = []
        for position in range(0, len(trial.audio), 257):
            parts.append(extractor.feed(trial.audio[position : position + 257]))
        chunked = np.concatenate(parts)
        np.testing.assert_allclose(whole, chunked, atol=1e-10)

    def test_controller_null_invalid_stale_and_manual(self):
        controller = AttentionController(min_switch_windows=1)
        controller.update(AttentionEstimate([0, 0], 1, 1), 1)
        np.testing.assert_equal(controller.gains(1), [1, 1])
        controller.update(AttentionEstimate([0.5, 0], 2, 2), 2)
        self.assertEqual(controller.choice(2), 0)
        self.assertIsNone(controller.choice(6))
        controller.update(AttentionEstimate([np.nan, 1], 7, 7), 7)
        self.assertIsNone(controller.choice(7))
        controller.set_manual(True, 1)
        self.assertEqual(controller.choice(100), 1)

    def test_audio_scaling_and_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            wavfile.write(path, 8000, np.full((50, 2), 16384, dtype=np.int16))
            audio, _rate = read_audio(path)
            np.testing.assert_equal(audio, 0.5)
            wavfile.write(path, 8000, np.full(50, 128, dtype=np.uint8))
            np.testing.assert_equal(read_audio(path)[0], 0)
        mixer = AudioMixer(8000)
        mixed = mixer.process(np.ones((100, 2)), [1, 1])
        self.assertLessEqual(float(mixed.max()), 0.981)

    def test_lags_and_model_roundtrip(self):
        random = np.random.default_rng(3)
        envelopes = random.normal(size=(500, 2))
        eeg = np.zeros((500, 1))
        eeg[2:, 0] = envelopes[:-2, 0]
        config = AuditoryConfig(lag_seconds=2 / 64)
        window = AuditoryWindow(
            eeg, envelopes, np.arange(500) / 64, 8, contract={"eeg_channels": ["F3"]}
        )
        model = RidgeDecoder(config, 0.01).fit([(window, np.zeros(500, dtype=int))])
        self.assertGreater(model.score(window)[0], 0.99)
        swapped = AuditoryWindow(
            eeg, envelopes[:, ::-1], window.timestamps, 8, contract=window.contract
        )
        np.testing.assert_allclose(model.score(window)[::-1], model.score(swapped))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = RidgeDecoder.load(path)
            np.testing.assert_equal(model.score(window), restored.score(window))
        window.contract = {"channels": ["wrong"]}
        with self.assertRaises(ValueError):
            model.score(window)

    def test_replay_chunk_invariance(self):
        trial = synthetic_trial()
        first = list(replay_windows(trial, AuditoryConfig(), chunk_seconds=0.032))
        second = list(replay_windows(trial, AuditoryConfig(), chunk_seconds=0.047))
        first = {round(w.timestamps[0], 6): w for w in first if w.valid}
        second = {round(w.timestamps[0], 6): w for w in second if w.valid}
        common = first.keys() & second.keys()
        self.assertGreater(len(common), 3)
        for key in common:
            np.testing.assert_allclose(first[key].eeg, second[key].eeg, atol=1e-8)
            np.testing.assert_allclose(
                first[key].envelopes, second[key].envelopes, atol=1e-8
            )

    def test_end_to_end_and_zero_model(self):
        model, _report = train(
            [synthetic_trial("train")], [synthetic_trial("validation")], alphas=(10.0,)
        )
        output, estimates, metrics = replay(synthetic_trial("test"), model)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertGreater(len(estimates), 3)
        self.assertGreater(metrics["correct_emphasis_seconds"], 0)
        assert model.weights is not None
        model.weights[:] = 0
        output, estimates, metrics = replay(synthetic_trial("test"), model)
        self.assertEqual(metrics["coverage"], 0)

    def test_split_and_latest_handoff(self):
        trial = synthetic_trial()
        with self.assertRaises(ValueError):
            check_split([trial], [trial])
        handoff = LatestEstimate()
        handoff.put(1)
        handoff.put(2)
        self.assertEqual(handoff.get(), 2)
        self.assertIsNone(handoff.get())


if __name__ == "__main__":
    unittest.main()
