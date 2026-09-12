"""Regression checks for the September 12 issue report, on the supported path."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.audio import read_audio
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import AttentionEstimate, load_trial, save_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import check_split, inject_fault, selection_metrics
from nova2026.auditory.streaming import AuditoryProcessor, stream_config
from nova2026.auditory.timing import TimestampedAudio, validate_audio_profile
from scripts.auditory.evaluate import score_controls
from scripts.auditory.replay import replay
from scripts.auditory.runner import ReplayFailure, replay_windows
from scripts.auditory.streamer import AuditoryReplayStreamer
from scripts.auditory.synthetic import synthetic_trial
from scripts.auditory.train import train


class ReportFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train([synthetic_trial("train")], [synthetic_trial("validation")], alphas=(10.,))[0]

    def test_confidence_rechecked_after_commitment(self):
        ctl = AttentionController(min_switch_windows=1)
        ctl.update(AttentionEstimate([.8, .1], 1, 1), 1)
        self.assertEqual(ctl.choice(1), 0)
        for time_, scores in enumerate(([0., 0.], [.06, .02], [-.5, -.5]), 2):
            ctl.update(AttentionEstimate(scores, time_, time_), time_)
            np.testing.assert_equal(ctl.gains(time_), [1, 1])

    def test_duplicate_and_expiry_and_required_clock(self):
        ctl = AttentionController(min_switch_windows=1)
        ctl.update(AttentionEstimate([.8, 0], 10, 10), 10)
        ctl.update(AttentionEstimate([0, .8], 10, 10.1), 10.1)
        self.assertEqual(ctl.choice(10.1), 0)
        self.assertIsNone(ctl.choice(14))
        with self.assertRaises(TypeError):
            ctl.gains()
        with self.assertRaises(ValueError):
            ctl.gains(float("nan"))

    def test_dwell_requires_consecutive_confident_switches(self):
        ctl = AttentionController(min_switch_windows=3)
        for t in (-1, 0, 1):
            ctl.update(AttentionEstimate([.8, 0], t, t), t)
        for t in (2, 3):
            ctl.update(AttentionEstimate([0, .8], t, t), t)
            self.assertEqual(ctl.choice(t), 0)
        ctl.update(AttentionEstimate([0, .8], 4, 4), 4)
        self.assertEqual(ctl.choice(4), 1)

    def test_wrong_width_rejected_before_broadcast(self):
        window = next(w for w in replay_windows(synthetic_trial("test"), self.model.config) if w.valid)
        window.eeg = window.eeg[:, :1]
        with self.assertRaises(ValueError):
            self.model.validate(window)
        with self.assertRaises(ValueError):
            self.model.score(window)

    def test_name_selection_and_missing_channels(self):
        trial = synthetic_trial("test")
        settings = stream_config(trial, self.model.config)
        a = AuditoryProcessor(settings)
        b = AuditoryProcessor(settings, ("F4", "extra", "F3"))
        rows = np.column_stack([trial.eeg[:, 1], np.zeros(len(trial.eeg)), trial.eeg[:, 0]])
        first = a.feed((trial.eeg, trial.timestamps))
        second = b.feed((rows, trial.timestamps))
        self.assertEqual(len(first), len(second))
        for x, y in zip(first, second):
            np.testing.assert_allclose(x.data, y.data)
        with self.assertRaises(ValueError):
            AuditoryProcessor(settings, ("F3",))

    def test_large_chunks_keep_every_hop(self):
        trial = synthetic_trial("test")
        proc = AuditoryProcessor(stream_config(trial, self.model.config))
        windows = []
        for left in range(0, len(trial.eeg), 700):
            windows.extend(proc.feed((trial.eeg[left:left+700], trial.timestamps[left:left+700])))
        np.testing.assert_equal(np.diff([w.start_sample for w in windows]), 64)
        self.assertGreater(len(windows), 15)

    def test_latency_inside_expiry(self):
        windows = [w for w in replay_windows(synthetic_trial("test"), self.model.config) if w.valid]
        ages = [w.available_at - w.timestamps[-1 - self.model.config.lag_samples] for w in windows]
        self.assertGreater(len(ages), 10)
        self.assertLess(max(ages), 3.)

    def test_transient_artifacts_continue(self):
        for duration in (.008, .25):
            trial = inject_fault(synthetic_trial("test"), "artifact", start=10, duration=duration)
            windows = list(replay_windows(trial, self.model.config))
            self.assertTrue(any("amplitude" in w.reasons for w in windows))
            self.assertTrue(any(w.valid and w.timestamps[0] > 13 for w in windows))

    def test_bad_chunks_rejected(self):
        for chunk in (0, -.032, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                list(replay_windows(synthetic_trial("test"), self.model.config, chunk_seconds=chunk))

    def test_short_audio_yields_diagnostics(self):
        trial = synthetic_trial("test")
        trial.audio = trial.audio[:trial.audio_rate.__int__()]
        windows = list(replay_windows(trial, self.model.config))
        self.assertGreater(len(windows), 0)
        self.assertTrue(all(not w.valid for w in windows))
        self.assertTrue(any("audio_unavailable" in w.reasons for w in windows))

    def test_library_rejects_development_data(self):
        with self.assertRaises(ValueError):
            replay(synthetic_trial("train"), self.model)

    def test_shared_individual_story_rejected(self):
        a, b = synthetic_trial("a"), synthetic_trial("b")
        a.group, b.group = "story1|story2", "story2|story3"
        with self.assertRaises(ValueError):
            check_split([a], [b])

    def test_metrics_include_partial_final_block(self):
        metrics = selection_metrics([0, .5, 1, 1.25], [0]*4, [0]*4, end_time=1.5)
        self.assertEqual(metrics["known_seconds"], 1.5)
        metrics = selection_metrics([0, .5, 1, 1.5], [0, 0, 1, 1], [0, 1, 1, 1], end_time=2)
        self.assertEqual(metrics["false_selection_changes"], 0)
        trial = synthetic_trial("test")
        audio, _, metrics = replay(trial, self.model)
        self.assertAlmostEqual(metrics["known_seconds"], len(audio) / trial.audio_rate)

    def test_noise_and_zero_abstain(self):
        for seed in range(3):
            trial = synthetic_trial(f"noise{seed}")
            trial.eeg = np.random.default_rng(seed).normal(size=trial.eeg.shape)
            _, _, metrics = replay(trial, self.model)
            self.assertLess(metrics["coverage"], .2)
        zero = copy.deepcopy(self.model)
        zero.weights[:] = 0
        _, _, metrics = replay(synthetic_trial("test"), zero)
        self.assertEqual(metrics["coverage"], 0)

    def test_metadata_missing_key_and_bad_weight_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(path)
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            meta = json.loads(str(arrays["metadata"]))
            meta["features"].pop("band")
            arrays["metadata"] = json.dumps(meta)
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                RidgeDecoder.load(path)
            with self.assertRaises(ValueError):
                self.model.save(Path(directory) / "wrong.npy")

    def test_nonfinite_trial_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.npz"
            trial = synthetic_trial("test")
            trial.eeg[0, 0] = np.nan
            save_trial(trial, path)
            with self.assertRaises(ValueError):
                load_trial(path)

    def test_zero_availability_is_preserved(self):
        from nova2026.auditory.alignment import EnvelopeBuffer
        raw = SimpleNamespace(data=np.zeros((64, 2)), timestamps=np.arange(-64, 0)/64,
                              available_at=0., valid=True, reasons=(), contract={}, segment=0)
        self.assertEqual(EnvelopeBuffer(64).align(raw).available_at, 0.)

    def test_float_pcm_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            wavfile.write(path, 8000, np.array([-2., 1.], dtype=np.float32))
            np.testing.assert_equal(read_audio(path)[0], [-1., .5])

    def test_timestamp_mapping_matches_offline_envelopes(self):
        trial = synthetic_trial("test")
        provider = TimestampedAudio(trial.audio, trial.audio_rate, self.model.config)
        for p in range(0, len(trial.audio), 256):
            provider.record(p, min(256, len(trial.audio)-p), 100 + p / trial.audio_rate)
        windows = [w for w in replay_windows(trial, self.model.config) if w.valid]
        for window in windows:
            raw = SimpleNamespace(data=window.eeg, timestamps=window.timestamps + 100,
                                  available_at=window.available_at + 100, valid=True,
                                  reasons=(), contract=window.contract, segment=0)
            aligned = provider.align(raw)
            self.assertTrue(aligned.valid)
            np.testing.assert_allclose(aligned.envelopes, window.envelopes, atol=1e-10)
        self.assertLess(abs(provider.diagnostics()["estimated_audio_drift_ppm"]), 1e-6)

    def test_unobserved_audio_and_clock_jump_abstain(self):
        trial = synthetic_trial("test")
        provider = TimestampedAudio(trial.audio, trial.audio_rate, self.model.config)
        raw = SimpleNamespace(data=np.zeros((64, 2)), timestamps=np.arange(64)/64,
                              available_at=1., valid=True, reasons=(), contract={}, segment=0)
        self.assertFalse(provider.align(raw).valid)
        provider.record(0, 8000, 0.)
        provider.record(8000, 8000, 1.1)
        raw.timestamps += .5
        self.assertIn("audio_clock_discontinuity", provider.align(raw).reasons)

    def test_audio_profile_rejects_offset_and_configuration_changes(self):
        profile = dict(device="measured output", sample_rate=8000, block_size=256,
                       residual_offset_seconds=.015)
        self.assertEqual(validate_audio_profile(profile, 8000, 256), profile)
        with self.assertRaises(ValueError):
            validate_audio_profile({**profile, "residual_offset_seconds": .060}, 8000, 256)
        with self.assertRaises(ValueError):
            validate_audio_profile(profile, 8000, 512)

    def test_zero_control_is_not_rng_accuracy(self):
        for seed in (0, 1, 7):
            result = score_controls(synthetic_trial("test"), self.model, seed)
            self.assertIsNone(result["controls"]["zero"]["accuracy"])
            self.assertEqual(result["controls"]["zero"]["committed"], 0)
            self.assertEqual(result["controls"]["real"]["accuracy"], 1.)

    def test_warm_start_roundtrip_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            self.model.save(path)
            base = RidgeDecoder.load(path)
            model, _ = train([synthetic_trial("personal")], [synthetic_trial("newvalid")], alphas=(10.,), base=base)
            self.assertTrue(np.all(np.isfinite(model.weights)))
            with self.assertRaises(ValueError):
                replay(synthetic_trial("train"), model)

    def test_eeg_failure_audio_continues_and_audio_error_wins(self):
        trial = synthetic_trial("test")
        trial.audio = trial.audio[:1024]
        for fail_audio in (False, True):
            writes = []
            class Device:
                def __init__(self, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
                def write(self, samples):
                    writes.append(len(samples))
                    time.sleep(.005)
                    if fail_audio:
                        raise OSError("audio failed")
                    return False
            streamer = AuditoryReplayStreamer(trial, self.model)
            streamer.initialize()
            with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Device)}), patch(
                "scripts.auditory.streamer.replay_windows", side_effect=ReplayFailure(0, "EEG failed")
            ):
                with self.assertRaises(OSError if fail_audio else ReplayFailure):
                    streamer.stream()
            if not fail_audio:
                self.assertEqual(sum(writes), len(trial.audio))
                self.assertIsNone(streamer.controller.choice(streamer.clock()))

    def test_blocked_device_has_bounded_shutdown(self):
        from threading import Event
        release = Event()
        trial = synthetic_trial("test")
        trial.audio = trial.audio[:256]
        class Device:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def write(self, samples):
                release.wait(10)
                return False
        streamer = AuditoryReplayStreamer(trial, self.model)
        streamer.initialize()
        started = time.monotonic()
        try:
            with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Device)}), patch(
                "scripts.auditory.streamer.replay_windows", return_value=iter(())
            ):
                with self.assertRaisesRegex(RuntimeError, "did not stop"):
                    streamer.stream()
            self.assertLess(time.monotonic() - started, 8)
        finally:
            release.set()
            streamer.worker.join(1)


if __name__ == "__main__":
    unittest.main()
