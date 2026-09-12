"""Timestamped prerecorded talkers with live EEG; WAV mode is paced in real time.

Run --help for source metadata required to validate a decoder before playback.
Sound-device mode uses PortAudio's outputBufferDacTime mapped to local LSL time.
WAV mode exercises software timing only and does not measure acoustic latency.
"""

import argparse
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.audio import AudioMixer, LatestEstimate
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import AttentionEstimate, load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.pipeline import AuditoryPipeline
from nova2026.auditory.streaming import AuditoryProcessor, stream_config
from nova2026.auditory.timing import TimestampedAudio, validate_audio_profile


def run(trial, model, stream, *, source_unit_exponent=0, output="wav", window=None, timing_profile=None):
    from mne_lsl.lsl import local_clock
    from nova2026.streaming import prepare
    from nova2026.streaming.bootstrap import StreamSession
    from nova2026.streaming.preprocess import unit_scaler

    frames = max(1, round(trial.audio_rate * .032))
    if output not in ("wav", "play"):
        raise ValueError("Unknown audio output mode.")
    if output == "play":
        timing_profile = validate_audio_profile(timing_profile, trial.audio_rate, frames)
        if timing_profile != model.training_info.get("audio_timing_profile"):
            raise ValueError("Audio configuration differs from model calibration; recalibrate.")
        print(f"Verified profile residual: {timing_profile['residual_offset_seconds'] * 1000:.1f} ms; "
              f"device={timing_profile['device']}, rate={trial.audio_rate:g}, block={frames}")
    else:
        print("Paced WAV: software clock test; no acoustic offset measurement.")

    history = model.training_info["history"] if window is None else window
    metadata = SimpleNamespace(sample_rate=float(stream.info["sfreq"]),
                               channel_names=tuple(model.contract["eeg_channels"]),
                               reference=trial.reference, upstream_processing=trial.upstream_processing)
    settings = stream_config(metadata, model.config, history)
    processor = AuditoryProcessor(settings)
    if processor.contract != model.contract:
        raise ValueError("Connected source/processing/window differs from model; retrain through this chain.")
    contract = prepare(stream, sfreq=settings.input_sfreq, channels=settings.eeg_channels,
                       n_eeg=len(settings.eeg_channels), source_unit_exponent=source_unit_exponent)
    args = SimpleNamespace(sfreq=settings.input_sfreq, out_sfreq=model.config.sample_rate,
                           window=history, hop=1., capacity=history + 4, warmup=2.,
                           block=max(1, round(settings.input_sfreq * .032)))
    session = StreamSession(stream, args, settings.eeg_channels, contract=contract)
    scaler = unit_scaler(source_unit_exponent, desired_exponent=-6)
    provider = TimestampedAudio(trial.audio, trial.audio_rate, model.config)
    controller, handoff = AttentionController(), LatestEstimate()
    mixer = AudioMixer(trial.audio_rate)
    pipeline = AuditoryPipeline(model, local_clock)
    stop = Event()
    errors, rendered, estimates = [], [], []
    position = 0

    def block(count, audible):
        nonlocal position
        remaining = min(count, len(trial.audio) - position)
        if remaining <= 0:
            return None
        estimate = handoff.get()
        if estimate is not None:
            controller.update(estimate, audible)
        result = mixer.process(trial.audio[position:position + remaining], controller.gains(audible))
        provider.record(position, remaining, audible)
        position += remaining
        rendered.append(result)
        return result

    def play():
        try:
            if output == "wav":
                epoch = local_clock()
                while not stop.is_set() and position < len(trial.audio):
                    audible = epoch + position / trial.audio_rate
                    if stop.wait(max(0, audible - local_clock())):
                        break
                    block(frames, audible)
            else:
                import sounddevice as sd

                def callback(outdata, count, time_info, status):
                    try:
                        if status:
                            raise RuntimeError(f"Audio device status: {status}")
                        audible = local_clock() + time_info.outputBufferDacTime - time_info.currentTime
                        result = block(count, audible)
                        outdata[:] = 0
                        if result is None or stop.is_set():
                            raise sd.CallbackStop
                        outdata[:len(result), 0] = result
                    except sd.CallbackStop:
                        raise
                    except Exception as error:
                        errors.append(error)
                        stop.set()
                        raise sd.CallbackAbort from error

                with sd.OutputStream(samplerate=trial.audio_rate, channels=1, dtype="float32",
                                     device=timing_profile["device"], blocksize=frames,
                                     callback=callback, finished_callback=stop.set):
                    stop.wait(len(trial.audio) / trial.audio_rate + 3)
        except Exception as error:
            errors.append(error)
        finally:
            stop.set()

    worker = Thread(target=play, daemon=True, name="timestamped-audio")
    worker.start()
    eeg_error = None
    try:
        while not stop.is_set():
            data, times = session.acquire.read(timeout=1.)
            data, times = session.ingest(data, times)
            data, times = scaler(data, times)
            for raw in processor.feed((data, times)):
                raw.available_at = local_clock()
                aligned = provider.align(raw)
                estimate, _ = pipeline.rundown(aligned)
                handoff.put(estimate)
                estimates.append(estimate.to_dict())
    except Exception as error:
        if not stop.is_set():
            eeg_error = error
            now = local_clock()
            handoff.put(AttentionEstimate(None, now, now, False, ("processing_failed",)))
    finally:
        # EEG failure releases acquisition but allows the independent audio to finish neutral.
        session.acquire.close()
        session.close(status="failed" if eeg_error else "completed")
        worker.join(timeout=max(0, (len(trial.audio) - position) / trial.audio_rate) + 3)
        stop.set()
    if worker.is_alive():
        raise RuntimeError("Audio worker exceeded its shutdown deadline.")
    if errors:
        raise errors[0] from eeg_error
    if eeg_error:
        raise eeg_error
    if not estimates:
        raise RuntimeError("No EEG decisions produced; run is not a successful live evaluation.")
    return np.concatenate(rendered), {"estimates": estimates, **provider.diagnostics()}


def main():
    from mne_lsl.stream import StreamLSL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, help="Candidate audio and declared source metadata")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--source-unit-exponent", type=int, default=0)
    parser.add_argument("--output", choices=("wav", "play"), default="wav")
    parser.add_argument("--window", type=float, help="Must match the trained model history")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timing-profile", help="Measured loopback/profile JSON required for --output play")
    args = parser.parse_args()
    trial, model = load_trial(args.trial), RidgeDecoder.load(args.model)
    stream = StreamLSL(bufsize=4., name=args.stream)
    stream.connect(acquisition_delay=None, processing_flags=["clocksync"], timeout=10)
    try:
        audio, report = run(trial, model, stream, source_unit_exponent=args.source_unit_exponent,
                            output=args.output, window=args.window,
                            timing_profile=json.loads(Path(args.timing_profile).read_text()) if args.timing_profile else None)
    finally:
        stream.disconnect()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination / "mixed.wav", round(trial.audio_rate), audio)
    (destination / "timing.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
