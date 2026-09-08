"""Real-time neuro-steered audio demo — the whole loop in one place.

    EEG  ─▶ backward decoder ─▶ reconstructed envelope ─┐
                                                        ├─▶ which talker? ─▶ gains
    two talkers ─▶ live envelopes ──────────────────────┘                    │
    two talkers ─▶ AttentionMixer (ramping gains) ─▶ output  ◀───────────────┘

Everything is pluggable so it runs today with no hardware and swaps to the real
rig on build day:

    EEG source   : --eeg replay   (a recorded trial, default)   |  --eeg lsl  (ANT Neuro / mne-lsl)
    audio source : --audio file   (two known tracks, default)   |  --audio mic (two clip-ons)
    output       : --output wav   (render to a file, default)    |  --output play (live, sounddevice)

Development default (no hardware, fully offline):
    python demo_realtime.py --data-dir /path/to/AAD --seconds 45 --output wav

Build-day live version:
    python demo_realtime.py --data-dir /path/to/AAD --eeg lsl --audio mic --output play
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from scipy.signal import butter, sosfiltfilt

from config import (FS, AUDIO_FS, BLOCK_S, DECISION_STEP_S, DECISION_WINDOW_S, ALPHA, BAND)
from envelopes import build_envelope_cache, envelope_from_array
from dataset import load_kuleuven_subject, preprocess_eeg
from decoder import fit_decoder, RealtimeDecoder, _corr
from attention_mixer import AttentionMixer
from audio_sources import FileSource, MicSource
from eeg_sources import ReplayEEG, LSLEEG

# Same band the decoder was trained in — the live audio envelope must be filtered
# to it before correlating, or the reconstruction and envelope live in different
# frequency ranges and the comparison is meaningless.
_BP = butter(4, list(BAND), "bandpass", fs=FS, output="sos")


def _live_envelope(raw_audio):
    """Envelope of a raw audio window, band-passed to the decoder's band (at FS)."""
    env = envelope_from_array(raw_audio, AUDIO_FS).astype(np.float64)
    return sosfiltfilt(_BP, env)


def _corrs(recon, env_windows):
    """Correlate the EEG reconstruction with each source's envelope (trim to match)."""
    out = []
    for env in env_windows:
        m = min(len(recon), len(env))
        out.append(_corr(recon[-m:], env[-m:]))
    return out


def run_demo(args):
    stim_dir = os.path.join(args.data_dir, "stimuli")
    stim_dir = stim_dir if os.path.isdir(stim_dir) else args.data_dir
    env_cache = build_envelope_cache(stim_dir, os.path.join(args.out, "envelopes_64hz.npz"))

    # ---- train the decoder on all trials except the one we will replay ----------
    # Train on filtered EEG (offline convention); replay streams the RAW EEG and the
    # loop applies preprocess_eeg per window — the same transform the live headset
    # path uses — so training and live inference see identically-shaped inputs.
    mat = os.path.join(args.data_dir, f"{args.subject}.mat")
    trials = load_kuleuven_subject(mat, env_cache, filtered=True)
    held = args.trial if args.trial is not None else 0
    decoder = RealtimeDecoder(fit_decoder([t for i, t in enumerate(trials) if i != held], ALPHA))
    trial = trials[held]
    print(f"Subject {args.subject}, replaying trial {held} "
          f"(attended track = {trial['att_fn']}).")

    # ---- wire up the pluggable sources -----------------------------------------
    if args.eeg == "replay":
        raw_eeg = load_kuleuven_subject(mat, env_cache, filtered=False)[held]["eeg"]
        eeg_src = ReplayEEG(raw_eeg)
    else:
        eeg_src = LSLEEG(args.lsl_name)
    if args.audio == "file":
        if args.files:
            a_path, b_path = args.files
        else:                                  # replay: play this trial's own two talkers
            def _find(fn):
                for r, _, fs in os.walk(stim_dir):
                    if fn in fs:
                        return os.path.join(r, fn)
                raise FileNotFoundError(fn)
            a_path, b_path = _find(trial["att_fn"]), _find(trial["unatt_fn"])
        audio_src = FileSource(a_path, b_path)
    else:
        audio_src = MicSource()

    mixer = AttentionMixer(n_sources=2)

    # ---- block clock ------------------------------------------------------------
    # EEG (FS) and audio (AUDIO_FS) advance at different rates; we read EEG to match
    # elapsed audio time exactly each block so the two never drift out of alignment.
    win_audio = int(DECISION_WINDOW_S * AUDIO_FS)
    decide_every = max(1, int(round(DECISION_STEP_S / BLOCK_S)))
    max_blocks = int(args.seconds / BLOCK_S) if args.seconds else 10**9
    block_samples = int(round(BLOCK_S * AUDIO_FS))

    ring_a = np.zeros(0, np.float32); ring_b = np.zeros(0, np.float32)
    eeg_ring = np.zeros((0, trial["eeg"].shape[1]))
    win_eeg = int(DECISION_WINDOW_S * FS)
    audio_samples = 0; eeg_read = 0

    out_audio = []          # mixed output blocks
    log = []                # (t, corr_att, corr_unatt, attended_idx, gainA_db, gainB_db)
    sink = _make_sink(args)

    for blk in range(max_blocks):
        a, b = audio_src.read()
        if a is None:
            break
        audio_samples += len(a)
        target_eeg = int(round(audio_samples * FS / AUDIO_FS))
        eeg_block = eeg_src.read(target_eeg - eeg_read)     # aligned to audio time
        if eeg_block is None:
            break
        eeg_read = target_eeg

        ring_a = np.concatenate([ring_a, a])[-win_audio:]
        ring_b = np.concatenate([ring_b, b])[-win_audio:]
        eeg_ring = np.vstack([eeg_ring, eeg_block])[-win_eeg:]

        if blk % decide_every == 0 and len(eeg_ring) >= FS * 2:   # ≥2 s of EEG
            recon = decoder.reconstruct(preprocess_eeg(eeg_ring))  # same prep as live
            envs = [_live_envelope(ring_a), _live_envelope(ring_b)]
            cA, cB = _corrs(recon, envs)
            att = mixer.set_decision([cA, cB])
            gdb = mixer.gains_db
            log.append((blk * BLOCK_S, cA, cB, att, gdb[0], gdb[1]))

        out_block = mixer.process([a, b])
        sink(out_block)
        out_audio.append(out_block)

    if hasattr(audio_src, "close"):
        audio_src.close()
    if hasattr(sink, "close"):
        sink.close()

    out = np.concatenate(out_audio) if out_audio else np.zeros(0, np.float32)
    _finish(args, out, log)
    return log


def _make_sink(args):
    """Return a callable that consumes each mixed block (live playback if requested)."""
    if args.output == "play":
        import sounddevice as sd                     # hardware only
        stream = sd.OutputStream(samplerate=AUDIO_FS, channels=1, dtype="float32")
        stream.start()
        fn = lambda block: stream.write(np.ascontiguousarray(block, dtype=np.float32))
        fn.close = lambda: (stream.stop(), stream.close())
        return fn
    return lambda block: None                        # wav mode collects in run_demo


def _finish(args, out, log):
    import scipy.io.wavfile as wav
    if args.output == "wav" and len(out):
        os.makedirs(args.out, exist_ok=True)
        peak = np.max(np.abs(out)) or 1.0
        wav.write(os.path.join(args.out, args.out_wav),
                  AUDIO_FS, (0.98 * out / peak * 32767).astype(np.int16))
        print(f"Wrote {os.path.join(args.out, args.out_wav)}")
    if log:
        correct = sum(1 for _, cA, cB, att, *_ in log if att == 0)  # track 0 = attended (replay)
        print(f"Decisions: {correct}/{len(log)} favoured the attended talker.")
        print(f"Final gains: A={log[-1][4]:+.1f} dB  B={log[-1][5]:+.1f} dB")


def build_parser():
    p = argparse.ArgumentParser(description="Real-time neuro-steered audio demo.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subject", default="S3")
    p.add_argument("--trial", type=int, default=0, help="Index of the trial to replay")
    p.add_argument("--eeg", choices=["replay", "lsl"], default="replay")
    p.add_argument("--audio", choices=["file", "mic"], default="file")
    p.add_argument("--output", choices=["wav", "play"], default="wav")
    p.add_argument("--files", nargs=2, metavar=("A.wav", "B.wav"),
                   help="Override: two audio files for --audio file")
    p.add_argument("--seconds", type=float, default=45.0, help="0 = run to end of stream")
    p.add_argument("--lsl-name", default=None, help="LSL stream name for --eeg lsl")
    p.add_argument("--out", default="results")
    p.add_argument("--out-wav", default="demo_output.wav")
    return p


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
