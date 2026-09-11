"""Wire the AAD adapter onto NOVA2026's real Streamer (replay, no EEG hardware).

This is the runnable bridge: it plays a recording into LSL, runs NOVA's `Streamer`
with our `AADPipeline` as the per-window decoder and our `ReliabilityController` as the
result sink, and drives audio gains from an INDEPENDENT thread (so slow inference or a
lost stream can never stall the audio path).

Run it only where NOVA's deps are installed (mne, mne-lsl + liblsl, soxr) and the
NOVA repo is importable:

    NOVA_REPO=/path/to/NOVA2026 PYTHONPATH=$NOVA_REPO/scripts:$NOVA_REPO/src:. \\
        python nova/run_adapter_replay.py --recording run.fif --decoder nova_aad.npz

TWO PIECES A REAL DEPLOYMENT STILL NEEDS (see NOVA_INTEGRATION.md):
  1. A decoder TRAINED on NOVA's montage (56 ch / 128 Hz) — the KU Leuven 64ch decoder
     cannot transfer. Collect two-talker AAD on the NOVA rig and train with fit_decoder.
  2. A live candidate-envelope provider that returns the two talkers' envelopes on the
     window's timestamp grid (from the audio actually presented to the listener).
Both are marked TODO below.
"""
from __future__ import annotations
import argparse
import os
import sys
import threading
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder import load_decoder, RealtimeDecoder
from nova.aad_adapter import AADPipeline, ReliabilityController


def build_streamer(recording_path, decoder, envelopes, controller):
    """Construct NOVA's Streamer + PlayerLSL for a recording. NOVA deps required."""
    import mne
    from mne_lsl.player import PlayerLSL
    from streaming.config import StreamConfig          # NOVA (on PYTHONPATH)
    from streaming.streamer import Streamer

    raw = mne.io.read_raw(recording_path, preload=True)
    source_id = f"nova-aad-replay-{os.getpid()}"
    config = StreamConfig(
        input_sfreq=float(raw.info["sfreq"]),
        source_unit_exponent=0,                        # 0=V, -6=µV — must match player
        stream_name="NOVA-AAD-Replay", source_id=source_id,
        eog_channels=("EOG",),
    )
    player = PlayerLSL(raw, chunk_size=10, n_repeat=1,
                       name=config.stream_name, source_id=source_id, annotations=False)

    pipeline = AADPipeline(decoder, envelopes, in_fs=config.output_sfreq)
    streamer = Streamer(config, pipeline=pipeline, on_result=controller.update)
    return player, streamer


def audio_thread(controller, get_blocks, out_write, block_s=0.032, stop=None):
    """Independent audio path: every block, read the latest gains and mix + play.

    Decoupled from inference — the review's requirement that slow/stale decoding never
    stalls or corrupts the audio. `get_blocks()` -> list of per-source blocks;
    `out_write(mixed)` plays one block.
    """
    while stop is None or not stop.is_set():
        blocks = get_blocks()
        if blocks is None:
            break
        g = controller.gains(now_source_time=time.monotonic())   # freshness re-checked here
        mixed = np.clip(sum(gi * b for gi, b in zip(g, blocks)), -1.0, 1.0)
        out_write(mixed.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True, help="EEG recording to replay (.fif/.set/.cnt)")
    ap.add_argument("--decoder", required=True, help="AAD decoder .npz trained on NOVA's montage")
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    w, meta = load_decoder(args.decoder)
    decoder = RealtimeDecoder(w, meta)
    if meta.get("n_channels") != 56:
        print(f"WARNING: decoder expects {meta.get('n_channels')} ch; NOVA delivers 56. "
              "Retrain on the NOVA montage (see NOVA_INTEGRATION.md).")

    # TODO(1): supply the two talkers' envelopes on the window's timestamp grid.
    def envelopes(t0, t1, n):
        raise NotImplementedError("Provide live candidate envelopes aligned to window timestamps.")

    controller = ReliabilityController()
    player, streamer = build_streamer(args.recording, decoder, envelopes, controller)

    stop = threading.Event()
    # TODO(2): wire get_blocks()/out_write() to the real audio device (sounddevice) or a file.
    # threading.Thread(target=audio_thread, args=(controller, get_blocks, out_write),
    #                  kwargs=dict(stop=stop), daemon=True).start()
    try:
        player.start(); streamer.initialize()
        stats = streamer.stream(duration=args.duration)
        print("stream stats:", stats)
    finally:
        stop.set(); streamer.close()
        if getattr(player, "running", False):
            player.stop()


if __name__ == "__main__":
    main()
