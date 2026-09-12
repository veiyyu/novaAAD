"""Wire the AAD tube onto NOVA2026's current streaming API (main @ da4d4c8).

NOVA2026 rebuilt streaming around a tube `Pipeline` fed by an `AcquisitionQueue`
(the old `Streamer`/`EEGWindow`/`on_result` model this file used to target is
gone). The live path is:

    DefaultStream (connect + validate + ChannelSelectionContract)
        -> AcquisitionQueue.get()   -> validated (data, timestamps) chunks
        -> WindowAssembler          -> fixed-length AADWindow (this file)
        -> Pipeline[..., AADTube]   -> AADResult
        -> ReliabilityController    -> per-talker gains
        -> audio_thread             -> mixed audio out (independent of inference)

NOVA imports are done lazily inside functions so the rest of novaAAD (and its
tests) run without NOVA installed. Run this only where NOVA2026 is importable
and its streaming deps (mne, mne-lsl + liblsl) are present:

    NOVA_REPO=/path/to/NOVA2026 \\
    PYTHONPATH=$NOVA_REPO/src:$NOVA_REPO:. \\
        python nova/run_adapter_replay.py --decoder nova_aad.npz --duration 30

THREE PIECES A REAL DEPLOYMENT STILL NEEDS (see NOVA_INTEGRATION.md):
  1. A decoder TRAINED on NOVA's montage (64 ch @ 128 Hz), saved WITH channel
     names, so the tube can align by name. The KU Leuven decoder does not
     transfer blind; collect two-talker AAD on the NOVA rig and fit it.
  2. A live candidate-envelope provider: envelopes(t0, t1, n) returning the two
     talkers' envelopes on the window's timestamp grid, from the audio actually
     presented to the listener.
  3. A connected/replayed LSL EEG stream for DefaultStream to validate and
     AcquisitionQueue to consume.
All three are marked TODO below.
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
from nova.aad_adapter import AADTube, AADWindow, ReliabilityController, NOVA_SFREQ


class WindowAssembler:
    """Turn a stream of validated (data, timestamps) chunks into fixed-length
    AADWindows that hop forward by `step_s`.

    NOVA's AcquisitionQueue hands out chunks of arbitrary length; the decoder
    wants a steady window. This keeps a ring of the most recent samples and emits
    an AADWindow every `step_s` once at least `window_s` of data is buffered.
    """

    def __init__(self, channel_names, sfreq=NOVA_SFREQ, window_s=5.0, step_s=1.0):
        self.channel_names = tuple(channel_names)
        self.sfreq = float(sfreq)
        self.win = int(round(window_s * sfreq))
        self.step = int(round(step_s * sfreq))
        self._buf = np.empty((0, len(self.channel_names)), dtype=np.float64)
        self._ts = np.empty((0,), dtype=np.float64)
        self._since = 0

    def push(self, data, timestamps):
        self._buf = np.vstack([self._buf, np.asarray(data, dtype=np.float64)])
        self._ts = np.concatenate([self._ts, np.asarray(timestamps, dtype=np.float64)])
        self._since += len(data)
        keep = max(self.win, self.step)                # never grow unbounded
        if len(self._buf) > keep:
            self._buf = self._buf[-keep:]; self._ts = self._ts[-keep:]

    def maybe_emit(self):
        """Return an AADWindow if a full window has accrued since the last emit.

        Subtracts (not zeroes) the hop from the accrued counter so emit points do
        not drift when chunks are irregular — successive windows stay one `step`
        apart on average.
        """
        if len(self._buf) >= self.win and self._since >= self.step:
            self._since -= self.step
            return AADWindow(data=self._buf[-self.win:].copy(),
                             timestamps=self._ts[-self.win:].copy(),
                             channel_names=self.channel_names, valid=True)
        return None


def audio_thread(controller, get_blocks, out_write, stop):
    """Independent audio path: each block, read the latest gains and mix + play.

    Decoupled from inference so slow/stale decoding never stalls or corrupts
    audio (the review's requirement). `get_blocks()` -> list of per-talker blocks
    (or None to end); `out_write(mixed)` plays one block.
    """
    while not stop.is_set():
        blocks = get_blocks()
        if blocks is None:
            break
        g = controller.gains(now_source_time=time.monotonic())   # freshness re-checked
        mixed = np.clip(sum(gi * b for gi, b in zip(g, blocks)), -1.0, 1.0)
        out_write(mixed.astype(np.float32))


def build_stream_and_queue(expected_channel_names):
    """Construct NOVA's DefaultStream + AcquisitionQueue. NOVA must be importable.

    TODO(3): connect DefaultStream to a live or replayed LSL EEG outlet and start
    acquisition. The exact connect call depends on your amplifier/replay setup;
    see scripts/dataproc/streaming/streams.py (DefaultStream.connect_stream) and
    scripts/dataproc/streaming/acquisition_queue.py (AcquisitionQueue).
    """
    from scripts.dataproc.streaming.streams import DefaultStream          # noqa: F401
    from scripts.dataproc.streaming.acquisition_queue import AcquisitionQueue  # noqa: F401
    raise NotImplementedError(
        "Wire DefaultStream.connect_stream(...) to your LSL outlet and construct "
        "AcquisitionQueue for it (TODO 3). See NOVA_INTEGRATION.md."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="AAD decoder .npz trained on NOVA's 64-ch montage")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    from nova2026.data.pipeline import Pipeline

    w, meta = load_decoder(args.decoder)
    decoder = RealtimeDecoder(w, meta)
    if meta.get("n_channels") != 64:
        print(f"WARNING: decoder expects {meta.get('n_channels')} ch; NOVA delivers 64. "
              "Retrain on the NOVA montage (see NOVA_INTEGRATION.md).")
    if not meta.get("channel_names"):
        print("WARNING: decoder saved without channel_names; the tube cannot align by "
              "name and falls back to a positional channel-count check.")

    # TODO(2): supply the two talkers' envelopes on the window's timestamp grid.
    def envelopes(t0, t1, n):
        raise NotImplementedError("Provide live candidate envelopes aligned to window timestamps.")

    # min_switch_windows=3: require a new talker to win 3 consecutive windows before
    # the audio switches, so momentary noise never flips the mix (review finding 6).
    controller = ReliabilityController(min_switch_windows=3)
    tube = AADTube(decoder, envelopes, in_fs=NOVA_SFREQ)

    # A NOVA Pipeline whose (only) tube is the AAD decision. Preprocessing (band +
    # resample) happens inside the tube's window_to_decoder_input, so no extra tube
    # is required; add a DefaultPipe tube here if you prefer NOVA-side filtering.
    pipeline = Pipeline()
    pipeline.add_tube(tube)

    stream, queue = build_stream_and_queue(getattr(decoder.meta, "channel_names", None))  # TODO(3)
    assembler = WindowAssembler(meta.get("channel_names") or (), sfreq=NOVA_SFREQ,
                                window_s=args.window_s, step_s=args.step_s)

    stop = threading.Event()
    # TODO: wire get_blocks()/out_write() to the audio device (sounddevice) or a file.
    # threading.Thread(target=audio_thread, args=(controller, get_blocks, out_write, stop),
    #                  daemon=True).start()
    t_end = time.monotonic() + args.duration
    try:
        while time.monotonic() < t_end:
            data, timestamps = queue.get(timeout=0.1)     # validated chunk (raises on bad data)
            assembler.push(data, timestamps)
            window = assembler.maybe_emit()
            if window is not None:
                result, _ = pipeline.rundown(window)       # tube: (AADResult, window)
                controller.update(result)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
