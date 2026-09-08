"""Render a shareable demo VIDEO (MP4) from a replayed trial — no hardware needed.

Runs the full neuro-steering pipeline on recorded EEG + the trial's two talkers,
then animates what the system is doing (which talker is being amplified, the live
gains, and the brain-to-envelope match) and muxes it with the boosted audio. The
result is a self-contained clip you can play as the demo or keep as a backup.

    python render_demo.py --data-dir /path/to/AAD --subject S3 --trial 4 --seconds 40

Needs ffmpeg on the PATH (matplotlib uses it to write the video).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import numpy as np
import scipy.io.wavfile as wav

from config import FS, AUDIO_FS, BLOCK_S, DECISION_STEP_S, DECISION_WINDOW_S, ALPHA
from envelopes import build_envelope_cache
from dataset import load_kuleuven_subject, preprocess_eeg
from decoder import fit_decoder, RealtimeDecoder
from attention_mixer import AttentionMixer
from audio_sources import FileSource
from demo_realtime import _live_envelope, _corrs

A_COL, B_COL = "#2E86DE", "#E1A400"
BG = "#0d1117"; CARD = "#161c26"; INK = "#e6eaf0"; MUTE = "#8b97a6"; GOOD = "#37c26b"


def run_pipeline(args):
    stim = os.path.join(args.data_dir, "stimuli")
    stim = stim if os.path.isdir(stim) else args.data_dir
    env = build_envelope_cache(stim, os.path.join(args.out, "envelopes_64hz.npz"))
    mat = os.path.join(args.data_dir, f"{args.subject}.mat")
    trials = load_kuleuven_subject(mat, env, filtered=True)
    dec = RealtimeDecoder(fit_decoder([t for i, t in enumerate(trials) if i != args.trial], ALPHA))
    trial = trials[args.trial]
    raw_eeg = load_kuleuven_subject(mat, env, filtered=False)[args.trial]["eeg"]

    def _find(fn):
        for r, _, fs in os.walk(stim):
            if fn in fs:
                return os.path.join(r, fn)
        raise FileNotFoundError(fn)
    src = FileSource(_find(trial["att_fn"]), _find(trial["unatt_fn"]))
    mixer = AttentionMixer(n_sources=2)

    decide_every = max(1, int(round(DECISION_STEP_S / BLOCK_S)))
    win_audio = int(DECISION_WINDOW_S * AUDIO_FS)
    win_eeg = int(DECISION_WINDOW_S * FS)
    max_blocks = int(args.seconds / BLOCK_S)
    ring_a = np.zeros(0, np.float32); ring_b = np.zeros(0, np.float32)
    eeg_ring = np.zeros((0, raw_eeg.shape[1]))
    eeg = raw_eeg; eeg_read = 0; audio_samples = 0
    out = []; frames = []; cA = cB = 0.0

    for blk in range(max_blocks):
        a, b = src.read()
        if a is None:
            break
        audio_samples += len(a)
        target = int(round(audio_samples * FS / AUDIO_FS))
        if target > len(eeg):
            break
        eeg_ring = np.vstack([eeg_ring, eeg[eeg_read:target]])[-win_eeg:]
        eeg_read = target
        ring_a = np.concatenate([ring_a, a])[-win_audio:]
        ring_b = np.concatenate([ring_b, b])[-win_audio:]
        if blk % decide_every == 0 and len(eeg_ring) >= FS * 2:
            recon = dec.reconstruct(preprocess_eeg(eeg_ring))
            cA, cB = _corrs(recon, [_live_envelope(ring_a), _live_envelope(ring_b)])
            mixer.set_decision([cA, cB])
        out.append(mixer.process([a, b]))
        gdb = mixer.gains_db
        frames.append((blk * BLOCK_S, gdb[0], gdb[1], cA, cB, mixer.attended))

    audio = np.concatenate(out) if out else np.zeros(0, np.float32)
    return np.array(frames), audio, trial


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    frames, audio, trial = run_pipeline(args)
    os.makedirs(args.out, exist_ok=True)
    wav.write(os.path.join(args.out, "_demo_audio.wav"), AUDIO_FS,
              (0.98 * audio / (np.max(np.abs(audio)) or 1) * 32767).astype(np.int16))

    fps = 20
    t_end = frames[-1, 0]
    times = np.arange(0, t_end, 1 / fps)
    idx = np.searchsorted(frames[:, 0], times).clip(0, len(frames) - 1)
    F = frames[idx]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11, 6.2), dpi=120)
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.945, "Neuro-Steered Hearing — live attention decoding",
             ha="center", color=INK, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.905, "The system reads the listener's EEG and amplifies the talker "
             "they're attending to", ha="center", color=MUTE, fontsize=11)

    def card(x0):
        ax = fig.add_axes([x0, 0.30, 0.32, 0.50]); ax.set_facecolor(CARD)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_color("#2a333f"); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    axA, axB = card(0.10), card(0.58)
    barA = patches.Rectangle((0.30, 0.10), 0.40, 0.0, color=A_COL); axA.add_patch(barA)
    barB = patches.Rectangle((0.30, 0.10), 0.40, 0.0, color=B_COL); axB.add_patch(barB)
    axA.text(0.5, 0.90, "Talker A", ha="center", color=A_COL, fontsize=17, fontweight="bold")
    axB.text(0.5, 0.90, "Talker B", ha="center", color=B_COL, fontsize=17, fontweight="bold")
    statusA = axA.text(0.5, 0.02, "", ha="center", color=INK, fontsize=12, fontweight="bold")
    statusB = axB.text(0.5, 0.02, "", ha="center", color=INK, fontsize=12, fontweight="bold")
    dbA = axA.text(0.5, 0.52, "", ha="center", color=INK, fontsize=14)
    dbB = axB.text(0.5, 0.52, "", ha="center", color=INK, fontsize=14)

    axc = fig.add_axes([0.10, 0.10, 0.80, 0.14]); axc.set_facecolor(BG)
    axc.set_xlim(0, t_end); axc.set_ylim(-0.05, max(0.35, F[:, 3:5].max() * 1.1))
    axc.tick_params(colors=MUTE, labelsize=8)
    for s in axc.spines.values():
        s.set_color("#2a333f")
    axc.set_title("brain–voice match (EEG reconstruction correlation)", color=MUTE,
                  fontsize=9, loc="left")
    (lA,) = axc.plot([], [], color=A_COL, lw=2)
    (lB,) = axc.plot([], [], color=B_COL, lw=2)
    tclock = fig.text(0.90, 0.90, "", ha="right", color=MUTE, fontsize=11)

    def draw(i):
        t, gA, gB, cA, cB, att = F[i]
        for bar, g in ((barA, gA), (barB, gB)):
            h = np.clip((g + 12) / 12, 0.02, 1) * 0.70
            bar.set_height(h)
        att = int(round(att))
        for ax, status, db, g, on, name in (
                (axA, statusA, dbA, gA, att == 0, "A"),
                (axB, statusB, dbB, gB, att == 1, "B")):
            ax.set_facecolor("#12202b" if on else CARD)
            for s in ax.spines.values():
                s.set_color(GOOD if on else "#2a333f")
            status.set_text("● AMPLIFIED" if on else "○ turned down")
            status.set_color(GOOD if on else MUTE)
            db.set_text(f"{g:+.0f} dB")
        m = F[:i + 1]
        lA.set_data(m[:, 0], m[:, 3]); lB.set_data(m[:, 0], m[:, 4])
        tclock.set_text(f"t = {t:4.1f} s")
        return barA, barB, statusA, statusB, dbA, dbB, lA, lB, tclock

    anim = animation.FuncAnimation(fig, draw, frames=len(F), interval=1000 / fps, blit=False)
    silent = os.path.join(args.out, "_demo_silent.mp4")
    anim.save(silent, fps=fps, dpi=120,
              savefig_kwargs={"facecolor": BG},
              extra_args=["-pix_fmt", "yuv420p"])
    plt.close(fig)

    final = os.path.join(args.out, args.out_mp4)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent,
                    "-i", os.path.join(args.out, "_demo_audio.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", final], check=True)
    os.remove(silent); os.remove(os.path.join(args.out, "_demo_audio.wav"))
    print(f"Wrote {final}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subject", default="S3")
    p.add_argument("--trial", type=int, default=4)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--out", default="results")
    p.add_argument("--out-mp4", default="demo_video.mp4")
    render(p.parse_args())
