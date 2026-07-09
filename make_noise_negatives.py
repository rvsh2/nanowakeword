#!/usr/bin/env python3
"""Add speech-free negatives to dataset v2.

The satellite spends ~99% of its life hearing noise or silence, yet the v1
dataset had zero speech-free examples labelled 0 — models scored white noise
0.77-0.95. This writes 2 s clips: random ESC-50 crops, white/pink noise at
assorted levels, and (near-)silence.
"""
import os
import random
import sys
import wave

import numpy as np

BG_DIR = "/opt/ha/nanowakeword/training_data/noise/background"
OUT = {
    "/opt/ha/nanowakeword/training_data/agata_v2/negative": 1500,
    "/opt/ha/nanowakeword/training_data/agata_v2/negative_val": 300,
}
SR = 16000
LEN = 2 * SR

rng = np.random.default_rng(42)
random.seed(42)
bg_files = sorted(os.listdir(BG_DIR))


def esc50_crop():
    path = os.path.join(BG_DIR, random.choice(bg_files))
    with wave.open(path, "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if len(audio) <= LEN:
        audio = np.tile(audio, int(np.ceil(LEN / max(len(audio), 1))))
    start = rng.integers(0, len(audio) - LEN + 1)
    return audio[start:start + LEN]


def white_noise():
    level = rng.uniform(100, 6000)
    return (rng.normal(0, level, LEN)).clip(-32768, 32767).astype(np.int16)


def pink_noise():
    level = rng.uniform(100, 6000)
    white = rng.normal(0, 1, LEN)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(LEN, 1 / SR)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, LEN)
    pink = pink / (pink.std() + 1e-9) * level
    return pink.clip(-32768, 32767).astype(np.int16)


def silence():
    level = rng.uniform(0, 30)  # true digital silence up to faint hiss
    return (rng.normal(0, level, LEN)).astype(np.int16)


KINDS = [(esc50_crop, 0.5), (white_noise, 0.15), (pink_noise, 0.15), (silence, 0.2)]

for out_dir, count in OUT.items():
    os.makedirs(out_dir, exist_ok=True)
    for i in range(count):
        fn = random.choices([k for k, _ in KINDS], weights=[w for _, w in KINDS])[0]
        audio = fn()
        with wave.open(os.path.join(out_dir, f"noise_{fn.__name__}_{i:05d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(audio.tobytes())
    print(f"{out_dir}: +{count} noise-only negatives", file=sys.stderr)
print("NOISE NEGATIVES DONE")
