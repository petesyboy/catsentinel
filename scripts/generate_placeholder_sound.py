"""Synthesizes a placeholder hissing-snake deterrent sound at sounds/deterrent.wav.

Pure numpy + the stdlib `wave` module (no extra audio-synthesis dependency) --
filtered noise shaped into a hiss with a sharp attack and a long, slightly
trembling decay. Swap this file out for a real recording whenever you like;
nothing else needs to change.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "sounds" / "deterrent.wav"
SAMPLE_RATE = 44100
DURATION_SECONDS = 1.6


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def make_hiss(rng: np.random.Generator) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION_SECONDS)
    noise = rng.standard_normal(n)

    # Band-pass the white noise into a hiss-like range: remove rumble (high-pass
    # via subtracting a wide moving average) then soften harsh top end (low-pass
    # via a narrow moving average).
    highpassed = noise - moving_average(noise, window=25)
    hiss = moving_average(highpassed, window=3)

    # Envelope: fast attack, a slightly trembling sustain (like a real hiss
    # isn't perfectly flat), then a longer decay tail.
    t = np.linspace(0, DURATION_SECONDS, n)
    attack = np.clip(t / 0.03, 0, 1)
    decay = np.exp(-np.clip(t - 0.25, 0, None) / 0.5)
    tremor = 1.0 + 0.15 * np.sin(2 * np.pi * 28 * t + rng.uniform(0, 2 * np.pi))
    envelope = attack * decay * tremor

    signal = hiss * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.9
    return signal


def main() -> None:
    rng = np.random.default_rng(seed=1)
    signal = make_hiss(rng)
    pcm = (signal * 32767).astype(np.int16)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    print(f"Wrote {OUTPUT_PATH} ({DURATION_SECONDS}s, {SAMPLE_RATE}Hz mono)")


if __name__ == "__main__":
    main()
