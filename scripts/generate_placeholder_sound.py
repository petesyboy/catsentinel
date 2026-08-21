"""Synthesizes placeholder sounds: a hissing-snake deterrent at sounds/deterrent.wav
and a short chime notification at sounds/notification.wav.

Pure numpy + the stdlib `wave` module (no extra audio-synthesis dependency) --
filtered noise shaped into a hiss with a sharp attack and a long, slightly
trembling decay for the deterrent; two quick sine-wave tones for the chime.
Swap either file out for a real recording whenever you like; nothing else
needs to change.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DETERRENT_OUTPUT_PATH = ROOT / "sounds" / "deterrent.wav"
NOTIFICATION_OUTPUT_PATH = ROOT / "sounds" / "notification.wav"
SAMPLE_RATE = 44100
DURATION_SECONDS = 1.6
CHIME_DURATION_SECONDS = 0.35


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


def make_chime() -> np.ndarray:
    """A quick two-note upward chime -- just enough to notice a cat was seen,
    unlike the deterrent hiss which is meant to startle."""
    n = int(SAMPLE_RATE * CHIME_DURATION_SECONDS)
    t = np.linspace(0, CHIME_DURATION_SECONDS, n)

    note_split = n // 2
    freq = np.where(np.arange(n) < note_split, 880.0, 1320.0)  # A5 then E6
    tone = np.sin(2 * np.pi * freq * t)

    envelope = np.ones(n)
    fade = max(1, n // 10)
    envelope[:fade] *= np.linspace(0, 1, fade)
    envelope[-fade:] *= np.linspace(1, 0, fade)

    signal = tone * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.6
    return signal


def _write_wav(path: Path, signal: np.ndarray) -> None:
    pcm = (signal * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    rng = np.random.default_rng(seed=1)

    _write_wav(DETERRENT_OUTPUT_PATH, make_hiss(rng))
    print(f"Wrote {DETERRENT_OUTPUT_PATH} ({DURATION_SECONDS}s, {SAMPLE_RATE}Hz mono)")

    _write_wav(NOTIFICATION_OUTPUT_PATH, make_chime())
    print(f"Wrote {NOTIFICATION_OUTPUT_PATH} ({CHIME_DURATION_SECONDS}s, {SAMPLE_RATE}Hz mono)")


if __name__ == "__main__":
    main()
