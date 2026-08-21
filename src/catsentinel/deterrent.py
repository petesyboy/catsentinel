"""Deterrent actions, triggered when a stranger cat is confirmed.

v1 implements sound only. A water-sprayer relay is a natural v2 (see
WaterSprayerDeterrent TODO below) once that hardware is on hand -- it plugs into
the same interface, so pipeline.py never needs to change.
"""
from __future__ import annotations

import platform
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class Deterrent(ABC):
    @abstractmethod
    def trigger(self) -> None:
        """Fire the deterrent. Called at most once per cooldown window by the pipeline."""


class SoundDeterrent(Deterrent):
    """Plays a WAV file through the default audio output, asynchronously.

    Uses the OS's own playback mechanism (stdlib `winsound` on Windows,
    `aplay` as a subprocess on Linux/the Pi) rather than a third-party audio
    library -- `simpleaudio` turned out to be unreliable on this dev machine
    (it segfaulted), and because it ran in-process, that took the whole
    detection loop down with it. Both of these run outside the main process/
    thread and are wrapped defensively, so a playback failure can only ever
    log a warning, never freeze or crash the pipeline.
    """

    def __init__(self, sound_file: str):
        self._path = Path(sound_file)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Deterrent sound file not found: {self._path}. "
                "Place a .wav file there or update config.yaml's deterrent.sound_file."
            )
        self._is_windows = platform.system() == "Windows"

    def trigger(self) -> None:
        try:
            if self._is_windows:
                import winsound

                winsound.PlaySound(str(self._path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                subprocess.Popen(
                    ["aplay", "-q", str(self._path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:  # noqa: BLE001 -- deterrent failures must never take down the pipeline
            print(f"[catsentinel] WARNING: deterrent sound failed to play: {e}")


class WaterSprayerDeterrent(Deterrent):
    """TODO (stretch goal, needs hardware): drive a 5V relay -> solenoid valve
    (or a servo-actuated squirt gun) from a Pi GPIO pin via gpiozero, for a
    short timed burst. Not implemented yet -- no relay/valve hardware on hand.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "WaterSprayerDeterrent requires a GPIO relay + solenoid valve (or "
            "servo squirt mechanism) that isn't wired up yet. Use 'sound' for now."
        )

    def trigger(self) -> None:
        raise NotImplementedError


def build_deterrent(deterrent_config) -> Deterrent:
    deterrent_type = deterrent_config.type
    if deterrent_type == "sound":
        return SoundDeterrent(sound_file=deterrent_config.sound_file)
    if deterrent_type == "water_sprayer":
        return WaterSprayerDeterrent()
    raise ValueError(f"Unknown deterrent type: {deterrent_type!r}")
