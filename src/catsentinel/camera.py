"""Camera capture abstraction.

Two backends behind one interface, selected via config.yaml `camera.backend`:
  - "opencv": works with a laptop webcam or a USB camera on the Pi.
  - "picamera2": the official Raspberry Pi camera module.
Swapping backends is the only change needed to move from dev laptop to the Pi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next frame as a BGR numpy array, or None if unavailable."""

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OpenCVCamera(Camera):
    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 10):
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class PiCamera2Camera(Camera):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 10):
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self) -> np.ndarray | None:
        # picamera2 delivers RGB888; downstream code (OpenCV-based) expects BGR.
        frame = self._picam2.capture_array()
        return frame[:, :, ::-1]

    def close(self) -> None:
        self._picam2.stop()


def build_camera(camera_config) -> Camera:
    backend = camera_config.backend
    width, height, fps = camera_config.width, camera_config.height, camera_config.fps
    if backend == "opencv":
        return OpenCVCamera(device=camera_config.device, width=width, height=height, fps=fps)
    if backend == "picamera2":
        return PiCamera2Camera(width=width, height=height, fps=fps)
    raise ValueError(f"Unknown camera backend: {backend!r}")
