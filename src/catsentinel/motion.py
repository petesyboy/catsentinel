"""Cheap frame-differencing gate so the expensive ONNX detector only runs on frames
that actually changed, keeping continuous operation light on the Pi's CPU.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, min_area_fraction: float = 0.01, history_frames: int = 2):
        self._min_area_fraction = min_area_fraction
        self._history: list[np.ndarray] = []
        self._history_frames = max(1, history_frames)

    @staticmethod
    def _prep(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def detect(self, frame: np.ndarray) -> bool:
        """Return True if this frame differs enough from recent history to be 'motion'."""
        prepped = self._prep(frame)

        if not self._history:
            self._history.append(prepped)
            return False

        reference = self._history[-1]
        diff = cv2.absdiff(reference, prepped)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        changed_fraction = float(np.count_nonzero(thresh)) / thresh.size

        self._history.append(prepped)
        if len(self._history) > self._history_frames:
            self._history.pop(0)

        return changed_fraction >= self._min_area_fraction
