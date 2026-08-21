"""Main capture -> motion gate -> detect -> recognize -> deterrent -> log loop.

Requires `confirm_frames` consecutive "stranger" verdicts before triggering the
deterrent (avoids one noisy frame causing a false trigger), and enforces
`cooldown_seconds` between triggers (avoids re-triggering every frame while the
same stranger cat lingers in view).
"""
from __future__ import annotations

import time

from .camera import Camera
from .deterrent import Deterrent
from .detector import CatDetector
from .events import EventLog
from .motion import MotionGate
from .recognizer import CatRecognizer

_DEBUG_PRINT_INTERVAL_SECONDS = 0.5
_DEBUG_WINDOW_NAME = "Cat Sentinel (debug)"


class Pipeline:
    def __init__(
        self,
        camera: Camera,
        detector: CatDetector,
        recognizer: CatRecognizer,
        deterrent: Deterrent,
        event_log: EventLog,
        motion_gate: MotionGate | None = None,
        confirm_frames: int = 3,
        cooldown_seconds: float = 30.0,
        loop_delay_seconds: float = 0.0,
        debug: bool = False,
        notifier: Deterrent | None = None,
        notification_cooldown_seconds: float = 5.0,
    ):
        self._camera = camera
        self._detector = detector
        self._recognizer = recognizer
        self._deterrent = deterrent
        self._event_log = event_log
        self._motion_gate = motion_gate
        self._confirm_frames = confirm_frames
        self._cooldown_seconds = cooldown_seconds
        self._loop_delay_seconds = loop_delay_seconds
        self._debug = debug
        self._notifier = notifier
        self._notification_cooldown_seconds = notification_cooldown_seconds

        self._stranger_streak = 0
        self._last_trigger_time = 0.0
        self._last_notification_time = 0.0
        self._last_debug_print_time = 0.0
        self._quit_requested = False

    def _debug_print(self, message: str, force: bool = False) -> None:
        if not self._debug:
            return
        now = time.monotonic()
        if force or (now - self._last_debug_print_time) >= _DEBUG_PRINT_INTERVAL_SECONDS:
            print(f"[catsentinel:debug] {message}")
            self._last_debug_print_time = now

    def _show_debug_window(self, frame, box=None, label: str | None = None) -> None:
        if not self._debug:
            return
        import cv2

        display = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(_DEBUG_WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._quit_requested = True

    def step(self) -> None:
        frame = self._camera.read()
        if frame is None:
            self._debug_print("no frame from camera", force=True)
            return

        if self._motion_gate is not None and not self._motion_gate.detect(frame):
            self._debug_print("watching -- no motion")
            self._show_debug_window(frame, label="no motion")
            return

        detections = self._detector.detect(frame)
        if not detections:
            self._stranger_streak = 0
            self._debug_print("motion seen, but no cat detected")
            self._show_debug_window(frame, label="motion, no cat detected")
            return

        best = max(detections, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        if self._notifier is not None:
            now = time.monotonic()
            if (now - self._last_notification_time) >= self._notification_cooldown_seconds:
                self._notifier.trigger()
                self._last_notification_time = now

        is_our_cat, probability = self._recognizer.identify(crop)
        verdict = "mine" if is_our_cat else "stranger"
        deterrent_triggered = False

        if verdict == "stranger":
            self._stranger_streak += 1
            now = time.monotonic()
            if (
                self._stranger_streak >= self._confirm_frames
                and (now - self._last_trigger_time) >= self._cooldown_seconds
            ):
                self._deterrent.trigger()
                deterrent_triggered = True
                self._last_trigger_time = now
                print(f"[catsentinel] STRANGER cat confirmed (probability={probability:.2f}) -- deterrent triggered")
        else:
            self._stranger_streak = 0
            print(f"[catsentinel] Your cat (probability={probability:.2f})")

        self._debug_print(
            f"cat detected (confidence={best.confidence:.2f}) -> {verdict} (probability={probability:.2f})",
            force=True,
        )
        self._show_debug_window(
            frame, box=best.box, label=f"{verdict} ({probability:.2f})"
        )

        self._event_log.record(
            frame=frame,
            verdict=verdict,
            probability=probability,
            detector_confidence=best.confidence,
            deterrent_triggered=deterrent_triggered,
        )

    def run(self) -> None:
        print("[catsentinel] Starting pipeline. Press Ctrl+C to stop" + (", or 'q' in the debug window." if self._debug else "."))
        try:
            while not self._quit_requested:
                self.step()
                if self._loop_delay_seconds:
                    time.sleep(self._loop_delay_seconds)
        except KeyboardInterrupt:
            print("[catsentinel] Stopping.")
        finally:
            if self._debug:
                import cv2

                cv2.destroyAllWindows()
            self._camera.close()
            self._event_log.close()
