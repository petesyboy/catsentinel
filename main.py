"""Cat Sentinel entrypoint. Wires config.yaml into a running Pipeline.

Usage: python main.py [path/to/config.yaml] [--debug]

--debug opens a live camera preview window (with detection box + verdict
overlay) and prints per-frame status -- what a normal run stays silent about,
like "no motion" or "motion seen but no cat detected". Useful for testing;
press 'q' in the window (or Ctrl+C) to stop.
"""
from __future__ import annotations

import sys

from catsentinel.camera import build_camera
from catsentinel.config import load_config
from catsentinel.deterrent import build_deterrent
from catsentinel.detector import CatDetector
from catsentinel.events import EventLog
from catsentinel.motion import MotionGate
from catsentinel.pipeline import Pipeline
from catsentinel.recognizer import CatRecognizer, Embedder


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    debug = "--debug" in sys.argv[1:]

    config_path = args[0] if args else "config.yaml"
    print(f"[catsentinel] Loading config from {config_path}...")
    config = load_config(config_path)

    print(f"[catsentinel] Opening camera (backend={config.camera.backend})...")
    camera = build_camera(config.camera)

    print("[catsentinel] Loading cat detector model...")
    detector = CatDetector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence_threshold,
        input_size=config.detector.input_size,
        cat_class_id=config.detector.cat_class_id,
    )

    print("[catsentinel] Loading embedding + classifier models...")
    embedder = Embedder(
        model_path=config.recognizer.model_path,
        input_size=config.recognizer.input_size,
    )
    recognizer = CatRecognizer(
        embedder=embedder,
        classifier_path=config.recognizer.classifier_path,
    )

    deterrent = build_deterrent(config.deterrent)

    event_log = EventLog(
        db_path=config.storage.events_db,
        events_dir=config.storage.events_dir,
        save_snapshots=config.storage.save_snapshots,
    )

    motion_gate = None
    if config.motion.enabled:
        motion_gate = MotionGate(
            min_area_fraction=config.motion.min_area_fraction,
            history_frames=config.motion.history_frames,
        )

    print("[catsentinel] Setup complete.")
    pipeline = Pipeline(
        camera=camera,
        detector=detector,
        recognizer=recognizer,
        deterrent=deterrent,
        event_log=event_log,
        motion_gate=motion_gate,
        confirm_frames=config.pipeline.confirm_frames,
        cooldown_seconds=config.pipeline.cooldown_seconds,
        loop_delay_seconds=config.pipeline.loop_delay_seconds,
        debug=debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
