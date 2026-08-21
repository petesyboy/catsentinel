"""Shared helpers for the setup scripts (not part of the installed package)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from catsentinel.config import load_config  # noqa: E402
from catsentinel.detector import CatDetector  # noqa: E402
from catsentinel.recognizer import Embedder  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_detector_and_embedder(config=None) -> tuple[CatDetector, Embedder]:
    config = config or load_config(ROOT / "config.yaml")
    detector = CatDetector(
        model_path=str(ROOT / config.detector.model_path),
        confidence_threshold=config.detector.confidence_threshold,
        input_size=config.detector.input_size,
        cat_class_id=config.detector.cat_class_id,
    )
    embedder = Embedder(
        model_path=str(ROOT / config.recognizer.model_path),
        input_size=config.recognizer.input_size,
    )
    return detector, embedder


def embed_photo_folder(folder: Path, detector: CatDetector, embedder: Embedder) -> list[tuple[str, np.ndarray]]:
    """Embeds every photo in `folder`, cropping to the detected cat when possible
    (falls back to the whole image), same as what happens to a live detection.
    """
    import cv2

    results = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[skip] could not read {path.name}")
            continue
        detections = detector.detect(frame)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.box
            crop = frame[y1:y2, x1:x2]
            source = "detected cat crop"
        else:
            crop = frame
            source = "full image (no cat detected)"
        results.append((path.name, embedder.embed(crop)))
        print(f"[ok] {path.name} -> embedded ({source})")
    return results
