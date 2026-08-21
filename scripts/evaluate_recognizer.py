"""Checks how well the trained classifier (scripts/train_classifier.py) separates
your cat from other cats, using a folder of "not your cat" test photos.

Usage: python scripts/evaluate_recognizer.py [path/to/not_your_cat_photos]
(defaults to data/reference_photos/not_mycat)
"""
from __future__ import annotations

import sys
from pathlib import Path

from _common import ROOT, build_detector_and_embedder
from catsentinel.config import load_config
from catsentinel.recognizer import CatRecognizer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    other_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/reference_photos/not_mycat"
    if not other_dir.exists():
        raise SystemExit(f"No such folder: {other_dir}")

    config = load_config(ROOT / "config.yaml")
    detector, embedder = build_detector_and_embedder(config)
    recognizer = CatRecognizer(embedder=embedder, classifier_path=str(ROOT / config.recognizer.classifier_path))

    import cv2

    print("--- Classifier probability that each photo is 'your cat' (should be low) ---")
    correct = 0
    total = 0
    for path in sorted(p for p in other_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[skip] could not read {path.name}")
            continue
        detections = detector.detect(frame)
        crop = frame
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.box
            crop = frame[y1:y2, x1:x2]

        is_mine, prob = recognizer.identify(crop)
        total += 1
        correct += not is_mine
        flag = "  <-- WRONGLY classified as your cat!" if is_mine else ""
        print(f"  {prob:.3f}  {path.name}{flag}")

    print(f"\nCorrectly rejected {correct}/{total} ({100 * correct / total:.0f}%) as not your cat.")


if __name__ == "__main__":
    main()
