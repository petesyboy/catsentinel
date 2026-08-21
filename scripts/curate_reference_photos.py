"""Culls a raw dump of cat photos down to a smaller, non-redundant reference set.

Reads data/reference_photos/mycat_raw/*, drops blurry shots (variance-of-Laplacian)
and near-duplicate burst shots (average-hash + Hamming distance), then copies the
survivors into data/reference_photos/mycat/ (capped at --max-photos, spread evenly
if there are more survivors than that).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def average_hash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    small = cv2.resize(gray, (hash_size, hash_size))
    return (small > small.mean()).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data/reference_photos/mycat_raw"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/reference_photos/mycat"))
    parser.add_argument("--blur-threshold", type=float, default=50.0,
                         help="Below this variance-of-Laplacian score, a photo is considered too blurry.")
    parser.add_argument("--dup-hamming-threshold", type=int, default=6,
                         help="Average-hash Hamming distance below which two photos count as near-duplicates.")
    parser.add_argument("--max-photos", type=int, default=50)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No photos found in {raw_dir}")

    candidates = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] unreadable: {path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = blur_score(gray)
        if score < args.blur_threshold:
            print(f"[skip] too blurry ({score:.1f}): {path.name}")
            continue
        candidates.append((path, score, average_hash(gray)))

    # Dedupe near-identical/burst shots, keeping the sharpest of each cluster.
    candidates.sort(key=lambda c: -c[1])  # sharpest first
    kept: list[tuple[Path, float, np.ndarray]] = []
    for path, score, ahash in candidates:
        if any(hamming_distance(ahash, k[2]) <= args.dup_hamming_threshold for k in kept):
            print(f"[skip] near-duplicate of an already-kept photo: {path.name}")
            continue
        kept.append((path, score, ahash))

    if len(kept) > args.max_photos:
        # Spread the sample evenly across the kept set rather than just taking the sharpest N,
        # to preserve variety (angle/lighting/pose) instead of biasing toward one look.
        indices = np.linspace(0, len(kept) - 1, args.max_photos).round().astype(int)
        kept = [kept[i] for i in sorted(set(indices))]

    for f in out_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            f.unlink()

    for path, _score, _hash in kept:
        shutil.copy2(path, out_dir / path.name)

    print(f"\nStarted with {len(paths)} raw photos.")
    print(f"Kept {len(kept)} after blur/duplicate filtering -> {out_dir}")


if __name__ == "__main__":
    main()
