"""Processes data/reference_photos/not_mycat/*.jpg -> data/embeddings/not_mycat.npy.

These are photos of cats that are NOT yours -- used together with mycat.npy to
train the classifier (see train_classifier.py) that actually separates your cat
from others, rather than relying on a single similarity threshold.
"""
from __future__ import annotations

import numpy as np

from _common import IMAGE_EXTENSIONS, ROOT, build_detector_and_embedder, embed_photo_folder
from catsentinel.config import load_config


def main() -> None:
    config = load_config(ROOT / "config.yaml")

    photos_dir = ROOT / "data" / "reference_photos" / "not_mycat"
    if not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in photos_dir.iterdir()):
        raise SystemExit(f"No photos found in {photos_dir}.")

    detector, embedder = build_detector_and_embedder(config)
    embeddings = embed_photo_folder(photos_dir, detector, embedder)
    if not embeddings:
        raise SystemExit(f"No usable photos found in {photos_dir}")

    output_path = ROOT / config.recognizer.negative_embeddings_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.stack([emb for _name, emb in embeddings]))
    print(f"Wrote {len(embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
