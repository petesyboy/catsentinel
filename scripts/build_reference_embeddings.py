"""Processes data/reference_photos/mycat/*.jpg -> data/embeddings/mycat.npy.

For each reference photo: if the cat detector finds a cat in it, embed just that
crop (matches what happens at runtime); otherwise embed the whole photo. Run this
once after adding/changing reference photos, then again any time you add more.
"""
from __future__ import annotations

import numpy as np

from _common import IMAGE_EXTENSIONS, ROOT, build_detector_and_embedder, embed_photo_folder
from catsentinel.config import load_config


def main() -> None:
    config = load_config(ROOT / "config.yaml")

    photos_dir = ROOT / "data" / "reference_photos" / "mycat"
    if not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in photos_dir.iterdir()):
        raise SystemExit(
            f"No photos found in {photos_dir}. Export some photos of your cat there first."
        )

    detector, embedder = build_detector_and_embedder(config)
    embeddings = embed_photo_folder(photos_dir, detector, embedder)
    if not embeddings:
        raise SystemExit(f"No usable photos found in {photos_dir}")

    output_path = ROOT / config.recognizer.embeddings_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.stack([emb for _name, emb in embeddings]))
    print(f"Wrote {len(embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
