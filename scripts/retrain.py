"""Runs the full retrain pipeline in one command:
build_reference_embeddings -> build_negative_embeddings -> train_classifier.

Run this any time after reviewing new sightings with review_events.py (or after
adding/removing photos in data/reference_photos/mycat/ or not_mycat/ by hand).

Usage: python scripts/retrain.py
"""
from __future__ import annotations

import build_negative_embeddings
import build_reference_embeddings
import train_classifier


def main() -> None:
    print("=== 1/3: building reference embeddings (mycat) ===")
    build_reference_embeddings.main()

    print("\n=== 2/3: building negative embeddings (not_mycat) ===")
    build_negative_embeddings.main()

    print("\n=== 3/3: training classifier ===")
    train_classifier.main()

    print("\nDone -- classifier retrained.")


if __name__ == "__main__":
    main()
