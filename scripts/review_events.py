"""Interactive review tool: walk through logged sightings, confirm or correct the
pipeline's own verdict against the saved snapshot, and feed confirmed/corrected
ones back into the training photo folders.

This is deliberately human-in-the-loop: the pipeline's own guesses are never
trusted as ground truth for retraining, only what you confirm here. Otherwise an
early mistake would just reinforce itself over time instead of getting corrected.

Usage: python scripts/review_events.py [--limit N]

For each snapshot with a green title bar and window, press:
  f  -- this is your cat (Freddy)         -> data/reference_photos/mycat/
  s  -- this is a stranger cat            -> data/reference_photos/not_mycat/
  x  -- skip (bad photo / can't tell)     -> marked reviewed, not used for training
  q  -- quit (remaining sightings stay unreviewed for next time)

Afterwards, re-run build_reference_embeddings.py -> build_negative_embeddings.py
-> train_classifier.py to fold the new labels into the classifier.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2

from _common import ROOT
from catsentinel.config import load_config
from catsentinel.events import EventLog

WINDOW_NAME = "Cat Sentinel -- review (f=Freddy, s=stranger, x=skip, q=quit)"


def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    config = load_config(ROOT / "config.yaml")
    mycat_dir = ROOT / "data/reference_photos/mycat"
    not_mycat_dir = ROOT / "data/reference_photos/not_mycat"
    mycat_dir.mkdir(parents=True, exist_ok=True)
    not_mycat_dir.mkdir(parents=True, exist_ok=True)

    event_log = EventLog(
        db_path=str(ROOT / config.storage.events_db),
        events_dir=str(ROOT / config.storage.events_dir),
        save_snapshots=config.storage.save_snapshots,
    )

    sightings = event_log.unreviewed(limit=limit)
    if not sightings:
        print("Nothing to review -- no unreviewed sightings with a saved snapshot.")
        event_log.close()
        return

    print(f"{len(sightings)} sightings to review. f=Freddy, s=stranger, x=skip, q=quit.\n")

    counts = {"mine": 0, "stranger": 0, "skip": 0}
    quit_early = False

    for sighting in sightings:
        snapshot_path = Path(sighting.snapshot_path)
        if not snapshot_path.exists():
            print(f"[skip] snapshot missing on disk: {snapshot_path}")
            event_log.set_reviewed_label(sighting.id, "skip")
            counts["skip"] += 1
            continue

        frame = cv2.imread(str(snapshot_path))
        if frame is None:
            print(f"[skip] could not read: {snapshot_path}")
            event_log.set_reviewed_label(sighting.id, "skip")
            counts["skip"] += 1
            continue

        display = frame.copy()
        label = f"pipeline said: {sighting.verdict} (probability={sighting.probability:.2f})"
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, sighting.timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow(WINDOW_NAME, display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("f"), ord("s"), ord("x"), ord("q")):
                break

        if key == ord("q"):
            quit_early = True
            break

        if key == ord("f"):
            dest = mycat_dir / f"reviewed_{sighting.id}{snapshot_path.suffix}"
            shutil.copy2(snapshot_path, dest)
            event_log.set_reviewed_label(sighting.id, "mine")
            counts["mine"] += 1
            print(f"[mine]     sighting {sighting.id} -> {dest.name}")
        elif key == ord("s"):
            dest = not_mycat_dir / f"reviewed_{sighting.id}{snapshot_path.suffix}"
            shutil.copy2(snapshot_path, dest)
            event_log.set_reviewed_label(sighting.id, "stranger")
            counts["stranger"] += 1
            print(f"[stranger] sighting {sighting.id} -> {dest.name}")
        elif key == ord("x"):
            event_log.set_reviewed_label(sighting.id, "skip")
            counts["skip"] += 1
            print(f"[skip]     sighting {sighting.id}")

    cv2.destroyAllWindows()
    event_log.close()

    print(f"\nReviewed: {counts['mine']} as Freddy, {counts['stranger']} as stranger, {counts['skip']} skipped.")
    if quit_early:
        print("Stopped early -- remaining sightings are still unreviewed for next time.")
    if counts["mine"] or counts["stranger"]:
        print(
            "\nNew labels added. Retrain with:\n"
            "  python scripts/build_reference_embeddings.py\n"
            "  python scripts/build_negative_embeddings.py\n"
            "  python scripts/train_classifier.py"
        )


if __name__ == "__main__":
    main()
