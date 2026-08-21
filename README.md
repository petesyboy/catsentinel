# Cat Sentinel

Watches the back door / cat flap, detects when a cat walks into view, figures out
whether it's your cat or a stranger, and if it's a stranger, plays a startling sound
to scare it off before it comes inside. Every sighting is logged with a snapshot so
you can review and tune it over time.

- **Detection** ("is there a cat"): pretrained YOLOv8n (COCO), runs out of the box.
- **Recognition** ("is it *our* cat"): each detected cat crop is embedded with a
  pretrained MobileNetV2, then classified by a small logistic regression head
  trained on embeddings of your cat vs. other cats. A raw cosine-similarity
  threshold was tried first but didn't separate the two cleanly; the trained
  classifier does (97% recognition of Freddy, 100% rejection of other cats in
  leave-one-out testing — see "Tuning" below for how that was measured).
- **Deterrent** (v1): sound, played through the default audio output. A water
  sprayer is a stretch goal once you have the relay/valve hardware (see
  `src/catsentinel/deterrent.py`).

You can develop and test this entirely on a laptop with a webcam before any Pi
hardware arrives — `config.yaml`'s `camera.backend` is the only thing you flip
when you move to the real Pi + camera.

## Shopping list

You already have: Raspberry Pi 4 Model B (starter kit).

Still need:
1. **Camera** — either the official Raspberry Pi Camera Module 3 (best quality,
   needs the ribbon-cable CSI port, use `camera.backend: picamera2`), or a plain
   USB webcam (simpler, no ribbon cable, use `camera.backend: opencv`).
2. **Speaker** — a small USB speaker, or a speaker wired via the Pi's 3.5mm jack /
   a cheap I2S DAC board. Anything the Pi can play a WAV file through works.
3. *(Stretch goal, later)* — for the water-sprayer deterrent: a 5V relay module +
   a small 12V solenoid valve and a garden-style micro water pump (or a
   servo-actuated toy squirt gun instead of a valve+pump, which is mechanically
   simpler). Not needed for v1.

## One-time setup

1. Create a virtualenv and install the project (from this folder):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Pi/Linux
   pip install -e .
   ```
2. Generate the ONNX models (needs extra one-off packages — see script docstring
   for why; you only need to run this on *one* machine, then copy `models/*.onnx`
   to the Pi):
   ```bash
   pip install ultralytics torch torchvision onnxscript
   python scripts/download_models.py
   ```
   On Windows, prefix with `set PYTHONIOENCODING=utf-8 &&` if you hit a
   `UnicodeEncodeError` (the exporter prints a checkmark the default console
   codepage can't display).
3. Export a handful of photos of your cat from cloud storage (Google
   Photos/Drive/iCloud) and drop them into `data/reference_photos/mycat/`
   (30-50 varied photos — different angles/lighting works better than a single
   great shot). If you have way more than that, run
   `python scripts/curate_reference_photos.py` first to cull blurry/duplicate
   shots down to a manageable, varied set automatically.
4. Do the same with a handful of photos of *other* cats (old photos of previous
   pets, a neighbor's cat, whatever you have) into
   `data/reference_photos/not_mycat/` — these teach the classifier what to
   reject. Even 15-20 helps a lot; it doesn't need to be as large as your own
   cat's set.
5. Build embeddings for both sets, then train the classifier:
   ```bash
   pip install scikit-learn   # training-only, not needed at runtime
   python scripts/build_reference_embeddings.py
   python scripts/build_negative_embeddings.py
   python scripts/train_classifier.py
   ```
   This prints leave-one-out cross-validated accuracy so you can see how well
   it's actually separating your cat from others before relying on it.
6. Drop a `.wav` file at `sounds/deterrent.wav` (any short, sharp sound — a
   dog bark, hiss, or loud beep works well). `config.yaml`'s
   `deterrent.sound_file` points here.

## Running

```bash
python main.py
```

By default `config.yaml` uses `camera.backend: opencv` with `device: 0`, which
is your laptop's built-in/USB webcam — good for testing the whole pipeline before
the Pi hardware shows up. Point a photo of your cat, then a photo of a different
cat, at the webcam and watch the console output plus `data/events.db` /
`data/events/`.

Add `--debug` (`python main.py --debug`) for a live preview window with a box
around any detected cat and the verdict overlaid, plus per-frame console status
("no motion" / "motion seen, but no cat detected" / a verdict) — the normal run
stays silent unless it reaches a full verdict, which can look like nothing's
happening. Press `q` in the preview window (or Ctrl+C) to stop. If you're on
Windows and see no console output at all even with `--debug`, run with
`python -u main.py --debug` — some terminals (Git Bash/MinTTY in particular)
don't get detected as interactive, so Python buffers output until it builds up.

The deterrent sound plays via the OS's own async playback (`winsound` on
Windows, `aplay` on Linux/the Pi) rather than a bundled audio library, and any
playback failure is caught and logged rather than crashing the pipeline. On the
Pi, `aplay` needs `alsa-utils` (`sudo apt install alsa-utils`), which is
preinstalled on Raspberry Pi OS.

## Moving to the Raspberry Pi

1. Copy this whole folder to the Pi (or `git clone` if you push it to a repo),
   including everything in `models/` (the embedding model exports as a pair --
   `mobilenetv2_embed.onnx` *and* `mobilenetv2_embed.onnx.data` -- both are
   needed) and `data/embeddings/cat_classifier.npz` (the only file the runtime
   actually needs from the training step -- no need to copy the raw reference
   photos or scikit-learn over unless you want to retrain on the Pi itself).
2. `pip install -e .[pi]` (adds `picamera2`).
3. In `config.yaml`, set `camera.backend: picamera2` if using the official
   camera module, or leave it as `opencv` with the right `device` index if using
   a USB webcam.
4. Plug in your speaker, run `python main.py`.
5. Optional: set it up as a systemd service so it starts on boot and restarts if
   it crashes (not included yet — ask if you want this added).

## Tuning

- **Recognition accuracy**: run `python scripts/evaluate_recognizer.py` to see
  how well the trained classifier separates your cat from a folder of other cat
  photos (defaults to `data/reference_photos/not_mycat/`). If it's letting too
  many strangers through, or misfiring on your own cat, add more/better photos
  to `data/reference_photos/mycat/` and `not_mycat/` and re-run
  `build_reference_embeddings.py` → `build_negative_embeddings.py` →
  `train_classifier.py`. `train_classifier.py` picks its own decision threshold
  via leave-one-out cross-validation, so you don't need to hand-tune one.
- `pipeline.confirm_frames` / `pipeline.cooldown_seconds` (in `config.yaml`) —
  how many consecutive "stranger" frames before the deterrent fires, and how
  long to wait before it can fire again.
- `motion.min_area_fraction` (in `config.yaml`) — raise if the detector is
  running too often (e.g. from lighting flicker); lower if real cat entrances
  are being missed.
- Once the camera's live, `data/events.db`'s `probability` column shows real
  scores from actual footage — the best source of truth for whether the
  classifier needs more training photos.

## Improving recognition from real footage

Every sighting is logged with a snapshot, so the classifier can get better over
time from actual door visits, not just the original photo dump:

```bash
python scripts/review_events.py
```

Shows you each unreviewed snapshot next to the pipeline's own verdict. Press
`f` if it's really Freddy, `s` if it's really a stranger, `x` to skip a bad/
ambiguous photo, `q` to stop (anything left stays unreviewed for next time).
Confirmed/corrected photos get filed into `data/reference_photos/mycat/` or
`not_mycat/` automatically. This is deliberately human-confirmed, not
automatic — the pipeline's own guesses are never trusted as ground truth for
retraining, so an early mistake can't reinforce itself.

Afterwards, retrain to fold the new labels in:
```bash
python scripts/build_reference_embeddings.py
python scripts/build_negative_embeddings.py
python scripts/train_classifier.py
```

## Project layout

```
config.yaml                        # all tunable settings — the only file you
                                    # change moving from laptop to Pi
main.py                            # entrypoint
src/catsentinel/
  camera.py                        # webcam / Pi camera capture
  motion.py                        # cheap frame-diff gate before running the detector
  detector.py                      # YOLOv8n ONNX -> cat bounding boxes
  recognizer.py                    # embedding + trained classifier -> your cat vs stranger
  deterrent.py                     # sound trigger (+ stubbed water sprayer)
  events.py                        # SQLite sighting log + snapshots
  pipeline.py                      # wires it all into the main loop
scripts/
  download_models.py               # one-time ONNX model export
  curate_reference_photos.py       # cull a big raw photo dump down to a good reference set
  build_reference_embeddings.py    # mycat photos -> embeddings/mycat.npy
  build_negative_embeddings.py     # not-mycat photos -> embeddings/not_mycat.npy
  train_classifier.py              # trains + exports embeddings/cat_classifier.npz
  evaluate_recognizer.py           # checks classifier accuracy against a test photo folder
  review_events.py                 # human-in-the-loop review of logged sightings
data/
  reference_photos/mycat/          # your cat's photos go here
  reference_photos/not_mycat/      # other cats' photos go here
  embeddings/                      # generated: mycat.npy, not_mycat.npy, cat_classifier.npz
  events.db, events/               # sighting log + snapshots (generated)
```
