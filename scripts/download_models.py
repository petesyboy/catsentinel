"""One-time setup: produces models/yolov8n.onnx and models/mobilenetv2_embed.onnx.

Run this once on any normal dev machine (it does NOT need to be the Pi -- these
are one-off exports; only the resulting small .onnx files need to be copied over
to the Pi afterwards). Uses the official ultralytics/torchvision downloaders
rather than a hardcoded URL, so it won't break if some mirror moves.

Needs extra packages not required at runtime:
    pip install ultralytics torch torchvision onnxscript

On Windows, also run with UTF-8 output forced (torch's exporter prints unicode
checkmarks that crash the default Windows console codepage otherwise):
    set PYTHONIOENCODING=utf-8 && python scripts/download_models.py
"""
from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def export_yolov8n() -> None:
    target = MODELS_DIR / "yolov8n.onnx"
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Missing dependency. Run: pip install ultralytics") from e

    print("Downloading yolov8n.pt and exporting to ONNX (COCO-pretrained, includes 'cat')...")
    model = YOLO("yolov8n.pt")
    exported_path = Path(model.export(format="onnx", imgsz=640, opset=12))
    exported_path.replace(target)
    print(f"Wrote {target}")


def export_mobilenetv2_embedding() -> None:
    target = MODELS_DIR / "mobilenetv2_embed.onnx"
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    try:
        import torch
        import torchvision
    except ImportError as e:
        raise SystemExit("Missing dependency. Run: pip install torch torchvision") from e

    print("Downloading pretrained MobileNetV2 and exporting embedding head to ONNX...")
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = torchvision.models.mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()  # drop the 1000-class head -> 1280-d embedding
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, str(target),
        input_names=["input"], output_names=["embedding"],
        opset_version=12,
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_yolov8n()
    export_mobilenetv2_embedding()
    print("Done. Next: put reference photos in data/reference_photos/mycat/ and run "
          "scripts/build_reference_embeddings.py")
