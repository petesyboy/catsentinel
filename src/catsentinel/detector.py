"""YOLOv8n ONNX object detector, filtered down to "cat" detections.

Uses the stock COCO-pretrained yolov8n.onnx (see scripts/download_models.py) --
no training required, "cat" (COCO class 15) is already one of its 80 classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame coordinates
    confidence: float


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return canvas, scale, pad_left, pad_top


class CatDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        input_size: int = 640,
        cat_class_id: int = 15,
        nms_iou_threshold: float = 0.45,
    ):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._confidence_threshold = confidence_threshold
        self._input_size = input_size
        self._cat_class_id = cat_class_id
        self._nms_iou_threshold = nms_iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        size = self._input_size
        canvas, scale, pad_left, pad_top = _letterbox(frame, size)

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
        blob = blob.transpose(2, 0, 1)[None]  # HWC -> NCHW

        outputs = self._session.run(None, {self._input_name: blob})[0]  # (1, 84, N)
        preds = outputs[0].T  # (N, 84): 4 box coords + 80 class scores

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = (class_ids == self._cat_class_id) & (confidences >= self._confidence_threshold)
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]

        # cx, cy, w, h (letterboxed input space) -> x1, y1, w, h (for cv2.dnn.NMSBoxes)
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        nms_boxes = np.stack([x1, y1, boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)

        keep = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), confidences.tolist(),
            self._confidence_threshold, self._nms_iou_threshold,
        )
        keep = np.array(keep).flatten() if len(keep) else np.array([], dtype=int)

        detections: list[Detection] = []
        for i in keep:
            bx, by, bw, bh = nms_boxes[i]
            # undo letterbox padding/scale to get original-frame coordinates
            ox1 = (bx - pad_left) / scale
            oy1 = (by - pad_top) / scale
            ox2 = (bx + bw - pad_left) / scale
            oy2 = (by + bh - pad_top) / scale
            h, w = frame.shape[:2]
            ox1, oy1 = max(0, int(ox1)), max(0, int(oy1))
            ox2, oy2 = min(w, int(ox2)), min(h, int(oy2))
            detections.append(Detection(box=(ox1, oy1, ox2, oy2), confidence=float(confidences[i])))

        return detections
