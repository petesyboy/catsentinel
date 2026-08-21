"""Cat re-identification: embed each detected cat crop with a pretrained MobileNetV2,
then classify with a small logistic regression head trained on embeddings of your
cat vs. other cats (see scripts/train_classifier.py).

A plain cosine-similarity-to-reference-set threshold was tried first, but testing
showed it doesn't cleanly separate your cat from others -- their score ranges
overlapped too much for any single threshold to work well. The trained classifier
weighs the embedding dimensions instead of comparing them uniformly, and testing
showed it separates the two far better.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Embedder:
    """Wraps the MobileNetV2 ONNX model; turns a BGR image crop into a unit-norm embedding."""

    def __init__(self, model_path: str, input_size: int = 224):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size

    def embed(self, bgr_image: np.ndarray) -> np.ndarray:
        size = self._input_size
        resized = cv2.resize(bgr_image, (size, size))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = normalized.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW

        output = self._session.run(None, {self._input_name: blob})[0]
        vec = output.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class CatRecognizer:
    """Loads the classifier trained by scripts/train_classifier.py -- just a
    weight vector, bias, and decision threshold, so this needs only numpy at
    runtime (no scikit-learn on the Pi).
    """

    def __init__(self, embedder: Embedder, classifier_path: str):
        self._embedder = embedder
        self._coef, self._intercept, self._threshold = self._load_classifier(classifier_path)

    @staticmethod
    def _load_classifier(path: str) -> tuple[np.ndarray, float, float]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No trained classifier found at {p}. "
                "Run build_reference_embeddings.py, build_negative_embeddings.py, "
                "then train_classifier.py first."
            )
        data = np.load(p)
        return data["coef"], float(data["intercept"]), float(data["threshold"])

    def identify(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        """Returns (is_our_cat, probability_it_is_our_cat)."""
        embedding = self._embedder.embed(crop_bgr)
        probability = float(_sigmoid(np.dot(self._coef, embedding) + self._intercept))
        return probability >= self._threshold, probability
