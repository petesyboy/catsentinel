"""Trains a small logistic regression classifier on top of the embeddings to
separate your cat from other cats -- replaces the raw cosine-similarity
threshold, which testing showed doesn't cleanly separate the two.

Needs data/embeddings/mycat.npy and data/embeddings/not_mycat.npy (see
build_reference_embeddings.py / build_negative_embeddings.py). Uses
leave-one-out cross-validation to pick both the regularization strength and the
decision threshold honestly, given how little data there is.

Exports just the learned weight vector + bias + threshold to a plain .npz, so
the Pi only ever needs numpy at runtime, not scikit-learn (that's a setup-only
dependency, like torch/ultralytics for the ONNX exports).

Needs extra package not required at runtime: `pip install scikit-learn`
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from _common import ROOT
from catsentinel.config import load_config


def loo_probabilities(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Out-of-sample predicted probability for each sample, holding it out of training."""
    probs = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return probs


def best_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold that maximizes balanced accuracy (mean of per-class recall)."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = probs >= t
        mine_recall = preds[y == 1].mean() if (y == 1).any() else 0.0
        other_recall = (~preds[y == 0]).mean() if (y == 0).any() else 0.0
        score = (mine_recall + other_recall) / 2
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def main() -> None:
    config = load_config(ROOT / "config.yaml")

    mine = np.load(ROOT / config.recognizer.embeddings_path)
    others = np.load(ROOT / config.recognizer.negative_embeddings_path)

    X = np.concatenate([mine, others], axis=0)
    y = np.concatenate([np.ones(len(mine)), np.zeros(len(others))])

    candidate_Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    best_C, best_probs, best_bal_acc, best_t = None, None, -1.0, 0.5
    print("Selecting regularization strength via leave-one-out cross-validation:")
    for C in candidate_Cs:
        probs = loo_probabilities(X, y, C)
        t, bal_acc = best_threshold(probs, y)
        print(f"  C={C:<6} best LOO balanced accuracy={bal_acc:.3f} at threshold={t:.2f}")
        if bal_acc > best_bal_acc:
            best_C, best_probs, best_bal_acc, best_t = C, probs, bal_acc, t

    preds = best_probs >= best_t
    mine_recall = preds[y == 1].mean()
    other_recall = (~preds[y == 0]).mean()
    print(f"\nChosen: C={best_C}, threshold={best_t:.2f}")
    print(f"  Leave-one-out: your cat correctly recognized {mine_recall * 100:.0f}% of the time")
    print(f"  Leave-one-out: other cats correctly rejected {other_recall * 100:.0f}% of the time")

    # Fit the final model on all available data.
    final_model = LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000)
    final_model.fit(X, y)

    output_path = ROOT / config.recognizer.classifier_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        coef=final_model.coef_[0].astype(np.float32),
        intercept=np.float32(final_model.intercept_[0]),
        threshold=np.float32(best_t),
    )
    print(f"\nWrote classifier to {output_path}")


if __name__ == "__main__":
    main()
