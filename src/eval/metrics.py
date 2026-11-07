"""Metric functions shared by every pipeline (numpy only)."""
from __future__ import annotations

import numpy as np


# ---------- classification ----------

def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def make_accuracy(probs: np.ndarray, y: np.ndarray, class_to_make: np.ndarray) -> float:
    """Top-1 accuracy at manufacturer level (prediction mapped to its make)."""
    return float((class_to_make[probs.argmax(1)] == class_to_make[y]).mean())


def per_class_accuracy(probs: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    pred = probs.argmax(1)
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = y == c
        if m.any():
            acc[c] = (pred[m] == c).mean()
    return acc


def confused_pairs(probs: np.ndarray, y: np.ndarray, top: int = 10) -> list[tuple[int, int, int]]:
    """Most frequent (true, predicted, count) mistakes."""
    pred = probs.argmax(1)
    wrong = pred != y
    pairs, counts = np.unique(np.stack([y[wrong], pred[wrong]], 1), axis=0, return_counts=True)
    order = np.argsort(-counts)[:top]
    return [(int(pairs[i, 0]), int(pairs[i, 1]), int(counts[i])) for i in order]


# ---------- confidence / calibration ----------

def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, avg_conf, frac = correct[m].mean(), conf[m].mean(), m.mean()
        ece += frac * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                     "accuracy": float(acc), "confidence": float(avg_conf)})
    return float(ece), bins


def coverage_curve(probs: np.ndarray, y: np.ndarray,
                   thresholds: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
    """For each threshold tau: share of images accepted (conf >= tau) and accuracy on them."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"threshold": t, "coverage": float(m.mean()),
                     "accuracy": float(correct[m].mean()) if m.any() else float("nan"),
                     "rejected_error_rate": float((~correct[~m]).mean()) if (~m).any() else float("nan")})
    return rows


# ---------- localization / detection ----------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU between two (N, 4) arrays of x1, y1, x2, y2 boxes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1, iy1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix2, iy2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def average_precision(gt_boxes: np.ndarray, det_image_ids: np.ndarray, det_boxes: np.ndarray,
                      det_scores: np.ndarray, iou_thr: float = 0.5) -> float:
    """Single-class AP (101-point COCO interpolation) with one GT box per image.

    gt_boxes: (N_images, 4); det_*: all detections across images (any number per image).
    """
    order = np.argsort(-det_scores)
    ids, boxes = det_image_ids[order], det_boxes[order]
    matched = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(order))
    for i, (img, box) in enumerate(zip(ids, boxes)):
        if not matched[img] and box_iou(box[None], gt_boxes[img][None])[0] >= iou_thr:
            matched[img] = True
            tp[i] = 1
    if len(order) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / len(gt_boxes)
    precision = ctp / np.arange(1, len(order) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([precision[j] if j < len(precision) else 0.0 for j in idx]))


def map_50_95(gt_boxes, det_image_ids, det_boxes, det_scores) -> tuple[float, float]:
    thrs = np.arange(0.5, 0.96, 0.05)
    aps = [average_precision(gt_boxes, det_image_ids, det_boxes, det_scores, t) for t in thrs]
    return aps[0], float(np.mean(aps))
