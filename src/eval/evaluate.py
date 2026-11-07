"""Single evaluation harness. Every pipeline (classical, deep, hybrid, detector,
robustness variants, ablations) saves a predictions file with
`save_predictions(...)`, then this script turns it into metrics JSON.

    python -m src.eval.evaluate results/preds/convnext_base_448.npz [more.npz ...]

Predictions file contract (.npz):
    filenames      (N,)      image filenames from index.csv, same split
    probs          (N, 196)  class probabilities            [classification runs]
    boxes          (N, 4)    best box per image, x1y1x2y2    [detection runs]
    box_scores     (N,)      score of that box
    det_image_ids, det_boxes, det_scores   all detections   [optional, for mAP]
    latency_ms     (N,)      per-image inference time        [optional]
    meta           JSON str  {name, family, split, condition, model_size_mb, ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src import config
from src.data.dataset import load_index
from src.eval import metrics as M

PREDS_DIR = config.RESULTS_DIR / "preds"
METRICS_DIR = config.RESULTS_DIR / "metrics"


def save_predictions(name: str, filenames, *, split: str = "test", family: str = "deep",
                     condition: str = "clean", probs: Optional[np.ndarray] = None,
                     boxes: Optional[np.ndarray] = None, box_scores: Optional[np.ndarray] = None,
                     det_image_ids=None, det_boxes=None, det_scores=None,
                     latency_ms: Optional[np.ndarray] = None, model_size_mb: Optional[float] = None,
                     extra_arrays: Optional[dict] = None, **extra_meta: Any) -> Path:
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "family": family, "split": split, "condition": condition,
            "model_size_mb": model_size_mb, **extra_meta}
    arrays = {k: v for k, v in dict(probs=probs, boxes=boxes, box_scores=box_scores,
                                    det_image_ids=det_image_ids, det_boxes=det_boxes,
                                    det_scores=det_scores, latency_ms=latency_ms).items() if v is not None}
    arrays.update({k: v for k, v in (extra_arrays or {}).items() if v is not None})
    path = PREDS_DIR / f"{name}.npz"
    np.savez_compressed(path, filenames=np.asarray(list(filenames), dtype=str), meta=json.dumps(meta), **arrays)
    return path


def evaluate_file(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    df = load_index(meta["split"]).set_index("filename")
    df = df.loc[data["filenames"]]  # align ground truth to prediction order
    y = df["class_id"].to_numpy()
    out: dict[str, Any] = {"meta": meta, "n_images": int(len(df))}

    if "probs" in data:
        probs = data["probs"]
        class_to_make = (load_index().drop_duplicates("class_id").sort_values("class_id")["make_id"].to_numpy())
        ece, bins = M.expected_calibration_error(probs, y)
        conf = probs.max(1)
        correct = probs.argmax(1) == y
        out["classification"] = {
            "top1": M.topk_accuracy(probs, y, 1),
            "top5": M.topk_accuracy(probs, y, 5),
            "make_top1": M.make_accuracy(probs, y, class_to_make),
            "mean_confidence": float(conf.mean()),
            "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
            "ece": ece,
            "reliability_bins": bins,
            "coverage_curve": M.coverage_curve(probs, y),
            "top_confusions": M.confused_pairs(probs, y),
        }

    if "boxes" in data:
        gt = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ious = M.box_iou(data["boxes"], gt)
        if "det_boxes" in data:
            ids, boxes, scores = data["det_image_ids"], data["det_boxes"], data["det_scores"]
        else:
            ids, boxes, scores = np.arange(len(gt)), data["boxes"], data["box_scores"]
        map50, map5095 = M.map_50_95(gt, ids, boxes, scores)
        out["detection"] = {
            "mean_iou": float(ious.mean()), "median_iou": float(np.median(ious)),
            "iou_ge_0.5": float((ious >= 0.5).mean()), "iou_ge_0.75": float((ious >= 0.75).mean()),
            "map50": map50, "map50_95": map5095,
        }

    if "latency_ms" in data:
        lat = data["latency_ms"]
        out["efficiency"] = {"latency_ms_mean": float(lat.mean()),
                             "latency_ms_p95": float(np.percentile(lat, 95)),
                             "model_size_mb": meta.get("model_size_mb")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", type=Path)
    args = ap.parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for p in args.preds:
        res = evaluate_file(p)
        out = METRICS_DIR / f"{res['meta']['name']}.json"
        out.write_text(json.dumps(res, indent=2))
        c, d = res.get("classification", {}), res.get("detection", {})
        summary = ", ".join(f"{k}={v:.4f}" for k, v in
                            [("top1", c.get("top1")), ("top5", c.get("top5")), ("ece", c.get("ece")),
                             ("mIoU", d.get("mean_iou")), ("mAP50", d.get("map50"))] if v is not None)
        print(f"{res['meta']['name']}: {summary} -> {out}")


if __name__ == "__main__":
    main()
