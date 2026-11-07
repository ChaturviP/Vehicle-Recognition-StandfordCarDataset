"""Run a YOLO detector over a split and save predictions for the shared harness.

    # fine-tuned detector (test split)
    python -m src.detect.evaluate_yolo --weights models/detector_yolo11s_cars.pt
    # zero-shot COCO baseline (car + truck classes)
    python -m src.detect.evaluate_yolo --weights yolo11s.pt --coco --name yolo11s_coco_zeroshot

Writes results/preds/<name>.npz (best box per image + all detections for mAP,
per-image latency). Then: python -m src.eval.evaluate results/preds/<name>.npz
Images where nothing is detected fall back to the full image with score 0, so
downstream crops never fail.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.dataset import load_index
from src.eval.evaluate import save_predictions

COCO_VEHICLE_CLASSES = [2, 7]  # car, truck


def run(weights: str, split: str, coco: bool, conf: float, imgsz: int, half: bool):
    import torch
    from ultralytics import YOLO

    df = load_index(split)
    model = YOLO(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    kwargs = dict(conf=conf, imgsz=imgsz, max_det=10, device=device, verbose=False,
                  half=half and device == 0)
    if coco:
        kwargs["classes"] = COCO_VEHICLE_CLASSES

    model.predict(df["path"].iloc[0], **kwargs)  # warm-up (CUDA init, cudnn autotune)

    n = len(df)
    best_boxes = np.zeros((n, 4))
    best_scores = np.zeros(n)
    latency = np.zeros(n)
    det_ids, det_boxes, det_scores = [], [], []

    for i, path in enumerate(tqdm(df["path"], desc=f"{Path(weights).stem} on {split}")):
        r = model.predict(path, **kwargs)[0]
        latency[i] = sum(r.speed.values())  # preprocess + inference + postprocess, ms
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        if len(scores):
            j = int(scores.argmax())
            best_boxes[i], best_scores[i] = boxes[j], scores[j]
            det_ids.append(np.full(len(scores), i))
            det_boxes.append(boxes)
            det_scores.append(scores)
        else:
            h, w = r.orig_shape
            best_boxes[i] = (0, 0, w, h)

    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
    return df, best_boxes, best_scores, latency, cat(det_ids, (0,)).astype(int), cat(det_boxes, (0, 4)), cat(det_scores, (0,))


def _device_label(half: bool) -> str:
    import torch
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} {'fp16' if half else 'fp32'}"
    return "CPU fp32"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="results name (default: weights file stem)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--coco", action="store_true", help="pretrained COCO model: keep car/truck classes only")
    ap.add_argument("--conf", type=float, default=0.001, help="low threshold so mAP sees the full PR curve")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-half", action="store_true")
    args = ap.parse_args()

    name = args.name or Path(args.weights).stem
    if args.split != "test":
        name = f"{name}_{args.split}"
    df, boxes, scores, latency, ids, all_boxes, all_scores = run(
        args.weights, args.split, args.coco, args.conf, args.imgsz, not args.no_half)

    size_mb = Path(args.weights).stat().st_size / 2**20 if Path(args.weights).exists() else None
    out = save_predictions(
        name, df["filename"], split=args.split, family="detector",
        boxes=boxes, box_scores=scores,
        det_image_ids=ids, det_boxes=all_boxes, det_scores=all_scores,
        latency_ms=latency, model_size_mb=size_mb,
        device=_device_label(not args.no_half),
        zero_shot=args.coco, weights=str(args.weights),
    )
    print(f"no-detection images: {(scores == 0).sum()} / {len(scores)}")
    print(f"Saved {out}\nNext: python -m src.eval.evaluate {out}")


if __name__ == "__main__":
    main()
