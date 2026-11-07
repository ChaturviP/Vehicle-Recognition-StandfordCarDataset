"""Extract classical features for train / val / test and cache them.

    python -m src.classical.extract_features                     # full pipeline, all splits
    python -m src.classical.extract_features --variant nogc --no-grabcut   # ablation variants
    python -m src.classical.extract_features --variant noclahe --no-clahe

Train and val use ground-truth boxes; test uses the detector's boxes (end-to-end)
unless --test-crop gt. Output: data/cache/classical_<variant>_<split>.npz with one
array per feature group, filenames and per-image extraction time (ms).
Runs on all CPU cores; ~60 ms per image per core with GrabCut on, i.e. a few minutes
for all 16k images on a multi-core CPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src import config
from src.classical.features import GROUPS, extract
from src.classical.preprocess import run_pipeline
from src.data.dataset import load_det_boxes, load_index


def _one(path: str, box, opts: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:  # cv2 can fail on some unicode paths on Windows
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    st = run_pipeline(img, box, **opts)
    feats = extract(st)
    return feats, (time.perf_counter() - t0) * 1000


def cache_path(variant: str, split: str) -> Path:
    return config.CACHE_DIR / f"classical_{variant}_{split}.npz"


def extract_split(split: str, variant: str, opts: dict, test_crop: str, det_preds: Path, n_jobs: int) -> Path:
    df = load_index(split)
    if split == "test" and test_crop == "det":
        boxes = load_det_boxes(det_preds, df)
    else:
        boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(float)

    results = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(_one)(p, b, opts) for p, b in tqdm(list(zip(df["path"], boxes)), desc=f"{variant}/{split}"))
    feats, times = zip(*results)
    arrays = {g: np.stack([f[g] for f in feats]) for g in GROUPS}
    out = cache_path(variant, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, filenames=np.asarray(df["filename"], dtype=str),
                        class_id=df["class_id"].to_numpy(), extract_ms=np.array(times), **arrays)
    print(f"{out}  ({len(df)} images, {np.mean(times):.1f} ms/img/core)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-gamma", action="store_true")
    ap.add_argument("--no-grabcut", action="store_true")
    ap.add_argument("--test-crop", default="det", choices=["det", "gt"])
    ap.add_argument("--det-preds", type=Path, default=config.RESULTS_DIR / "preds" / "detector_yolo11s_cars.npz")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    opts = {"use_clahe": not args.no_clahe, "use_gamma": not args.no_gamma, "use_grabcut": not args.no_grabcut}
    for split in args.splits.split(","):
        extract_split(split, args.variant, opts, args.test_crop, args.det_preds, args.jobs)


if __name__ == "__main__":
    main()
