"""Run a trained classifier over a split and save predictions for the harness.

    # end-to-end: crops from the fine-tuned detector's boxes (the real system)
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt \
        --crop det --det-preds results/preds/detector_yolo11s_cars.npz
    # oracle crops (ground-truth boxes) and full images, for the ablation table
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop gt
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop none
    # validation split (needed later for temperature scaling and fusion weights)
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --split val --crop gt

Saves results/preds/<name>.npz with probs, logits, make_logits; embeddings go to
data/cache/emb_<name>.npy (used by the hybrid fusion step).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data.dataset import CarsDataset, load_det_boxes, load_index
from src.deep.model import build_transforms, load_checkpoint
from src.eval.evaluate import save_predictions


@torch.no_grad()
def predict(model, loader, device, tta: bool):
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    all_logits, all_make, all_emb = [], [], []
    for x, _, _, _ in tqdm(loader, desc="predict"):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
            logits, make_logits, emb = model(x, return_all=True)
            if tta:
                l2, m2, e2 = model(torch.flip(x, dims=[3]), return_all=True)
                logits, emb = (logits + l2) / 2, (emb + e2) / 2
                make_logits = (make_logits + m2) / 2 if make_logits is not None else None
        all_logits.append(logits.float().cpu())
        all_emb.append(emb.float().cpu())
        if make_logits is not None:
            all_make.append(make_logits.float().cpu())
    cat = lambda xs: torch.cat(xs).numpy() if xs else None
    return cat(all_logits), cat(all_make), cat(all_emb)


@torch.no_grad()
def benchmark_latency(model, ds, device, n: int = 200) -> np.ndarray:
    """Per-image latency at batch size 1: crop + transform + forward."""
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    times = []
    for i in range(min(n + 10, len(ds))):
        t0 = time.perf_counter()
        x = ds[i][0][None].to(device).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 10:  # skip warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--crop", default="det", choices=["det", "gt", "none"])
    ap.add_argument("--det-preds", type=Path, help="detector npz (required for --crop det)")
    ap.add_argument("--tta", action="store_true", help="average with horizontally flipped input")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    model = model.to(memory_format=torch.channels_last)
    hp = ckpt["args"]

    df = load_index(args.split)
    if args.crop == "det":
        if not args.det_preds:
            raise SystemExit("--crop det needs --det-preds results/preds/<detector>.npz")
        ds = CarsDataset(df, build_transforms(hp["img"], False), crop="pred", boxes=load_det_boxes(args.det_preds, df))
    else:
        ds = CarsDataset(df, build_transforms(hp["img"], False), crop=args.crop)
    dl = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    logits, make_logits, emb = predict(model, dl, device, args.tta)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    latency = benchmark_latency(model, ds, device)

    name = args.name or f"{hp['name']}_{args.crop}crop{'_tta' if args.tta else ''}" + \
        ("" if args.split == "test" else f"_{args.split}")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.CACHE_DIR / f"emb_{name}.npy", emb.astype(np.float16))
    out = save_predictions(
        name, df["filename"], split=args.split, family="deep", probs=probs,
        latency_ms=latency, model_size_mb=args.ckpt.stat().st_size / 2**20,
        extra_arrays={"logits": logits, "make_logits": make_logits},
        device=(torch.cuda.get_device_name(0) + " bf16") if device == "cuda" else "CPU fp32",
        backbone=hp["backbone"], img_size=hp["img"], crop=args.crop, tta=args.tta,
        det_preds=str(args.det_preds) if args.det_preds else None, ckpt=str(args.ckpt),
    )
    acc = (probs.argmax(1) == df["class_id"].to_numpy()).mean()
    print(f"{name}: top-1 {acc:.4f}, latency {latency.mean():.1f} ms/img (batch 1)\nSaved {out}")


if __name__ == "__main__":
    main()
