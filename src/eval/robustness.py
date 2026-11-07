"""Robustness of the classifier (and optionally the detector) under controlled corruptions.

    python -m src.eval.robustness --ckpt models/classifier_convnext_base_448.pt
    python -m src.eval.robustness --ckpt models/classifier_convnext_base_448.pt --detector models/detector_yolo11s_cars.pt

Classifier: each test image is cropped with the (clean) detector box, the crop is
corrupted, then classified. Conditions cover illumination (gamma, brightness, low light
with sensor noise), blur, occlusion (random block covering 10/25/50% of the crop) and
resolution (crop downscaled to 64/128/224 px on its long side, then upscaled back).
"+clahe" rows apply the classical CLAHE + auto-gamma enhancement to the corrupted crop
before the CNN: the hybrid preprocessing path.
Pose: accuracy on clean images bucketed by detected-box aspect ratio (front/rear views
are narrow boxes, side views are wide).

Uses a fixed random subset of the test split (--n, default 4000) to keep runtime
~10-15 min on a 4070. Writes results/preds/<base>_rob_<condition>.npz per condition,
results/robustness/summary.json and results/robustness/robustness.png.
"""
from __future__ import annotations

import argparse
import json
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src import config
from src.classical.preprocess import auto_gamma, clahe_lab
from src.data.dataset import crop_to_box, load_det_boxes, load_index
from src.deep.model import build_transforms, load_checkpoint
from src.eval.evaluate import save_predictions

OUT_DIR = config.RESULTS_DIR / "robustness"


# ---------- corruptions (PIL RGB in, PIL RGB out; seeded per image) ----------

def _np(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def gamma(img, rng, g):
    return _pil(_np(img) ** g)


def brightness(img, rng, f):
    return _pil(_np(img) * f)


def low_light(img, rng):
    a = _np(img) * 0.3
    return _pil(a + rng.normal(0, 0.04, a.shape))


def blur(img, rng, sigma):
    return _pil(cv2.GaussianBlur(_np(img), (0, 0), sigma))


def occlude(img, rng, frac):
    a = _np(img).copy()
    h, w = a.shape[:2]
    ar = rng.uniform(0.5, 2.0)
    bh = int(min(h, np.sqrt(frac * h * w / ar)))
    bw = int(min(w, frac * h * w / max(bh, 1)))
    y, x = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
    a[y:y + bh, x:x + bw] = 0.5
    return _pil(a)


def downscale(img, rng, size):
    w, h = img.size
    s = size / max(w, h)
    if s >= 1:
        return img
    small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def clahe_enhance(img):
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    bgr, _ = auto_gamma(clahe_lab(bgr))
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


CONDITIONS = {
    "clean": None,
    "dark_gamma2.2": partial(gamma, g=2.2),
    "bright_gamma0.45": partial(gamma, g=0.45),
    "brightness-40": partial(brightness, f=0.6),
    "brightness+40": partial(brightness, f=1.4),
    "lowlight_noise": low_light,
    "blur_sigma2": partial(blur, sigma=2.0),
    "occlusion10": partial(occlude, frac=0.10),
    "occlusion25": partial(occlude, frac=0.25),
    "occlusion50": partial(occlude, frac=0.50),
    "res224": partial(downscale, size=224),
    "res128": partial(downscale, size=128),
    "res64": partial(downscale, size=64),
}
CLAHE_ON = ["clean", "dark_gamma2.2", "brightness-40", "lowlight_noise"]


class CorruptedCrops(Dataset):
    def __init__(self, paths, boxes, corrupt, use_clahe, transform, seed=config.SEED):
        self.paths, self.boxes, self.corrupt = list(paths), boxes, corrupt
        self.use_clahe, self.transform, self.seed = use_clahe, transform, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = crop_to_box(Image.open(self.paths[i]).convert("RGB"), self.boxes[i])
        if self.corrupt is not None:
            img = self.corrupt(img, np.random.default_rng(self.seed + i))
        if self.use_clahe:
            img = clahe_enhance(img)
        return self.transform(img)


@torch.no_grad()
def predict_logits(model, ds, device, batch, workers):
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = []
    for x in DataLoader(ds, batch, num_workers=workers, pin_memory=True):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def pose_buckets(df, boxes, correct) -> list[dict]:
    aspect = (boxes[:, 2] - boxes[:, 0]) / np.maximum(boxes[:, 3] - boxes[:, 1], 1)
    edges = [(0, 1.3, "narrow (<1.3): front / rear views"), (1.3, 1.8, "medium (1.3-1.8): three-quarter"),
             (1.8, 99, "wide (>=1.8): side views")]
    return [{"bucket": name, "n": int(m.sum()), "top1": float(correct[m].mean()) if m.any() else None}
            for lo, hi, name in edges for m in [(aspect >= lo) & (aspect < hi)]]


def detector_robustness(weights, df, n, conds) -> list[dict]:
    """Detector IoU under whole-image corruptions (occlusion placed inside the GT box)."""
    from ultralytics import YOLO
    from src.eval.metrics import box_iou

    yolo = YOLO(weights)
    sub = df.sample(min(n, len(df)), random_state=config.SEED)
    gt = sub[["x1", "y1", "x2", "y2"]].to_numpy(float)
    rows = []
    for cond in conds:
        fn, boxes = CONDITIONS[cond], []
        for k, r in enumerate(sub.itertuples()):
            img = Image.open(r.path).convert("RGB")
            if fn is not None:
                rng = np.random.default_rng(config.SEED + k)
                if cond.startswith("occlusion"):
                    car = img.crop((int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
                    img.paste(fn(car, rng), (int(r.x1), int(r.y1)))
                else:
                    img = fn(img, rng)
            res = yolo.predict(np.asarray(img)[:, :, ::-1], conf=0.25, verbose=False, half=True)[0]
            b = res.boxes
            boxes.append(b.xyxy[b.conf.argmax()].cpu().numpy() if len(b) else np.zeros(4))
        iou = box_iou(np.array(boxes), gt)
        rows.append({"condition": cond, "n": len(sub), "mean_iou": float(iou.mean()),
                     "iou_ge_0.5": float((iou >= 0.5).mean())})
        print(f"  detector {cond:<18} mean IoU {iou.mean():.4f}  IoU>=0.5 {(iou >= 0.5).mean():.3f}")
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = [r for r in rows if not r["clahe"]]
    clahe = {r["condition"]: r["top1"] for r in rows if r["clahe"]}
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(len(base))
    ax.bar(xs - 0.2, [r["top1"] for r in base], 0.4, label="CNN on corrupted crop", color="#4C72B0")
    ax.bar(xs + 0.2, [clahe.get(r["condition"], np.nan) for r in base], 0.4,
           label="+ classical CLAHE/gamma (hybrid)", color="#DD8452")
    ax.set_xticks(xs, [r["condition"] for r in base], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="top-1 accuracy", ylim=(0, 1), title="Classifier robustness (test subset)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--det-preds", default=str(config.RESULTS_DIR / "preds" / "detector_yolo11s_cars.npz"))
    ap.add_argument("--n", type=int, default=4000, help="test images to use (fixed random subset)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--no-clahe-rows", action="store_true")
    ap.add_argument("--detector", default=None, help="also test detector IoU (weights path)")
    ap.add_argument("--det-n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    model = model.to(memory_format=torch.channels_last)
    hp = ckpt["args"]
    base = hp["name"]
    T = 1.0
    cfg = config.MODELS_DIR / "hybrid_config.json"
    if cfg.exists():
        T = json.loads(cfg.read_text())["temperature"]

    df_all = load_index("test")
    boxes_all = load_det_boxes(args.det_preds, df_all)
    sub = df_all.sample(min(args.n, len(df_all)), random_state=config.SEED).index.to_numpy()
    df, boxes = df_all.iloc[sub].reset_index(drop=True), boxes_all[sub]
    y = df["class_id"].to_numpy()
    tf = build_transforms(hp["img"], False)

    runs = [(c, False) for c in args.conditions.split(",")]
    if not args.no_clahe_rows:
        runs += [(c, True) for c in CLAHE_ON if c in args.conditions.split(",")]
    rows, clean_correct = [], None
    for cond, use_clahe in runs:
        ds = CorruptedCrops(df["path"], boxes, CONDITIONS[cond], use_clahe, tf)
        logits = predict_logits(model, ds, device, args.batch, args.workers)
        probs = torch.softmax(torch.from_numpy(logits) / T, 1).numpy()
        correct = probs.argmax(1) == y
        top5 = (np.argsort(-probs, 1)[:, :5] == y[:, None]).any(1).mean()
        tag = cond + ("+clahe" if use_clahe else "")
        save_predictions(f"{base}_rob_{tag}", df["filename"], probs=probs, family="deep", condition=tag,
                         base_model=base, n_subset=len(df), temperature=T)
        rows.append({"condition": cond, "clahe": use_clahe, "top1": float(correct.mean()),
                     "top5": float(top5), "mean_conf": float(probs.max(1).mean())})
        if cond == "clean" and not use_clahe:
            clean_correct = correct
        print(f"{tag:<24} top-1 {correct.mean():.4f}  top-5 {top5:.4f}  mean conf {probs.max(1).mean():.3f}")

    summary = {"model": base, "n_images": len(df), "temperature": T, "classifier": rows}
    if clean_correct is not None:
        summary["pose_buckets_clean"] = pose_buckets(df, boxes, clean_correct)
        for b in summary["pose_buckets_clean"]:
            print(f"  pose {b['bucket']:<38} n={b['n']:<5} top-1 {b['top1']:.4f}")
    if args.detector:
        summary["detector"] = detector_robustness(
            args.detector, df_all, args.det_n,
            ["clean", "dark_gamma2.2", "lowlight_noise", "blur_sigma2", "occlusion25", "occlusion50", "res128", "res64"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, OUT_DIR / "robustness.png")
    print(f"Wrote {OUT_DIR / 'summary.json'} and {OUT_DIR / 'robustness.png'}")


if __name__ == "__main__":
    main()
