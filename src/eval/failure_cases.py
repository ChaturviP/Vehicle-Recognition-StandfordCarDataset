"""Render failure-case figures for the failure analysis (CPU only, ~1 min).

    python -m src.eval.failure_cases

Uses the end-to-end test predictions (results/preds/convnext_base_448_detcrop.npz), the
detector predictions and the index. Writes to results/failures/:
  1_similar_classes.png  - most-confused class pairs: test image | an example of the predicted class
  2_detector_misses.png  - lowest-IoU detections: ground truth (green) vs prediction (red)
  3_confident_errors.png - wrong answers the model was most confident about
  4_occlusion_resolution.png - one car under 50% occlusion and 64 px resolution, with predictions
  summary.json           - counts behind each category
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config
from src.data.dataset import crop_to_box, load_index
from src.eval import metrics as M

OUT = config.RESULTS_DIR / "failures"
PREDS = config.RESULTS_DIR / "preds"


def _load(name):
    d = np.load(PREDS / f"{name}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def _short(name, n=34):
    return name if len(name) <= n else name[:n - 1] + "…"


def similar_classes(test, probs, names, train, top=6):
    y, pred = test["class_id"].to_numpy(), probs.argmax(1)
    pairs = M.confused_pairs(probs, y, top)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(7, 2.2 * len(pairs)))
    rows = []
    for r, (a, b, n) in enumerate(pairs):
        i = np.where((y == a) & (pred == b))[0][0]
        t = test.iloc[i]
        ex = train[train.class_id == b].iloc[0]
        for c, (row, title) in enumerate([(t, f"TRUE: {_short(names[a])}"), (ex, f"PREDICTED: {_short(names[b])}")]):
            ax = axes[r, c]
            ax.imshow(crop_to_box(Image.open(row.path).convert("RGB"), (row.x1, row.y1, row.x2, row.y2)))
            ax.set_title(title, fontsize=7, color="green" if c == 0 else "red")
            ax.axis("off")
        axes[r, 0].text(-0.02, 0.5, f"{n}×", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
        rows.append({"true": names[a], "predicted": names[b], "count": n})
    fig.suptitle("Most-confused class pairs (left: test image, right: a training image of the predicted class)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "1_similar_classes.png", dpi=120)
    plt.close(fig)
    return rows


def detector_misses(test, det, top=6):
    gt = test[["x1", "y1", "x2", "y2"]].to_numpy(float)
    iou = M.box_iou(det["boxes"], gt)
    idx = np.argsort(iou)[:top]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, i in zip(axes.ravel(), idx):
        t = test.iloc[i]
        ax.imshow(Image.open(t.path).convert("RGB"))
        for (x1, y1, x2, y2), col in [(gt[i], "lime"), (det["boxes"][i], "red")]:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=2))
        ax.set_title(f"IoU {iou[i]:.2f} · {_short(t.class_name, 28)}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Lowest-IoU detections (green = ground truth, red = YOLO's top box)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "2_detector_misses.png", dpi=120)
    plt.close(fig)
    return {"iou_lt_0.5": int((iou < 0.5).sum()), "iou_eq_0": int((iou < 0.01).sum()),
            "worst": [{"file": test.iloc[i].filename, "iou": float(iou[i])} for i in idx]}


def confident_errors(test, probs, names, top=6):
    y, pred, conf = test["class_id"].to_numpy(), probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    idx = wrong[np.argsort(-conf[wrong])][:top]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for ax, i in zip(axes.ravel(), idx):
        t = test.iloc[i]
        ax.imshow(crop_to_box(Image.open(t.path).convert("RGB"), (t.x1, t.y1, t.x2, t.y2)))
        ax.set_title(f"true: {_short(names[y[i]], 30)}\npred: {_short(names[pred[i]], 30)} ({conf[i]:.0%})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Most confident mistakes (raw ConvNeXt confidence)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "3_confident_errors.png", dpi=120)
    plt.close(fig)
    return {"n_wrong": int(len(wrong)), "n_wrong_conf_ge_0.85": int((conf[wrong] >= 0.85).sum())}


def occlusion_resolution(test, names):
    """Show what the occlusion-50 and 64 px corruptions look like and what the model predicts."""
    import torch
    from src.deep.model import build_transforms, load_checkpoint
    from src.eval.robustness import CONDITIONS

    ckpt = config.MODELS_DIR / "classifier_convnext_base_448.pt"
    if not ckpt.exists():
        return None
    model, ck = load_checkpoint(ckpt, "cpu")  # CPU: the GPU may be busy training; 15 forward passes
    dev = next(model.parameters()).device
    tf = build_transforms(ck["args"]["img"], False)
    picks = test.sample(3, random_state=7)
    conds = ["clean", "occlusion25", "occlusion50", "res128", "res64"]
    fig, axes = plt.subplots(len(picks), len(conds), figsize=(2.4 * len(conds), 2 * len(picks)))
    for r, (k, t) in enumerate(picks.iterrows()):
        crop = crop_to_box(Image.open(t.path).convert("RGB"), (t.x1, t.y1, t.x2, t.y2))
        for c, cond in enumerate(conds):
            fn = CONDITIONS[cond]
            img = crop if fn is None else fn(crop, np.random.default_rng(config.SEED + int(k)))
            with torch.no_grad():
                p = torch.softmax(model(tf(img)[None].to(dev)).float(), 1)[0].cpu().numpy()
            ok = p.argmax() == t.class_id
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_title(f"{cond}\n{_short(names[p.argmax()], 24)} {p.max():.0%}", fontsize=6.5,
                         color="green" if ok else "red")
            ax.axis("off")
    fig.suptitle("Occlusion and resolution failures (green = correct, red = wrong)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "4_occlusion_resolution.png", dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    names = idx.drop_duplicates("class_id").sort_values("class_id")["class_name"].tolist()
    test, train = idx[idx.split == "test"].set_index("filename"), idx[idx.split == "train"]
    deep, det = _load("convnext_base_448_detcrop"), _load("detector_yolo11s_cars")
    t_deep = test.loc[deep["filenames"]].reset_index()
    t_det = test.loc[det["filenames"]].reset_index()

    summary = {"similar_class_pairs": similar_classes(t_deep, deep["probs"], names, train),
               "detector": detector_misses(t_det, det),
               "confident_errors": confident_errors(t_deep, deep["probs"], names)}
    y, pred = t_deep["class_id"].to_numpy(), deep["probs"].argmax(1)
    w = pred != y
    same_make = (t_deep["make_id"].to_numpy() == idx.drop_duplicates("class_id").set_index("class_id")
                 .loc[pred, "make_id"].to_numpy())
    model_of = lambda n: n.rsplit(" ", 1)[0]
    same_model_other_year = np.array([model_of(names[a]) == model_of(names[b]) for a, b in zip(y, pred)])
    summary["error_breakdown"] = {"errors": int(w.sum()), "same_make": int((w & same_make).sum()),
                                  "same_model_different_year": int((w & same_model_other_year).sum())}
    try:
        summary["occlusion_resolution_figure"] = occlusion_resolution(t_deep, names)
    except Exception as e:  # torch / checkpoint optional
        summary["occlusion_resolution_figure"] = f"skipped: {e}"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["error_breakdown"]), "| detector", {k: v for k, v in summary["detector"].items() if k != "worst"})
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
