"""Collect raw ablation runs (summarised in docs/ablation_study.md) (one row per component removed/changed).

    python -m src.eval.ablation

Rows are filled from results/metrics/<run>.json and results/robustness/summary.json;
runs that have not been produced yet are listed as "not run". Commands for each run:

  inference-only (GPU, a few minutes each):
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop gt
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop none
    python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop det --det-preds results/preds/detector_yolo11s_cars.npz --tta
  classical (CPU):
    python -m src.classical.extract_features --variant nogc --no-grabcut
    python -m src.classical.train_classical --variant nogc --pca 256 --C 3
    python -m src.classical.extract_features --variant noclahe --no-clahe --no-gamma
    python -m src.classical.train_classical --variant noclahe --pca 256 --C 3
    python -m src.classical.train_classical --groups hog --pca 256 --C 3
    python -m src.classical.train_classical --groups color,lbp,glcm,edge,shape,geom --pca 128 --C 3
  retraining (GPU, 25-60 min each; then predict with --crop det or --crop none):
    python -m src.deep.train_classifier --name convnext_224 --img 224 --batch 32 --accum 1
    python -m src.deep.train_classifier --name convnext_nomake --make-weight 0
    python -m src.deep.train_classifier --name convnext_fullimg --crop none
Then run `python -m src.eval.evaluate results/preds/<new>.npz` for each and re-run this script.
"""
from __future__ import annotations

import json

from src import config
from src.eval.evaluate import METRICS_DIR

D = "convnext_base_448_detcrop"  # reference deep run (end-to-end)
C = "classical_svm_full"         # reference classical run

ROWS = [
    # (group, row label, what changed, metrics run name, reference run name)
    ("Detection / ROI", "Full system: YOLO crop → ConvNeXt", "reference", D, None),
    ("Detection / ROI", "Oracle crop (ground-truth box)", "detector replaced by GT box", "convnext_base_448_gtcrop", D),
    ("Detection / ROI", "No ROI crop at test (full image)", "detection + cropping removed", "convnext_base_448_nonecrop", D),
    ("Detection / ROI", "Trained and tested on full images", "ROI cropping removed everywhere", "convnext_fullimg_nonecrop", D),
    ("Classifier", "Input 224 px instead of 448 px", "resolution", "convnext_224_detcrop", D),
    ("Classifier", "No auxiliary make head", "hierarchical loss removed", "convnext_nomake_detcrop", D),
    ("Classifier", "+ flip test-time augmentation", "TTA added", "convnext_base_448_detcrop_tta", D),
    ("Classifier", "Ensemble: crop model + full-image model", "2 models averaged", "ensemble_crop_full", D),
    ("Hybrid", "Hybrid on the 2-model ensemble", "ensemble + calibration + fusion", "hybrid_ensemble", "hybrid_full"),
    ("Hybrid", "+ temperature scaling", "calibration added (ECE)", "convnext_base_448_detcrop_calibrated", D),
    ("Hybrid", "+ gated classical score fusion", "fusion added", "hybrid_full", "convnext_base_448_detcrop_calibrated"),
    ("Classical", "Classical full (all features, GrabCut, CLAHE)", "reference", C, None),
    ("Classical", "No GrabCut (box as mask)", "GrabCut removed", "classical_svm_nogc", C),
    ("Classical", "No CLAHE / gamma", "illumination normalisation removed", "classical_svm_noclahe", C),
    ("Classical", "HOG only", "colour/texture/shape/geometry removed", "classical_svm_full_hog", C),
    ("Classical", "No HOG", "HOG removed", "classical_svm_full_color+lbp+glcm+edge+shape+geom", C),
    ("Detector", "Zero-shot COCO YOLO11s (no fine-tuning)", "fine-tuning removed (mAP@0.5:0.95)",
     "yolo11s_coco_zeroshot", "detector_yolo11s_cars"),
]


def _load(name):
    p = METRICS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _val(m, ece=False):
    if m is None:
        return None
    if "detection" in m and "classification" not in m:
        return m["detection"]["map50_95"]
    return m["classification"]["ece" if ece else "top1"]


def main() -> None:
    L = ["# Ablation study", "",
         "Test split (8,041 images) unless noted. Δ is relative to the row's reference. "
         "Detector rows show mAP@0.5:0.95; calibration rows also show ECE.", ""]
    group = None
    for g, label, change, run, ref in ROWS:
        if g != group:
            L += ["", f"## {g}", "", "| Variant | Change | Top-1 | Δ vs reference | ECE |", "|---|---|---|---|---|"]
            group = g
        m, r = _load(run), _load(ref) if ref else None
        v, rv = _val(m), _val(r)
        if v is None:
            L.append(f"| {label} | {change} | not run | – | – |")
            continue
        delta = "–" if rv is None else f"{100 * (v - rv):+.2f} pts"
        ece = m.get("classification", {}).get("ece")
        L.append(f"| {label} | {change} | {100 * v:.2f}% | {delta} | {'–' if ece is None else f'{ece:.4f}'} |")
    rob = config.RESULTS_DIR / "robustness" / "summary.json"
    if rob.exists():
        s = json.loads(rob.read_text())
        base = {r["condition"]: r["top1"] for r in s["classifier"] if not r["clahe"]}
        L += ["", "## Hybrid preprocessing under bad lighting (classical CLAHE + gamma before the CNN)", "",
              f"Test subset of {s['n_images']} images.", "",
              "| Condition | CNN only | + CLAHE/gamma | Δ |", "|---|---|---|---|"]
        for r in s["classifier"]:
            if r["clahe"] and r["condition"] in base:
                L.append(f"| {r['condition']} | {100 * base[r['condition']]:.2f}% | {100 * r['top1']:.2f}% | "
                         f"{100 * (r['top1'] - base[r['condition']]):+.2f} pts |")
    out = config.RESULTS_DIR / "ablation.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
