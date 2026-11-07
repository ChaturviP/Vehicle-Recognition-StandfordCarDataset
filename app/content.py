"""UI-independent content shared by the Gradio (app.py) and Streamlit (streamlit_app.py) front ends."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
ASSETS = ROOT / "assets" if (ROOT / "assets").exists() else ROOT / "results"

LEVEL_BADGE = {
    "model": "✅ **Confident: make, model and year**",
    "make": "🟠 **Manufacturer only**: the exact model/year is below the confidence threshold",
    "uncertain": "🔴 **Low confidence**: showing the top-5 candidates instead of one answer",
    "no_car": "⚪ **No car detected**",
}


def verdict_markdown(res) -> str:
    if res.level == "no_car":
        return LEVEL_BADGE["no_car"] + "\n\nTry a clearer photo where the car fills more of the frame."
    t = res.timings
    return "\n".join([
        LEVEL_BADGE[res.level], "", f"### {res.headline}", "",
        f"Detector box confidence {res.box_score:.0%}"
        + (f" · {len(res.other_boxes)} other vehicle(s) ignored (grey boxes)" if res.other_boxes else ""),
        "",
        f"Timing (CPU): detector {t['detector_ms']:.0f} ms · ConvNeXt INT8 {t['deep_ms']:.0f} ms · "
        f"classical pipeline {t['classical_ms']:.0f} ms"
        + (" · classical scores were fused (deep confidence below the gate)" if res.fused else "")])


def branch_table(res) -> str:
    rows = ["| Approach | Top-1 | Conf. | 2nd | 3rd | Time |", "|---|---|---|---|---|---|"]
    for name, b in res.per_branch.items():
        (n1, p1), (n2, p2), (n3, p3) = b["top3"]
        rows.append(f"| {name} | **{n1}** | {p1:.0%} | {n2} ({p2:.0%}) | {n3} ({p3:.0%}) | {b['ms']:.0f} ms |")
    return "\n".join(rows)


def _metrics(name):
    for d in (ASSETS / "metrics", ROOT / "results" / "metrics"):
        p = d / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def comparison_markdown() -> str:
    rows = [("Classical: handcrafted features + RBF SVM", "classical_svm_full"),
            ("Deep: ConvNeXt-Base @448 on YOLO crops", "convnext_base_448_detcrop"),
            ("Hybrid: calibrated deep + gated classical fusion", "hybrid_full")]
    L = ["### Classical vs deep vs hybrid (test split, 8,041 unseen images)", "",
         "| Approach | Top-1 | Top-5 | Make top-1 | ECE (calibration) | Size | Latency |",
         "|---|---|---|---|---|---|---|"]
    for label, name in rows:
        m = _metrics(name)
        if m is None:
            continue
        c, e = m["classification"], m.get("efficiency", {})
        L.append(f"| {label} | {c['top1']:.2%} | {c['top5']:.2%} | {c['make_top1']:.2%} | {c['ece']:.3f} | "
                 f"{e.get('model_size_mb', 0):.0f} MB | {e.get('latency_ms_mean', 0):.0f} ms |")
    L += ["", "Latency: classical = feature extraction on 1 CPU core; deep/hybrid = RTX 4070 bf16. "
          "This demo runs the INT8 ONNX model on CPU.", ""]
    d = _metrics("detector_yolo11s_cars")
    z = _metrics("yolo11s_coco_zeroshot")
    if d:
        L += ["### Vehicle detection (test split)", "", "| Detector | Mean IoU | mAP@0.5 | mAP@0.5:0.95 |",
              "|---|---|---|---|"]
        for label, m in [("YOLO11s fine-tuned on Stanford Cars", d), ("YOLO11s COCO zero-shot (baseline)", z)]:
            if m:
                x = m["detection"]
                L.append(f"| {label} | {x['mean_iou']:.3f} | {x['map50']:.3f} | {x['map50_95']:.3f} |")
    opt = None
    for p in (ASSETS / "optimize_summary.json", ROOT / "results" / "optimize" / "summary.json"):
        if p.exists():
            opt = json.loads(p.read_text())
            break
    if opt:
        L += ["", "### Model optimisation (CPU, 2 threads, 500 test images)", "",
              "| Classifier variant | Size | Latency | Top-1 |", "|---|---|---|---|"]
        for r in opt:
            if r.get("model") != "detector":
                L.append(f"| {r['variant']} | {r['size_mb']:.0f} MB | {r['latency_ms']:.0f} ms | {r['top1']:.1%} |")
    return "\n".join(L)


def img_path(*names):
    for n in names:
        for d in (ASSETS, ROOT / "results"):
            p = d / n
            if p.exists():
                return str(p)
    return None


ABOUT = """
**Pipeline.** Upload → YOLO11s detector (fine-tuned on Stanford Cars) finds the car → the box is
padded by 12% and cropped → two branches:

* **Classical CV**: CLAHE (LAB L-channel) and auto-gamma illumination normalisation, bilateral
  filtering, GrabCut foreground segmentation seeded by the box, Canny edges, contour/convex-hull
  analysis, then HSV colour histograms, LBP and GLCM texture, HOG, Hu moments and box geometry →
  PCA → RBF SVM.
* **Deep learning**: ConvNeXt-Base (ImageNet-22k pre-trained, fine-tuned at 448 px with an auxiliary
  manufacturer head), exported to ONNX and quantised to INT8 for CPU.

**Hybrid decision.** Deep logits are temperature-scaled (calibrated on a held-out validation split).
When the deep model is unsure (< 50%), the classical SVM's scores are fused in. The calibrated
confidence then decides the answer level: full make/model/year, manufacturer only, or "uncertain"
with the top-5 candidates. On the test split this answers 91% of images at 97.6% accuracy, gives
the make only for 4% (99.7% correct), and flags 5% as uncertain.

**Limits.** The model knows only the 196 Stanford Cars classes (mostly 1991–2012 US-market cars).
A newer or unseen model will be mapped to its closest known class; the confidence policy is
there to flag such cases. It identifies the single most prominent car per image.
"""
