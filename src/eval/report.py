"""Render all results into eval_P5_<TeamID>.md at the repo root.

    python -m src.eval.report --team T01 --usns "1RVU23CSE001,1RVU23CSE002"

The first line of the file is the comma-separated USNs; sections follow the order of the
problem statement's evaluation table. Sources: results/metrics/*.json (harness output),
results/optimize/summary.json, results/robustness/summary.json, results/hybrid/hybrid_summary.json.
Missing sources are skipped, so the file can be regenerated at any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config
from src.eval.evaluate import METRICS_DIR

FAMILY_ORDER = {"classical": 0, "deep": 1, "hybrid": 2, "detector": 3}
PRIMARY = [("Classical: handcrafted features + RBF SVM", "classical_svm_full"),
           ("Deep: ConvNeXt-Base @448 on YOLO crops", "convnext_base_448_detcrop"),
           ("Hybrid: calibrated deep + gated classical fusion", "hybrid_full")]
LABELS = {
    "convnext_base_448_gtcrop": "Deep, ground-truth crop (oracle detector)",
    "convnext_base_448_nonecrop": "Deep, full image (no ROI crop)",
    "convnext_base_448_detcrop_tta": "Deep + flip TTA",
    "convnext_base_448_detcrop_calibrated": "Deep + temperature scaling",
    "convnext_224_detcrop": "Deep, 224 px input",
    "convnext_nomake_detcrop": "Deep, no auxiliary make head",
    "convnext_fullimg_nonecrop": "Deep, trained + tested on full images",
    "ensemble_crop_full": "Deep ensemble: crop model + full-image model",
    "ensemble_crop_full_calibrated": "Deep ensemble + temperature scaling",
    "hybrid_ensemble": "Hybrid on the 2-model ensemble",
    "classical_svm_full_hog": "Classical, HOG features only",
    "classical_svm_full_color+lbp+glcm+edge+shape+geom": "Classical, all features except HOG",
    "classical_svm_nogc": "Classical, no GrabCut",
    "classical_svm_noclahe": "Classical, no CLAHE/gamma",
    "detector_yolo11s_cars": "YOLO11s fine-tuned on Stanford Cars",
    "yolo11s_coco_zeroshot": "YOLO11s COCO zero-shot (car+truck), baseline",
}


def _pct(v) -> str:
    return "–" if v is None or v != v else f"{100 * v:.2f}%"  # v != v catches NaN


def _num(v, fmt="{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_results(metrics_dir: Path = METRICS_DIR) -> list[dict]:
    res = [json.loads(p.read_text()) for p in sorted(metrics_dir.glob("*.json"))]
    return sorted(res, key=lambda r: (FAMILY_ORDER.get(r["meta"]["family"], 9), r["meta"]["name"]))


def render(results: list[dict], team: str, usns: list[str], results_dir: Path = config.RESULTS_DIR) -> str:
    by_name = {r["meta"]["name"]: r for r in results}
    is_subset = lambda r: bool(r["meta"].get("n_subset")) or "base_model" in r["meta"]
    full = [r for r in results if r["meta"].get("condition", "clean") == "clean" and not is_subset(r)]
    cls = [r for r in full if "classification" in r]
    det = [r for r in full if "detection" in r]
    opt = _json(results_dir / "optimize" / "summary.json")
    rob = _json(results_dir / "robustness" / "summary.json")
    hyb = _json(results_dir / "hybrid" / "hybrid_summary.json")
    label = lambda r: LABELS.get(r["meta"]["name"], r["meta"]["name"])

    L = [",".join(usns), "", f"# Evaluation Results: P5 Fine-Grained Vehicle Recognition, Team {team}", "",
         "Unless noted, every number is on the official Stanford Cars **test split: 8,041 images never used "
         "for training or tuning** (hyper-parameters, temperature, fusion weight and thresholds were selected "
         "on a separate validation split carved from the training set). Test-time crops come from the "
         "YOLO detector, so classification numbers are end-to-end.", ""]

    # 1 ---------------------------------------------------------------------
    L += ["## 1. Classification performance (Top-1 / Top-5)", "",
          "| Approach | Top-1 | Top-5 | Make top-1 |", "|---|---|---|---|"]
    for lab, name in PRIMARY:
        if name in by_name and "classification" in by_name[name]:
            c = by_name[name]["classification"]
            L.append(f"| **{lab}** | **{_pct(c['top1'])}** | {_pct(c['top5'])} | {_pct(c['make_top1'])} |")
    others = [r for r in cls if r["meta"]["name"] not in dict((n, l) for l, n in PRIMARY)]
    if others:
        L += ["", "Additional variants (ablations, same test split):", "",
              "| Variant | Family | Top-1 | Top-5 | Make top-1 |", "|---|---|---|---|---|"]
        L += [f"| {label(r)} | {r['meta']['family']} | {_pct(r['classification']['top1'])} | "
              f"{_pct(r['classification']['top5'])} | {_pct(r['classification']['make_top1'])} |" for r in others]
    if "ensemble_crop_full" in by_name:
        e = by_name["ensemble_crop_full"]["classification"]
        L.append(f"\n**Best result: two-model ensemble, {_pct(e['top1'])} top-1 / {_pct(e['top5'])} top-5.** "
                 "The crop model and the full-image model make largely different errors (300 of ~400 shared); "
                 "averaging them was adopted because it also beats the single model on the validation split "
                 "(95.42% vs 95.09%). The deployed app uses the single crop model (half the size and CPU time).")
    L.append("\nChance level: top-1 0.51%, top-5 2.55% (196 classes).")

    # 2 / 3 -----------------------------------------------------------------
    L += ["", "## 2. Object localization (IoU)", "",
          "| Detector | Mean IoU | Median IoU | IoU ≥ 0.5 | IoU ≥ 0.75 |", "|---|---|---|---|---|"]
    L += [f"| {label(r)} | {_num(r['detection']['mean_iou'], '{:.4f}')} | {_num(r['detection']['median_iou'], '{:.4f}')} | "
          f"{_pct(r['detection']['iou_ge_0.5'])} | {_pct(r['detection']['iou_ge_0.75'])} |" for r in det]
    L += ["", "IoU is computed between the detector's highest-scoring box and the single annotated box per image.",
          "", "## 3. Object detection (mAP)", "", "| Detector | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|"]
    L += [f"| {label(r)} | {_num(r['detection']['map50'], '{:.4f}')} | {_num(r['detection']['map50_95'], '{:.4f}')} |"
          for r in det]
    L += ["", "Single-class AP (COCO 101-point interpolation) over all detections (confidence ≥ 0.001); "
          "mAP@0.5:0.95 averages IoU thresholds 0.50:0.05:0.95."]

    # 4 ---------------------------------------------------------------------
    L += ["", "## 4. Model efficiency (size, inference time)", "",
          "| Model | Size (MB) | Latency, mean (ms/image) | p95 (ms) | Hardware / precision |", "|---|---|---|---|---|"]
    for r in full:
        e = r.get("efficiency")
        if e and (r["meta"]["name"] in dict((n, l) for l, n in PRIMARY) or r["meta"]["family"] == "detector"):
            L.append(f"| {LABELS.get(r['meta']['name'], dict((n, l) for l, n in PRIMARY).get(r['meta']['name']))} | "
                     f"{_num(e['model_size_mb'], '{:.1f}')} | {_num(e['latency_ms_mean'], '{:.1f}')} | "
                     f"{_num(e['latency_ms_p95'], '{:.1f}')} | {r['meta'].get('device', '–')} |")
    h, d, c = by_name.get("hybrid_full"), by_name.get("convnext_base_448_detcrop_calibrated"), by_name.get("classical_svm_full")
    if h and d and c and h["meta"].get("fusion_w", 0) > 0 and "efficiency" in h:
        gate = h["meta"]["fusion_gate"]
        cov = next((r["coverage"] for r in d["classification"]["coverage_curve"] if abs(r["threshold"] - gate) < 1e-9), None)
        if cov is not None:
            frac = 1 - cov
            dl, cl = by_name["convnext_base_448_detcrop"]["efficiency"]["latency_ms_mean"], c["efficiency"]["latency_ms_mean"]
            L.append(f"\nHybrid latency in practice: the classical branch is only needed when the calibrated deep confidence is "
                     f"below {gate} ({100 * frac:.1f}% of test images), so the expected cost is {dl:.1f} + {100 * frac:.1f}% × "
                     f"{cl:.0f} ≈ **{dl + frac * cl:.1f} ms/image** (the row above assumes it runs on every image).")
    L += ["", "Batch size 1. Deep/detector: RTX 4070 (GPU, bf16/fp16), including pre- and post-processing. "
          "Classical: feature extraction (GrabCut dominates) measured during parallel extraction, 1 core per image."]
    if opt:
        L += ["", "**Optimization (deployed configuration): ONNX export + INT8 dynamic quantization, CPU with 2 threads**",
              "", "| Variant | Size (MB) | Latency (ms/image) | Top-1 | Top-5 | n |", "|---|---|---|---|---|---|"]
        for o in opt:
            if o.get("model") == "detector":
                L.append(f"| Detector, {o['variant']} | {o['size_mb']:.1f} | {o['latency_ms']:.1f} | – | – | – |")
            else:
                L.append(f"| Classifier, {o['variant']} | {o['size_mb']:.1f} | {o['latency_ms']:.1f} | "
                         f"{_pct(o['top1'])} | {_pct(o['top5'])} | {o['n']} |")
        cl = {o["variant"]: o for o in opt if o.get("model") != "detector"}
        if "pytorch_fp32" in cl and "onnx_int8_dynamic" in cl:
            a, b = cl["pytorch_fp32"], cl["onnx_int8_dynamic"]
            L.append(f"\nINT8: **{a['size_mb'] / b['size_mb']:.1f}× smaller, {a['latency_ms'] / b['latency_ms']:.1f}× faster** "
                     f"on CPU, top-1 change {100 * (b['top1'] - a['top1']):+.1f} pts on the same {b['n']} test crops.")

    # 5 ---------------------------------------------------------------------
    L += ["", "## 5. Robustness (performance under varying conditions)", ""]
    if rob:
        base = {r["condition"]: r for r in rob["classifier"] if not r["clahe"]}
        clean = base.get("clean", {}).get("top1")
        L += [f"Classifier on a fixed random subset of **{rob['n_images']} test images**; each detector crop is corrupted, "
              "then classified.", "", "| Condition | Top-1 | Δ vs clean | Top-5 | Mean confidence |", "|---|---|---|---|---|"]
        for c, r in base.items():
            d = "–" if clean is None or c == "clean" else f"{100 * (r['top1'] - clean):+.2f} pts"
            L.append(f"| {c} | {_pct(r['top1'])} | {d} | {_pct(r['top5'])} | {r['mean_conf']:.3f} |")
        cl = [r for r in rob["classifier"] if r["clahe"]]
        if cl:
            L += ["", "Hybrid preprocessing: classical CLAHE + auto-gamma applied before the CNN:", "",
                  "| Condition | CNN only | + CLAHE/gamma | Δ |", "|---|---|---|---|"]
            L += [f"| {r['condition']} | {_pct(base[r['condition']]['top1'])} | {_pct(r['top1'])} | "
                  f"{100 * (r['top1'] - base[r['condition']]['top1']):+.2f} pts |" for r in cl if r["condition"] in base]
        if rob.get("pose_buckets_clean"):
            L += ["", "Viewing angle (clean images, bucketed by detected-box aspect ratio):", "",
                  "| View | n | Top-1 |", "|---|---|---|"]
            L += [f"| {b['bucket']} | {b['n']} | {_pct(b['top1'])} |" for b in rob["pose_buckets_clean"]]
        if rob.get("detector"):
            L += ["", f"Detector under whole-image corruptions ({rob['detector'][0]['n']} test images):", "",
                  "| Condition | Mean IoU | IoU ≥ 0.5 |", "|---|---|---|"]
            L += [f"| {d['condition']} | {d['mean_iou']:.4f} | {_pct(d['iou_ge_0.5'])} |" for d in rob["detector"]]
    else:
        L.append("_Robustness results not available._")

    # 6 ---------------------------------------------------------------------
    L += ["", "## 6. Confidence and threshold analysis", ""]
    rows = [("Deep, raw softmax", "convnext_base_448_detcrop"), ("Deep + temperature scaling", "convnext_base_448_detcrop_calibrated"),
            ("Hybrid (deployed)", "hybrid_full")]
    L += ["| Model | Mean confidence | Conf. when correct | Conf. when wrong | ECE (15 bins) |", "|---|---|---|---|---|"]
    for lab, name in rows:
        if name in by_name:
            c = by_name[name]["classification"]
            L.append(f"| {lab} | {c['mean_confidence']:.3f} | {_num(c['mean_confidence_correct'], '{:.3f}')} | "
                     f"{_num(c['mean_confidence_wrong'], '{:.3f}')} | {c['ece']:.4f} |")
    if hyb:
        L.append(f"\nTemperature T = {hyb['temperature']:.3f} was fitted on the validation split (raw model is "
                 "under-confident because of label smoothing and CutMix/MixUp).")
    if "hybrid_full" in by_name:
        L += ["", "Accept a prediction only if its calibrated confidence ≥ τ (hybrid):", "",
              "| τ | Coverage (answered) | Accuracy on answered | Error rate on rejected |", "|---|---|---|---|"]
        L += [f"| {row['threshold']:.2f} | {_pct(row['coverage'])} | {_pct(row['accuracy'])} | {_pct(row['rejected_error_rate'])} |"
              for row in by_name["hybrid_full"]["classification"]["coverage_curve"]]
    if hyb:
        p = hyb["policy_test"]
        pol = p["policy"]
        L += ["", f"**Low-confidence handling (deployed policy, thresholds chosen on validation: τ_model = {pol['tau_class']}, "
              f"τ_make = {pol['tau_make']})**", "", "| Answer given | Share of test images | Accuracy |", "|---|---|---|",
              f"| Make + model + year | {_pct(p['model']['coverage'])} | {_pct(p['model']['accuracy'])} |",
              f"| Manufacturer only | {_pct(p['make']['coverage'])} | {_pct(p['make']['accuracy'])} (make) |",
              f"| \"Uncertain\" + top-5 shown | {_pct(p['uncertain']['coverage'])} | true class in top-5: "
              f"{_pct(p['uncertain']['top5_contains_true'])} |", "",
              "Policy sweep (τ_make fixed at 0.9):", "",
              "| τ_model | Full answer: share / accuracy | Make only: share / accuracy | Uncertain share |", "|---|---|---|---|"]
        for s in hyb["policy_sweep_test"]:
            L.append(f"| {s['policy']['tau_class']} | {_pct(s['model']['coverage'])} / {_pct(s['model']['accuracy'])} | "
                     f"{_pct(s['make']['coverage'])} / {_pct(s['make']['accuracy'])} | {_pct(s['uncertain']['coverage'])} |")
    L += ["", "Plots: `results/hybrid/reliability_coverage.png` (reliability diagram, accuracy vs coverage), "
          "`results/robustness/robustness.png`. Ablations: `docs/ablation_study.md`. Failure cases: `docs/failure_analysis.md`."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="Team ID, e.g. T01")
    ap.add_argument("--usns", required=True, help="Comma-separated USNs")
    args = ap.parse_args()
    usns = [u.strip() for u in args.usns.split(",") if u.strip()]
    out = config.PROJECT_ROOT / f"eval_P5_{args.team}.md"
    out.write_text(render(load_results(), args.team, usns), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
