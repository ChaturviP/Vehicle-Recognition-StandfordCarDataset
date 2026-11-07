"""Assemble the Hugging Face Space and (optionally) upload it.

    python scripts/build_space.py                                        # build ./space_build only
    python scripts/build_space.py --repo-id <hf-username>/fine-grained-vehicle-recognition --upload

Upload needs a Hugging Face account and a write token: run `hf auth login` once
(or set HF_TOKEN). The Space is created if it does not exist (Gradio SDK, free CPU).
Large model files are uploaded with Git LFS automatically by huggingface_hub.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_build"

CODE = ["src/__init__.py", "src/config.py",
        "src/classical/__init__.py", "src/classical/preprocess.py", "src/classical/features.py",
        "src/hybrid/__init__.py", "src/hybrid/confidence.py"]
MODELS = ["models/detector_yolo11s_cars.onnx", "models/convnext_base_448_int8.onnx",
          "models/classical_svm_full.joblib", "models/hybrid_config.json"]
METRICS = ["classical_svm_full", "convnext_base_448_detcrop", "hybrid_full",
           "detector_yolo11s_cars", "yolo11s_coco_zeroshot"]
PLOTS = {"results/hybrid/reliability_coverage.png": "reliability_coverage.png",
         "results/robustness/robustness.png": "robustness.png",
         "results/classical/stages_test_samples.png": "stages_test_samples.png"}

REQUIREMENTS = """\
numpy
pandas
pillow
scipy
scikit-learn==1.9.1
scikit-image==0.26.0
opencv-python-headless
onnxruntime==1.31.1
joblib
"""

README = """\
---
title: Fine-Grained Vehicle Recognition
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app.py
pinned: false
---

# Intelligent Fine-Grained Vehicle Recognition (Hackathon P5)

Detects a car with YOLO11s, then identifies its make, model and year among the 196 Stanford Cars
classes with a ConvNeXt-Base (INT8 ONNX, CPU) combined with a classical computer-vision branch
(CLAHE, GrabCut, Canny/contours, HSV/LBP/GLCM/HOG features + SVM) and calibrated low-confidence
handling. Source code: https://github.com/ChaturviP/Vehicle-Recognition-StandfordCarDataset
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    missing = [p for p in CODE + MODELS if not (ROOT / p).exists()]
    if not (ROOT / "data/processed/classes.csv").exists():
        missing.append("data/processed/classes.csv")
    if missing:
        sys.exit("Missing files (run the earlier steps first):\n  " + "\n  ".join(missing))

    copy(ROOT / "app/app.py", OUT / "app.py")
    copy(ROOT / "app/inference.py", OUT / "inference.py")
    for p in CODE + MODELS:
        copy(ROOT / p, OUT / p)
    copy(ROOT / "data/processed/classes.csv", OUT / "assets/classes.csv")
    for name in METRICS:
        p = ROOT / "results/metrics" / f"{name}.json"
        if p.exists():
            copy(p, OUT / "assets/metrics" / p.name)
    if (ROOT / "results/optimize/summary.json").exists():
        copy(ROOT / "results/optimize/summary.json", OUT / "assets/optimize_summary.json")
    for src, dst in PLOTS.items():
        if (ROOT / src).exists():
            copy(ROOT / src, OUT / "assets" / dst)
    ex = ROOT / "app/examples"
    if ex.exists():
        for p in sorted(ex.glob("*.jpg"))[:8]:
            copy(p, OUT / "examples" / p.name)

    import gradio
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / "README.md").write_text(README.format(gradio=gradio.__version__), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**20
    print(f"Built {OUT} ({size:.0f} MB). Test it locally with:  python space_build/app.py")
    return OUT


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy vehicle recognition app")
    print(f"Uploaded. Space: https://huggingface.co/spaces/{repo_id}  (first build takes a few minutes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", help="<hf-username>/<space-name>")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    build()
    if args.upload:
        if not args.repo_id:
            sys.exit("--upload needs --repo-id <hf-username>/<space-name>")
        upload(args.repo_id)
