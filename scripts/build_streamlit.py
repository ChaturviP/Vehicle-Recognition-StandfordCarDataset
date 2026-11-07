"""Assemble a self-contained folder for Streamlit Community Cloud.

    python scripts/build_streamlit.py            # -> ./streamlit_build

Push that folder to its own (new, empty) GitHub repository, then deploy it at
share.streamlit.io with main file `streamlit_app.py` and Python 3.12. It contains only
what the app needs (ONNX models, SVM, minimal source, assets) with light, CPU-only
requirements; every file is under GitHub's 100 MB limit.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_space import CODE, METRICS, MODELS, PLOTS, ROOT, copy  # noqa: E402

OUT = ROOT / "streamlit_build"

REQUIREMENTS = """\
streamlit>=1.50
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

CONFIG = (ROOT / ".streamlit" / "config.toml").read_text()

README = """\
# Fine-Grained Vehicle Recognition: live demo (Hackathon P5)

Streamlit app for the P5 vehicle recognition system: YOLO11s detects the car, a ConvNeXt-Base
(INT8 ONNX, CPU) and a classical computer-vision branch (CLAHE, GrabCut, Canny/contours,
HSV/LBP/GLCM/HOG features + SVM) identify make, model and year among 196 Stanford Cars classes,
with calibrated low-confidence handling.

Full source code, training, evaluation and documentation:
https://github.com/ChaturviP/Vehicle-Recognition-StandfordCarDataset

Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.
"""

GITIGNORE = "__pycache__/\n*.pyc\n.streamlit/secrets.toml\n"


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    missing = [p for p in CODE + MODELS if not (ROOT / p).exists()]
    if not (ROOT / "data/processed/classes.csv").exists():
        missing.append("data/processed/classes.csv")
    if missing:
        sys.exit("Missing files (run the earlier steps first):\n  " + "\n  ".join(missing))

    for name in ("streamlit_app.py", "inference.py", "content.py", "ui.py"):
        copy(ROOT / "app" / name, OUT / name)
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
    for p in sorted((ROOT / "app/static").glob("*")):      # hero texture, backdrop, motion script
        copy(p, OUT / "static" / p.name)
    if (ROOT / "docs/architecture.png").exists():
        copy(ROOT / "docs/architecture.png", OUT / "assets/architecture.png")
    ex = ROOT / "app/examples"
    if ex.exists():
        for p in sorted(ex.glob("*.jpg"))[:8] + sorted(ex.glob("*.md")):
            copy(p, OUT / "examples" / p.name)

    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / ".streamlit").mkdir(exist_ok=True)
    (OUT / ".streamlit/config.toml").write_text(CONFIG)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitignore").write_text(GITIGNORE)

    big = [(f, f.stat().st_size / 2**20) for f in OUT.rglob("*") if f.is_file() and f.stat().st_size > 95 * 2**20]
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**20
    print(f"Built {OUT} ({size:.0f} MB).")
    for f, mb in big:
        print(f"  WARNING: {f.name} is {mb:.1f} MB; GitHub rejects files over 100 MB")
    print("Test locally:  streamlit run streamlit_build/streamlit_app.py")
    return OUT


if __name__ == "__main__":
    build()
