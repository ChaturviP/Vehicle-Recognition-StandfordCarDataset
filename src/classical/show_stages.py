"""Render the classical pipeline's intermediate stages as one figure.

    python -m src.classical.show_stages --split test --n 4          # first 4 test images (GT boxes)
    python -m src.classical.show_stages --image path/to/car.jpg --box 10 20 400 260

Writes results/classical/stages_<name>.png (used in the design doc and app).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.classical.preprocess import run_pipeline, visualize
from src.data.dataset import load_index


def render(items: list[tuple[str, np.ndarray, tuple]], out: Path) -> Path:
    stages = [visualize(run_pipeline(img, box)) for _, img, box in items]
    cols = len(stages[0])
    fig, axes = plt.subplots(len(stages), cols, figsize=(2.6 * cols, 1.6 * len(stages)), squeeze=False)
    for r, (st, (title, _, _)) in enumerate(zip(stages, items)):
        for c, (name, im) in enumerate(st.items()):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(title, fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.image:
        img = cv2.imdecode(np.fromfile(str(args.image), np.uint8), cv2.IMREAD_COLOR)
        box = args.box or (0, 0, img.shape[1], img.shape[0])
        items, name = [(args.image.stem, img, box)], args.name or args.image.stem
    else:
        df = load_index(args.split).sample(args.n, random_state=config.SEED)
        items = [(r.class_name[:28], cv2.imdecode(np.fromfile(r.path, np.uint8), cv2.IMREAD_COLOR),
                  (r.x1, r.y1, r.x2, r.y2)) for r in df.itertuples()]
        name = args.name or f"{args.split}_samples"
    print(f"Wrote {render(items, config.RESULTS_DIR / 'classical' / f'stages_{name}.png')}")


if __name__ == "__main__":
    main()
