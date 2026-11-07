"""Export the index to Ultralytics YOLO format (single class "car").

    python -m src.detect.prepare_yolo

Creates data/yolo/{images,labels}/{train,val,test} and data/yolo/cars.yaml.
Images are hard-linked when possible (same drive, no extra disk use) and
copied otherwise. File names are prefixed with the split to avoid clashes
(the train and test splits both contain 00001.jpg, ...).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src import config
from src.data.dataset import load_index


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export(out_dir: Path = config.YOLO_DIR) -> Path:
    df = load_index()
    for split, part in df.groupby("split"):
        img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in part.itertuples():
            stem = f"{split}_{Path(r.filename).stem}"
            _link_or_copy(Path(r.path), img_dir / f"{stem}.jpg")
            cx = (r.x1 + r.x2) / 2 / r.width
            cy = (r.y1 + r.y2) / 2 / r.height
            w = (r.x2 - r.x1) / r.width
            h = (r.y2 - r.y1) / r.height
            (lbl_dir / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        print(f"{split}: {len(part)} images")

    yaml_path = out_dir / "cars.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: car\n"
    )
    print(f"Wrote {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.YOLO_DIR)
    export(ap.parse_args().out)
