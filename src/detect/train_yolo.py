"""Fine-tune an Ultralytics YOLO detector on Stanford Cars (single class "car").

    python -m src.detect.train_yolo                       # yolo11s, 30 epochs
    python -m src.detect.train_yolo --model yolo11m.pt --epochs 40 --batch 16

Needs data/yolo/cars.yaml from `python -m src.detect.prepare_yolo`.
The best checkpoint is copied to models/detector_<name>.pt.
"""
from __future__ import annotations

import argparse
import shutil

from src import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained COCO weights to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None, help="run name (default: <model>_cars)")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = config.YOLO_DIR / "cars.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run `python -m src.detect.prepare_yolo` first")

    name = args.name or f"{args.model.removesuffix('.pt')}_cars"
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        single_cls=True,
        cos_lr=True,
        patience=10,
        seed=config.SEED,
        deterministic=False,
        amp=True,
        project=str(config.PROJECT_ROOT / "runs" / "detect"),
        name=name,
        exist_ok=True,
        plots=True,
    )

    best = config.PROJECT_ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.MODELS_DIR / f"detector_{name}.pt"
    shutil.copy2(best, dst)
    print(f"Best weights -> {dst}")


if __name__ == "__main__":
    main()
