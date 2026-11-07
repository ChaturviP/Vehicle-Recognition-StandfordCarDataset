"""Build data/processed/index.csv and classes.csv from the Kaggle Stanford Cars mirror.

Usage (from the project root):
    python -m src.data.prepare --root "C:/Users/win11/Downloads/archive (10)"

index.csv columns:
    split (train|val|test), filename, path, x1, y1, x2, y2, width, height,
    class_id (0-based), class_name, make, model, year, body_type, make_id
The validation split is a stratified 15% of the official train split; the
official test split is kept untouched for final evaluation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config
from src.data.labels import folder_name, parse_class_name

ANNO_COLS = ["filename", "x1", "y1", "x2", "y2", "class_id"]


def load_classes(root: Path) -> pd.DataFrame:
    names = [line.strip() for line in (root / "names.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != config.NUM_CLASSES:
        raise ValueError(f"Expected {config.NUM_CLASSES} class names, found {len(names)}")
    labels = [parse_class_name(n, i) for i, n in enumerate(names)]
    df = pd.DataFrame([vars(l) for l in labels]).rename(columns={"name": "class_name"})
    makes = sorted(df["make"].unique())
    df["make_id"] = df["make"].map({m: i for i, m in enumerate(makes)})
    return df


def _find_image_dir(root: Path) -> Path:
    for candidate in (root / "car_data" / "car_data", root / "car_data", root):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find train/ and test/ image folders under {root}")


def load_split(root: Path, image_dir: Path, split: str, classes: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(root / f"anno_{split}.csv", header=None, names=ANNO_COLS)
    df["class_id"] -= 1  # CSV is 1-based
    df = df.merge(classes, on="class_id", how="left", validate="many_to_one")

    split_dir = image_dir / split
    paths = [split_dir / folder_name(c) / f for c, f in zip(df["class_name"], df["filename"])]
    missing = [p for p in paths if not p.exists()]
    if missing:
        # Fall back to a filename lookup in case folder names differ in this mirror.
        lookup = {p.name: p for p in split_dir.rglob("*.jpg")}
        paths = [p if p.exists() else lookup.get(p.name) for p in paths]
        still_missing = [str(p) for p, f in zip(paths, df["filename"]) if p is None]
        if still_missing:
            raise FileNotFoundError(f"{len(still_missing)} {split} images not found, e.g. {still_missing[:3]}")
    df["path"] = [str(p.resolve()) for p in paths]
    df["split"] = split
    return df


def add_image_sizes(df: pd.DataFrame) -> pd.DataFrame:
    sizes = []
    for p in df["path"]:
        with Image.open(p) as im:  # reads the header only
            sizes.append(im.size)
    df["width"], df["height"] = zip(*sizes)
    # Clamp boxes to image bounds (a few annotations touch or exceed the edge).
    df["x1"] = df["x1"].clip(lower=0)
    df["y1"] = df["y1"].clip(lower=0)
    df["x2"] = df[["x2", "width"]].min(axis=1)
    df["y2"] = df[["y2", "height"]].min(axis=1)
    return df


def assign_val_split(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    train_idx = df.index[df["split"] == "train"]
    _, val_idx = train_test_split(
        train_idx, test_size=frac, random_state=seed, stratify=df.loc[train_idx, "class_id"]
    )
    df.loc[val_idx, "split"] = "val"
    return df


def build_index(root: Path, val_frac: float = config.VAL_FRACTION, seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    classes = load_classes(root)
    image_dir = _find_image_dir(root)
    df = pd.concat([load_split(root, image_dir, s, classes) for s in ("train", "test")], ignore_index=True)
    df = add_image_sizes(df)
    df = assign_val_split(df, val_frac, seed)
    cols = ["split", "filename", "path", "x1", "y1", "x2", "y2", "width", "height",
            "class_id", "class_name", "make", "model", "year", "body_type", "make_id"]
    return df[cols], classes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=config.DATASET_ROOT, help="Kaggle mirror folder")
    ap.add_argument("--val-frac", type=float, default=config.VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    df, classes = build_index(args.root, args.val_frac, args.seed)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INDEX_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    classes.to_csv(config.CLASSES_CSV, index=False)

    print(f"Wrote {config.INDEX_CSV} ({len(df)} rows) and {config.CLASSES_CSV}")
    print(df["split"].value_counts().to_string())
    print(f"classes={df['class_id'].nunique()} makes={classes['make'].nunique()} "
          f"body_types={sorted(classes['body_type'].unique())}")


if __name__ == "__main__":
    main()
