"""PyTorch dataset over data/processed/index.csv with optional bbox cropping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from src import config

try:
    from torch.utils.data import Dataset
except ImportError:  # lets label/index code run without torch installed
    Dataset = object  # type: ignore[misc,assignment]


def load_index(splits: Sequence[str] | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.INDEX_CSV)
    if splits is not None:
        splits = [splits] if isinstance(splits, str) else list(splits)
        df = df[df["split"].isin(splits)]
    return df.reset_index(drop=True)


def load_det_boxes(path: Path, df: pd.DataFrame) -> np.ndarray:
    """Best detector box per row of `df` from a detector predictions npz (same split)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    boxes = dict(zip(data["filenames"], data["boxes"]))
    if meta["split"] != df["split"].iloc[0]:
        raise SystemExit(f"Detector preds are for split {meta['split']!r}, not {df['split'].iloc[0]!r}")
    return np.stack([boxes[f] for f in df["filename"]])


def padded_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
               pad: float = config.CROP_PAD) -> tuple[int, int, int, int]:
    """Expand a box by `pad` x its size on each side, clamped to the image."""
    bw, bh = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - pad * bw)),
        int(max(0, y1 - pad * bh)),
        int(min(width, x2 + pad * bw)),
        int(min(height, y2 + pad * bh)),
    )


def crop_to_box(img: Image.Image, box: Sequence[float], pad: float = config.CROP_PAD) -> Image.Image:
    x1, y1, x2, y2 = box
    return img.crop(padded_box(x1, y1, x2, y2, img.width, img.height, pad))


class CarsDataset(Dataset):
    """Returns (image, class_id, make_id, row_index).

    crop: "gt" crops to the annotated box, "none" uses the full image, or pass a
    DataFrame-aligned list of predicted boxes via `boxes` to crop on detections.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None,
                 crop: str = "gt", pad: float = config.CROP_PAD,
                 boxes: Optional[Sequence[Sequence[float]]] = None):
        if crop not in {"gt", "none", "pred"}:
            raise ValueError("crop must be 'gt', 'none' or 'pred'")
        if crop == "pred" and (boxes is None or len(boxes) != len(df)):
            raise ValueError("crop='pred' needs one predicted box per row")
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.crop = crop
        self.pad = pad
        self.boxes = boxes

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, i: int) -> Image.Image:
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.crop == "gt":
            img = crop_to_box(img, (row.x1, row.y1, row.x2, row.y2), self.pad)
        elif self.crop == "pred":
            img = crop_to_box(img, self.boxes[i], self.pad)
        return img

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        row = self.df.iloc[i]
        return img, int(row["class_id"]), int(row["make_id"]), i
