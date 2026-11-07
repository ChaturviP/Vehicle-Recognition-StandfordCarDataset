"""Central paths and constants. Override the dataset location with the
STANFORD_CARS_ROOT environment variable or the --root CLI flag."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Kaggle "stanford-car-dataset-by-classes-folder" mirror layout:
#   <root>/anno_train.csv, anno_test.csv, names.csv
#   <root>/car_data/car_data/{train,test}/<class folder>/<file>.jpg
DATASET_ROOT = Path(os.environ.get("STANFORD_CARS_ROOT", PROJECT_ROOT / "data" / "raw"))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_CSV = PROCESSED_DIR / "index.csv"
CLASSES_CSV = PROCESSED_DIR / "classes.csv"
YOLO_DIR = DATA_DIR / "yolo"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 196
SEED = 42
VAL_FRACTION = 0.15
CROP_PAD = 0.12  # fraction of bbox size added on each side before cropping

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
