"""Train the classical-feature classifier on cached features (train), tune on val,
score on test, and save predictions for the harness.

    python -m src.classical.train_classical                       # all groups, RBF SVM
    python -m src.classical.train_classical --model logreg        # fast linear baseline
    python -m src.classical.train_classical --groups hog          # feature-group ablation
    python -m src.classical.train_classical --groups color,lbp,glcm,edge,shape,geom --name classical_svm_nohog
    python -m src.classical.train_classical --variant nogc        # GrabCut ablation (needs those caches)

Pipeline: per-feature standardisation -> PCA (whitened) -> classifier. The PCA size
and C are chosen on val. For ablation runs pass single values (e.g. --pca 256 --C 10)
to skip the search. Saves models/<name>.joblib and results/preds/<name>.npz
(+ <name>_val.npz for fusion).
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.classical.extract_features import cache_path
from src.classical.features import GROUPS
from src.eval.evaluate import save_predictions


def load_split(variant: str, split: str, groups: list[str]):
    d = np.load(cache_path(variant, split), allow_pickle=False)
    X = np.concatenate([d[g].astype(np.float32) for g in groups], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, d["class_id"], d["filenames"], d["extract_ms"]


def make_model(kind: str, n_components: int, C: float, seed: int):
    if kind == "svm":
        # Platt-scaled probabilities (replacement for the deprecated SVC(probability=True))
        clf = CalibratedClassifierCV(SVC(kernel="rbf", C=C, gamma="scale", random_state=seed),
                                     method="sigmoid", cv=3, ensemble=False)
    elif kind == "logreg":
        clf = LogisticRegression(C=C, max_iter=2000)
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.5, random_state=seed, verbose=-1)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, whiten=True, random_state=seed), clf)


def full_probs(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), config.NUM_CLASSES), np.float32)
    out[:, model.classes_] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="full", help="feature cache variant from extract_features")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--model", default="svm", choices=["svm", "logreg", "lgbm"])
    ap.add_argument("--pca", default="128,256", help="PCA sizes to try (comma-separated)")
    ap.add_argument("--C", default="1,3,10", help="C values to try (comma-separated)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    name = args.name or f"classical_{args.model}_{args.variant}" + \
        ("" if set(groups) == set(GROUPS) else "_" + "+".join(groups))
    Xtr, ytr, _, _ = load_split(args.variant, "train", groups)
    Xva, yva, fva, _ = load_split(args.variant, "val", groups)
    Xte, yte, fte, ext_ms = load_split(args.variant, "test", groups)
    print(f"[{name}] features: {Xtr.shape[1]} dims from {groups}")

    best = None
    Cs = [float(c) for c in args.C.split(",")] if args.model != "lgbm" else [1.0]
    for nc in [int(n) for n in args.pca.split(",")]:
        nc = min(nc, Xtr.shape[1], len(Xtr) - 1)
        for C in Cs:
            t0 = time.time()
            m = make_model(args.model, nc, C, config.SEED).fit(Xtr, ytr)
            acc = (m.predict(Xva) == yva).mean()
            print(f"  pca={nc:<4} C={C:<5} val top-1 {acc:.4f}  ({time.time() - t0:.0f}s)")
            if best is None or acc > best[0]:
                best = (acc, nc, C, m)
    acc, nc, C, model = best
    print(f"Selected pca={nc} C={C} (val top-1 {acc:.4f})")

    pva, pte = full_probs(model, Xva), full_probs(model, Xte)
    # per-image classifier latency (batch 1) on top of the measured feature-extraction time
    t = []
    for i in range(min(200, len(Xte))):
        t0 = time.perf_counter()
        model.predict_proba(Xte[i:i + 1])
        t.append((time.perf_counter() - t0) * 1000)
    latency = ext_ms + float(np.mean(t))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "groups": groups, "variant": args.variant, "pca": nc, "C": C}, path)
    size_mb = path.stat().st_size / 2**20

    meta = dict(family="classical", model_size_mb=size_mb, device="CPU (feature extraction 1 core)",
                classifier=args.model, groups=groups, variant=args.variant, pca=nc, C=C)
    save_predictions(f"{name}_val", fva, split="val", probs=pva, **meta)
    out = save_predictions(name, fte, split="test", probs=pte, latency_ms=latency, **meta)
    top1 = (pte.argmax(1) == yte).mean()
    top5 = (np.argsort(-pte, 1)[:, :5] == yte[:, None]).any(1).mean()
    print(f"TEST top-1 {top1:.4f}  top-5 {top5:.4f}  | {latency.mean():.0f} ms/img | {size_mb:.1f} MB\nSaved {out}")


if __name__ == "__main__":
    main()
