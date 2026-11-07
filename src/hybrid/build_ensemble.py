"""Average the probabilities of several trained deep models (equal weights, no tuning).

    python -m src.hybrid.build_ensemble

Default members: the crop model (YOLO crops) and the full-image model, which make largely
different mistakes. For each split the member predictions are aligned by filename and their
softmax probabilities averaged; "logits" = log(mean prob), so build_hybrid can temperature-scale
the ensemble exactly like a single model. Prints val and test top-1 for every member and the
ensemble: adopt the ensemble only if it also wins on VAL.

Writes results/preds/<name>.npz (test) and <name>_val.npz (val), family "deep".
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.eval.evaluate import PREDS_DIR, save_predictions
from src.hybrid.build_hybrid import labels_for

DEFAULT = "convnext_base_448_detcrop:convnext_base_448_gtcrop_val,convnext_fullimg_nonecrop:convnext_fullimg_nonecrop_val"


def _probs(d) -> np.ndarray:
    if "logits" in d:
        z = d["logits"].astype(np.float64)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    return d["probs"].astype(np.float64)


def load(name):
    d = np.load(PREDS_DIR / f"{name}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files} | {"meta": json.loads(str(d["meta"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", default=DEFAULT, help="comma list of test_name:val_name")
    ap.add_argument("--name", default="ensemble_crop_full")
    args = ap.parse_args()
    members = [m.split(":") for m in args.members.split(",")]

    for split, idx in (("val", 1), ("test", 0)):
        preds = [load(m[idx]) for m in members]
        files = preds[0]["filenames"]
        y, _ = labels_for(files, split)
        probs = []
        for m, d in zip(members, preds):
            order = {f: i for i, f in enumerate(d["filenames"])}
            p = _probs(d)[[order[f] for f in files]]
            probs.append(p)
            print(f"{split:<5} {m[idx]:<36} top-1 {(p.argmax(1) == y).mean():.4f}")
        avg = np.mean(probs, 0)
        print(f"{split:<5} {'ENSEMBLE (' + str(len(probs)) + ' models, equal weights)':<36} "
              f"top-1 {(avg.argmax(1) == y).mean():.4f}")
        meta = dict(family="deep", members=[m[idx] for m in members], ensemble="mean of softmax probabilities")
        if split == "test":
            lat = [d["latency_ms"].mean() for d in preds if "latency_ms" in d]
            size = [d["meta"].get("model_size_mb") or 0 for d in preds]
            meta |= dict(model_size_mb=float(sum(size)), device=preds[0]["meta"].get("device"))
            out = save_predictions(args.name, files, split="test", probs=avg.astype(np.float32),
                                   latency_ms=np.full(len(files), float(sum(lat))) if lat else None,
                                   extra_arrays={"logits": np.log(np.clip(avg, 1e-12, 1)).astype(np.float32)}, **meta)
        else:
            out = save_predictions(f"{args.name}_val", files, split="val", probs=avg.astype(np.float32),
                                   extra_arrays={"logits": np.log(np.clip(avg, 1e-12, 1)).astype(np.float32)}, **meta)
        print(f"      saved {out}")


if __name__ == "__main__":
    main()
