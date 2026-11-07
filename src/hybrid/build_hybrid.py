"""Build the hybrid predictor from saved deep + classical predictions.

    python -m src.hybrid.build_hybrid

Everything is selected on the VALIDATION split, then applied once to test:
  1. Temperature scaling of the deep logits (fit T on val NLL).
  2. Score fusion: log p_hybrid = log p_deep(T) + w * gate * log p_classical,
     gate = 1 for all images, or only when deep confidence < g. (w, g) chosen on val NLL
     (ties -> smaller w, i.e. prefer the simpler model).
  3. Low-confidence policy: model-level answer if conf >= tau_class, else make-level if
     make conf >= tau_make, else "uncertain" + top-5. tau_class is chosen on val as the
     smallest threshold whose accepted-set accuracy reaches --target-acc.

Writes results/preds/hybrid_*.npz (test + val), results/hybrid/hybrid_summary.json,
models/hybrid_config.json (used by the app) and reliability / coverage plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.eval import metrics as M
from src.eval.evaluate import PREDS_DIR, save_predictions
from src.hybrid.confidence import (Policy, calibrated_probs, evaluate_policy, fit_temperature,
                                   make_probs, sweep_policy)

OUT_DIR = config.RESULTS_DIR / "hybrid"


def load_preds(name: str) -> dict:
    d = np.load(PREDS_DIR / f"{name}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files} | {"meta": json.loads(str(d["meta"]))}


def labels_for(filenames: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.read_csv(config.INDEX_CSV)
    idx = idx[idx["split"] == split].set_index("filename").loc[filenames]
    return idx["class_id"].to_numpy(), idx["make_id"].to_numpy()


def class_to_make() -> tuple[np.ndarray, int]:
    cls = pd.read_csv(config.CLASSES_CSV).sort_values("class_id")
    return cls["make_id"].to_numpy(), int(cls["make_id"].max()) + 1


def fuse(p_deep: np.ndarray, p_cls: np.ndarray, w: float, gate: float) -> np.ndarray:
    if w == 0:
        return p_deep
    g = (p_deep.max(1) < gate)[:, None] if gate < 1 else np.ones((len(p_deep), 1), bool)
    logp = np.log(np.clip(p_deep, 1e-12, 1)) + w * g * np.log(np.clip(p_cls, 1e-6, 1))
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def plot_reliability(results: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    for name, (p, y) in results.items():
        ece, bins = M.expected_calibration_error(p, y)
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", ms=3,
                ls="--" if name == "hybrid" else "-", label=f"{name} (ECE {ece:.3f})")
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability diagram (test)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, (p, y) in results.items():
        conf, ok = p.max(1), p.argmax(1) == y
        order = np.argsort(-conf)
        cov = np.arange(1, len(p) + 1) / len(p)
        ax.plot(cov, np.cumsum(ok[order]) / np.arange(1, len(p) + 1), ls="--" if name == "hybrid" else "-", label=name)
    ax.set(xlabel="coverage (fraction of images answered)", ylabel="accuracy on answered",
           title="Accuracy vs coverage (test)", ylim=(0.9, 1.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep-test", default="convnext_base_448_detcrop")
    ap.add_argument("--deep-val", default="convnext_base_448_gtcrop_val")
    ap.add_argument("--classical-test", default="classical_svm_full")
    ap.add_argument("--classical-val", default="classical_svm_full_val")
    ap.add_argument("--target-acc", type=float, default=0.97, help="accuracy required at model level (val)")
    ap.add_argument("--tau-make", type=float, default=0.9)
    ap.add_argument("--name", default="hybrid_full")
    ap.add_argument("--tag", default="", help="suffix for summary/config/plot files, e.g. _ensemble, "
                                              "so experiments never overwrite the deployed hybrid_config.json")
    args = ap.parse_args()

    dv, dt = load_preds(args.deep_val), load_preds(args.deep_test)
    cv, ct = load_preds(args.classical_val), load_preds(args.classical_test)
    for a, b, s in [(dv, cv, "val"), (dt, ct, "test")]:
        if not np.array_equal(a["filenames"], b["filenames"]):
            raise SystemExit(f"deep and classical {s} predictions are not aligned (different image order)")
    yv, ymv = labels_for(dv["filenames"], "val")
    yt, ymt = labels_for(dt["filenames"], "test")
    c2m, n_makes = class_to_make()

    # 1. temperature scaling on val
    T = fit_temperature(dv["logits"], yv)
    pdv, pdt = calibrated_probs(dv["logits"], T), calibrated_probs(dt["logits"], T)

    # 2. fusion weight + gate on val
    grid = []
    for gate in (1.0, 0.7, 0.5):
        for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            if w == 0 and gate < 1:
                continue
            pf = fuse(pdv, cv["probs"], w, gate)
            grid.append({"w": w, "gate": gate, "val_top1": float((pf.argmax(1) == yv).mean()),
                         "val_nll": float(-np.log(np.clip(pf[np.arange(len(yv)), yv], 1e-12, 1)).mean())})
    # select on val NLL (a proper scoring rule): top-1 on 1,222 images moves by noise alone
    best = min(grid, key=lambda g: (round(g["val_nll"], 4), g["w"]))
    w, gate = best["w"], best["gate"]
    phv, pht = fuse(pdv, cv["probs"], w, gate), fuse(pdt, ct["probs"], w, gate)

    # 3. confidence policy: smallest tau whose val accepted-set accuracy >= target
    mpv, mpt = make_probs(phv, c2m, n_makes), make_probs(pht, c2m, n_makes)
    tau = 0.95
    for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
        ok = phv.max(1) >= t
        if ok.any() and (phv.argmax(1)[ok] == yv[ok]).mean() >= args.target_acc:
            tau = float(t)
            break
    policy = Policy(tau_class=tau, tau_make=args.tau_make)

    # latency: deep path, plus classical feature extraction only if fusion is used
    lat = dt["latency_ms"] if "latency_ms" in dt else None
    if lat is not None and w > 0 and "latency_ms" in ct:
        lat = lat.mean() + ct["latency_ms"]
    size = (dt["meta"].get("model_size_mb") or 0) + ((ct["meta"].get("model_size_mb") or 0) if w > 0 else 0)
    meta = dict(family="hybrid", model_size_mb=size, device=dt["meta"].get("device"), temperature=T,
                fusion_w=w, fusion_gate=gate, tau_class=tau, tau_make=args.tau_make,
                components=[args.deep_test] + ([args.classical_test] if w > 0 else []))
    save_predictions(args.name, dt["filenames"], split="test", probs=pht, latency_ms=lat, **meta)
    save_predictions(f"{args.name}_val", dv["filenames"], split="val", probs=phv, **meta)
    # ablation rows: calibration only (no fusion) and fusion without calibration
    save_predictions(f"{args.deep_test}_calibrated", dt["filenames"], split="test", probs=pdt,
                     latency_ms=dt.get("latency_ms"), **{**meta, "family": "deep", "fusion_w": 0.0,
                                                          "components": [args.deep_test]})

    raw_t = calibrated_probs(dt["logits"], 1.0)
    summary = {
        "temperature": T,
        "fusion_grid_val": grid,
        "fusion_selected": best,
        "test_top1": {"deep_raw": float((raw_t.argmax(1) == yt).mean()),
                      "deep_calibrated": float((pdt.argmax(1) == yt).mean()),
                      "hybrid": float((pht.argmax(1) == yt).mean()),
                      "classical": float((ct["probs"].argmax(1) == yt).mean())},
        "test_ece": {"deep_raw": M.expected_calibration_error(raw_t, yt)[0],
                     "deep_calibrated": M.expected_calibration_error(pdt, yt)[0],
                     "hybrid": M.expected_calibration_error(pht, yt)[0]},
        "policy_selected_on_val": policy.to_dict(),
        "policy_test": evaluate_policy(policy, pht, mpt, yt, ymt),
        "policy_sweep_test": sweep_policy(pht, mpt, yt, ymt, tau_make=args.tau_make),
        "policy_val": evaluate_policy(policy, phv, mpv, yv, ymv),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"hybrid_summary{args.tag}.json").write_text(json.dumps(summary, indent=2))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / f"hybrid_config{args.tag}.json").write_text(json.dumps(
        {"temperature": T, "fusion_w": w, "fusion_gate": gate, **policy.to_dict(),
         "deep": dt["meta"].get("ckpt"), "classical": ct["meta"].get("name") if w > 0 else None}, indent=2))
    plot_reliability({"deep raw": (raw_t, yt), f"deep + temperature (T={T:.2f})": (pdt, yt),
                      "hybrid": (pht, yt)}, OUT_DIR / f"reliability_coverage{args.tag}.png")

    s = summary
    print(f"Temperature T = {T:.3f} | fusion selected on val: w={w}, gate={gate} "
          f"(val NLL {best['val_nll']:.4f}, val top-1 {best['val_top1']:.4f})")
    print("  val grid: " + "; ".join(f"w={g['w']},g={g['gate']}: nll {g['val_nll']:.4f} top1 {g['val_top1']:.4f}" for g in grid))
    print("TEST top-1: " + ", ".join(f"{k} {v:.4f}" for k, v in s["test_top1"].items()))
    print("TEST ECE:   " + ", ".join(f"{k} {v:.4f}" for k, v in s["test_ece"].items()))
    pt = s["policy_test"]
    print(f"Policy (tau_class={tau}, tau_make={args.tau_make}) on TEST: "
          f"model-level {pt['model']['coverage']:.1%} @ {pt['model']['accuracy']:.2%} | "
          f"make-only {pt['make']['coverage']:.1%} @ {(pt['make']['accuracy'] or 0):.2%} | "
          f"uncertain {pt['uncertain']['coverage']:.1%} (true class in top-5: "
          f"{(pt['uncertain']['top5_contains_true'] or 0):.1%})")
    print(f"Wrote {OUT_DIR / f'hybrid_summary{args.tag}.json'}, {OUT_DIR / f'reliability_coverage{args.tag}.png'}, "
          f"{config.MODELS_DIR / f'hybrid_config{args.tag}.json'}")


if __name__ == "__main__":
    main()
