"""Calibration and low-confidence handling shared by the hybrid evaluation and the app."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising validation NLL of softmax(logits / T)."""
    logits = logits.astype(np.float64)
    nll = lambda T: -log_softmax(logits / T, 1)[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def calibrated_probs(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits.astype(np.float64) / T, 1)


def make_probs(probs: np.ndarray, class_to_make: np.ndarray, num_makes: int) -> np.ndarray:
    """Manufacturer probabilities = sum of the probabilities of that make's classes."""
    out = np.zeros((len(probs), num_makes))
    for m in range(num_makes):
        out[:, m] = probs[:, class_to_make == m].sum(1)
    return out


@dataclass
class Policy:
    """Decision rule for one image.

    conf(class) >= tau_class          -> "model": report make / model / year
    else conf(make) >= tau_make       -> "make":  report the manufacturer only
    else                              -> "uncertain": show the top-k candidates
    """
    tau_class: float = 0.9
    tau_make: float = 0.9
    top_k: int = 5

    def decide(self, probs: np.ndarray, mprobs: np.ndarray) -> np.ndarray:
        level = np.full(len(probs), "uncertain", dtype=object)
        level[mprobs.max(1) >= self.tau_make] = "make"
        level[probs.max(1) >= self.tau_class] = "model"
        return level

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_policy(policy: Policy, probs, mprobs, y, y_make) -> dict:
    level = policy.decide(probs, mprobs)
    out = {"policy": policy.to_dict()}
    for name, correct in [("model", probs.argmax(1) == y), ("make", mprobs.argmax(1) == y_make)]:
        m = level == name
        out[name] = {"coverage": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None}
    m = level == "uncertain"
    topk = np.argsort(-probs, 1)[:, :policy.top_k]
    out["uncertain"] = {"coverage": float(m.mean()),
                        f"top{policy.top_k}_contains_true": float((topk[m] == y[m, None]).any(1).mean()) if m.any() else None,
                        "top1_accuracy_if_forced": float((probs.argmax(1)[m] == y[m]).mean()) if m.any() else None}
    return out


def sweep_policy(probs, mprobs, y, y_make, taus=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95), tau_make: float = 0.9) -> list[dict]:
    return [evaluate_policy(Policy(t, tau_make), probs, mprobs, y, y_make) for t in taus]
