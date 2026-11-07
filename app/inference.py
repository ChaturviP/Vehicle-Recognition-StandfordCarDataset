"""CPU inference engine for the app: YOLO (ONNX) -> crop -> ConvNeXt INT8 (ONNX)
+ classical features/SVM -> calibrated hybrid prediction with low-confidence handling.

No PyTorch needed. Expected files (repo layout or the Hugging Face Space layout):
    models/detector_yolo11s_cars.onnx
    models/convnext_base_448_int8.onnx
    models/classical_svm_full.joblib
    models/hybrid_config.json
    app/classes.csv (repo, committed)  or  assets/classes.csv (standalone builds)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.classical.features import extract
from src.classical.preprocess import run_pipeline, visualize
from src.hybrid.confidence import Policy, make_probs

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "models").exists() else HERE.parent

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_PAD = 0.12


def _first(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these exist: {[str(p) for p in paths]}")


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- detector

class Detector:
    def __init__(self, path: Path, threads: int, imgsz: int = 640):
        self.sess, self.imgsz = _session(path, threads), imgsz
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, conf: float = 0.25, iou: float = 0.5):
        h, w = rgb.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0][0]          # (4 + nc, N)
        out = out.T
        scores = out[:, 4:].max(1)
        keep = scores >= conf
        if not keep.any():
            return np.zeros((0, 4)), np.zeros(0)
        b, s = out[keep, :4], scores[keep]
        xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)
        xyxy = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)[idx]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - left) / r).clip(0, w)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - top) / r).clip(0, h)
        order = np.argsort(-s[idx])
        return xyxy[order], s[idx][order]


# ---------------------------------------------------------------- result

@dataclass
class Result:
    box: tuple | None
    box_score: float
    other_boxes: list
    level: str                     # "model" | "make" | "uncertain" | "no_car"
    headline: str
    top5: list                     # [(class_name, prob)]
    make: str | None
    make_prob: float
    per_branch: dict               # name -> {"top3": [(name, p)], "ms": float}
    stages: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    fused: bool = False


class Engine:
    def __init__(self, threads: int | None = None):
        threads = threads or int(os.environ.get("ORT_THREADS", min(4, os.cpu_count() or 2)))
        m = ROOT / "models"
        self.detector = Detector(m / "detector_yolo11s_cars.onnx", threads)
        self.clf = _session(_first(m / "convnext_base_448_int8.onnx", m / "convnext_base_448_fp32.onnx"), threads)
        self.clf_inp = self.clf.get_inputs()[0].name
        self.img_size = self.clf.get_inputs()[0].shape[-1] if isinstance(self.clf.get_inputs()[0].shape[-1], int) else 448
        cl = joblib.load(m / "classical_svm_full.joblib")
        self.classical, self.groups = cl["model"], cl["groups"]
        self.cfg = json.loads((m / "hybrid_config.json").read_text())
        import pandas as pd
        cls = pd.read_csv(_first(ROOT / "assets" / "classes.csv", HERE / "classes.csv",
                                ROOT / "data" / "processed" / "classes.csv"))
        cls = cls.sort_values("class_id")
        self.names = cls["class_name"].tolist()
        self.class_make = cls["make_id"].to_numpy()
        self.make_names = cls.drop_duplicates("make_id").sort_values("make_id")["make"].tolist()
        self.meta = cls.set_index("class_id")[["make", "model", "year"]].to_dict("index")

    # ---- branches
    def deep_logits(self, crop_rgb: Image.Image) -> np.ndarray:
        x = np.asarray(crop_rgb.resize((self.img_size, self.img_size), Image.BILINEAR), np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return self.clf.run(None, {self.clf_inp: x})[0][0]

    def classical_probs(self, stages) -> np.ndarray:
        f = extract(stages, self.groups)
        X = np.nan_to_num(np.concatenate([f[g] for g in self.groups])[None])
        p = self.classical.predict_proba(X)[0]
        out = np.zeros(len(self.names))
        out[self.classical.classes_] = p
        return out

    def _top(self, p, k):
        idx = np.argsort(-p)[:k]
        return [(self.names[i], float(p[i])) for i in idx]

    # ---- full pipeline
    def predict(self, rgb: np.ndarray, tau_class: float | None = None, tau_make: float | None = None) -> Result:
        t = {}
        t0 = time.perf_counter()
        boxes, scores = self.detector(rgb)
        t["detector_ms"] = (time.perf_counter() - t0) * 1000
        if len(boxes) == 0:
            return Result(None, 0.0, [], "no_car", "No car detected in this image.", [], None, 0.0, {}, timings=t)
        box = tuple(float(v) for v in boxes[0])

        # deep branch on the padded crop
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        crop = Image.fromarray(rgb).crop((int(max(0, x1 - CROP_PAD * bw)), int(max(0, y1 - CROP_PAD * bh)),
                                          int(min(w, x2 + CROP_PAD * bw)), int(min(h, y2 + CROP_PAD * bh))))
        t0 = time.perf_counter()
        logits = self.deep_logits(crop)
        t["deep_ms"] = (time.perf_counter() - t0) * 1000
        p_raw = softmax(logits)
        p_deep = softmax(logits / self.cfg["temperature"])

        # classical branch (always run: its stages are shown, and it feeds gated fusion)
        t0 = time.perf_counter()
        st = run_pipeline(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), box)
        p_cls = self.classical_probs(st)
        t["classical_ms"] = (time.perf_counter() - t0) * 1000

        # hybrid: gated log-linear fusion, as selected on validation
        w_f, gate = self.cfg["fusion_w"], self.cfg["fusion_gate"]
        fused = w_f > 0 and p_deep.max() < gate
        if fused:
            logp = np.log(np.clip(p_deep, 1e-12, 1)) + w_f * np.log(np.clip(p_cls, 1e-6, 1))
            p_hyb = softmax(logp)
        else:
            p_hyb = p_deep

        policy = Policy(tau_class if tau_class is not None else self.cfg["tau_class"],
                        tau_make if tau_make is not None else self.cfg["tau_make"])
        mp = make_probs(p_hyb[None], self.class_make, len(self.make_names))[0]
        level = policy.decide(p_hyb[None], mp[None])[0]
        top = int(p_hyb.argmax())
        mk = self.make_names[int(mp.argmax())]
        if level == "model":
            m = self.meta[top]
            headline = f"{m['make']} {m['model']} ({m['year']}), {p_hyb[top]:.0%} confidence"
        elif level == "make":
            headline = (f"Manufacturer: {mk} ({mp.max():.0%} confidence). The exact model/year is uncertain "
                        f"(best guess {self.names[top]}, {p_hyb[top]:.0%}).")
        else:
            headline = (f"Uncertain: no single answer reaches the confidence threshold. "
                        f"Best guess {self.names[top]} ({p_hyb[top]:.0%}); see the top-5.")

        per_branch = {
            "Classical (handcrafted features + SVM)": {"top3": self._top(p_cls, 3), "ms": t["classical_ms"]},
            "Deep (ConvNeXt-Base INT8, uncalibrated)": {"top3": self._top(p_raw, 3), "ms": t["deep_ms"]},
            "Hybrid (calibrated + gated fusion + policy)": {
                "top3": self._top(p_hyb, 3), "ms": t["deep_ms"] + (t["classical_ms"] if fused else 0.0)},
        }
        return Result(box, float(scores[0]), [tuple(map(float, b)) for b in boxes[1:]], str(level), headline,
                      self._top(p_hyb, 5), mk, float(mp.max()), per_branch, visualize(st), t, bool(fused))


def draw(rgb: np.ndarray, res: Result, show_others: bool = True) -> np.ndarray:
    img = rgb.copy()
    th = max(2, int(round(max(img.shape[:2]) / 300)))
    fs = max(0.5, max(img.shape[:2]) / 1100)
    if show_others:
        for b in res.other_boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), max(1, th // 2))
    if res.box is None:
        return img
    color = {"model": (40, 180, 70), "make": (240, 170, 20), "uncertain": (220, 60, 50)}.get(res.level, (200, 200, 200))
    x1, y1, x2, y2 = (int(v) for v in res.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    if res.level == "model":
        label = f"{res.top5[0][0]}  {res.top5[0][1]:.0%}"
    elif res.level == "make":
        label = f"{res.make} (make only)  {res.make_prob:.0%}"
    else:
        label = f"uncertain: {res.top5[0][0]}?  {res.top5[0][1]:.0%}"
    (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th // 2))
    ty = y1 - 6 if y1 - tht - 10 > 0 else y2 + tht + 6
    cv2.rectangle(img, (x1, ty - tht - 6), (x1 + tw + 8, ty + base - 2), color, -1)
    cv2.putText(img, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
    return img
