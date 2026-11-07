"""Export the classifier (and detector) to ONNX, quantize to INT8, and benchmark on CPU.

    python -m src.optimize.export_onnx --ckpt models/classifier_convnext_base_448.pt --detector models/detector_yolo11s_cars.pt

Produces models/<name>_fp32.onnx and models/<name>_int8.onnx (dynamic INT8 quantization of
all MatMul/Gemm weights, which covers ConvNeXt's pointwise MLP layers = most of its weights).
--static additionally builds a static QDQ INT8 model (Conv included) calibrated on 200 val crops.

Benchmark (CPU, --threads intra-op threads; 2 mimics a free Hugging Face Space): model size,
batch-1 latency, and top-1 / top-5 on a fixed subset of test detector crops (--n). Saves
results/preds/<name>_<variant>.npz per variant and results/optimize/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import config
from src.data.dataset import CarsDataset, load_det_boxes, load_index
from src.deep.model import build_transforms, load_checkpoint
from src.eval.evaluate import save_predictions

OUT_DIR = config.RESULTS_DIR / "optimize"


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def export_fp32(model, img_size: int, path: Path) -> Path:
    dummy = torch.randn(1, 3, img_size, img_size)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:  # classic TorchScript-based exporter: no extra dependencies
        torch.onnx.export(LogitsOnly(model).eval(), dummy, str(path), dynamo=False, **kw)
    except TypeError:  # very old torch without the `dynamo` argument
        torch.onnx.export(LogitsOnly(model).eval(), dummy, str(path), **kw)
    return path


def quantize_dynamic_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    return dst


def quantize_static_int8(src: Path, dst: Path, calib: np.ndarray) -> Path:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": calib[i:i + 1]} for i in range(len(calib))])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    return dst


def ort_session(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def stack_inputs(ds, idx) -> np.ndarray:
    return np.stack([ds[i][0].numpy() for i in idx]).astype(np.float32)


def run_variant(name, fn, X, y, warmup=3, n_timed=30):
    for i in range(warmup):
        fn(X[i:i + 1])
    lat = []
    for i in range(min(n_timed, len(X))):
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    logits = np.concatenate([fn(X[i:i + 16]) for i in range(0, len(X), 16)])
    p = torch.softmax(torch.from_numpy(logits.astype(np.float32)), 1).numpy()
    top1 = float((p.argmax(1) == y).mean())
    top5 = float((np.argsort(-p, 1)[:, :5] == y[:, None]).any(1).mean())
    print(f"  {name:<16} top-1 {top1:.4f}  top-5 {top5:.4f}  latency {np.mean(lat):7.1f} ms (batch 1)")
    return p, np.array(lat), top1, top5


def benchmark_detector(weights: Path, threads: int, n: int) -> list[dict]:
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=640, simplify=True, dynamic=False))
    paths = load_index("test").sample(n, random_state=config.SEED)["path"].tolist()
    rows = []
    for variant, w in [("pytorch_fp32_cpu", weights), ("onnx_fp32_cpu", onnx_path)]:
        yolo = YOLO(str(w), task="detect")
        yolo.predict(paths[0], device="cpu", verbose=False)
        lat = [sum(yolo.predict(p, device="cpu", verbose=False)[0].speed.values()) for p in paths]
        rows.append({"model": "detector", "variant": variant, "size_mb": Path(w).stat().st_size / 2**20,
                     "latency_ms": float(np.mean(lat))})
        print(f"  detector {variant:<18} {rows[-1]['size_mb']:.1f} MB  {rows[-1]['latency_ms']:.1f} ms/img")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--det-preds", type=Path, default=config.RESULTS_DIR / "preds" / "detector_yolo11s_cars.npz")
    ap.add_argument("--n", type=int, default=500, help="test images for the accuracy check (CPU is slow)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--static", action="store_true", help="also build a static QDQ INT8 model")
    ap.add_argument("--skip-pytorch", action="store_true", help="skip the slow PyTorch-CPU baseline row")
    ap.add_argument("--detector", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, ckpt = load_checkpoint(args.ckpt, "cpu")
    hp = ckpt["args"]
    name, img = hp["name"], hp["img"]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = export_fp32(model, img, config.MODELS_DIR / f"{name}_fp32.onnx")
    int8 = quantize_dynamic_int8(fp32, config.MODELS_DIR / f"{name}_int8.onnx")
    print(f"Exported {fp32} and {int8}")

    tf = build_transforms(img, False)
    df_all = load_index("test")
    boxes_all = load_det_boxes(args.det_preds, df_all)
    sub = df_all.sample(min(args.n, len(df_all)), random_state=config.SEED).index.to_numpy()
    df = df_all.iloc[sub].reset_index(drop=True)
    X = stack_inputs(CarsDataset(df, tf, crop="pred", boxes=boxes_all[sub]), range(len(df)))
    y = df["class_id"].to_numpy()

    variants = {}
    if not args.skip_pytorch:
        def torch_fn(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()
        variants["pytorch_fp32"] = (torch_fn, args.ckpt)
    for vname, path in [("onnx_fp32", fp32), ("onnx_int8_dynamic", int8)]:
        sess = ort_session(path, args.threads)
        variants[vname] = (lambda x, s=sess: s.run(None, {"input": x})[0], path)
    if args.static:
        val = load_index("val").sample(200, random_state=config.SEED)
        calib = stack_inputs(CarsDataset(val, tf, crop="gt"), range(len(val)))
        st = quantize_static_int8(fp32, config.MODELS_DIR / f"{name}_int8_static.onnx", calib)
        sess = ort_session(st, args.threads)
        variants["onnx_int8_static"] = (lambda x, s=sess: s.run(None, {"input": x})[0], st)

    print(f"CPU benchmark: {len(df)} test crops, {args.threads} threads")
    rows = []
    for vname, (fn, path) in variants.items():
        p, lat, top1, top5 = run_variant(vname, fn, X, y)
        size = Path(path).stat().st_size / 2**20
        rows.append({"model": name, "variant": vname, "size_mb": size, "latency_ms": float(lat.mean()),
                     "top1": top1, "top5": top5, "n": len(df), "threads": args.threads})
        save_predictions(f"{name}_{vname}_cpu", df["filename"], probs=p, latency_ms=lat, model_size_mb=size,
                         family="deep", device=f"CPU {vname} ({args.threads} threads)", n_subset=len(df))
    if args.detector:
        rows += benchmark_detector(args.detector, args.threads, 50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
