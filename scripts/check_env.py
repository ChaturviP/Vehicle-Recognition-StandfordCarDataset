"""Print the versions and GPU status the project depends on.

    python scripts/check_env.py
"""
import importlib
import platform
import sys

print(f"Python   {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

for mod in ["numpy", "pandas", "PIL", "sklearn", "cv2", "skimage", "timm", "ultralytics",
            "onnx", "onnxruntime", "gradio", "lightgbm"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<12} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{mod:<12} MISSING")

try:
    import torch
    print(f"torch        {torch.__version__} (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU          {p.name}, {p.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        print(f"matmul test  ok ({(x @ x).float().abs().mean().item():.3f})")
    else:
        print("GPU          NOT AVAILABLE - install the CUDA build of torch (see README)")
except ImportError:
    print("torch        MISSING")
