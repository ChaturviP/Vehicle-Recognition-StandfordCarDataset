"""Gradio app: Intelligent Fine-Grained Vehicle Recognition (Stanford Cars, 196 classes).

Run locally from the repo root:   python app/app.py
On Hugging Face Spaces this file is the Space's app.py (see scripts/build_space.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from content import ABOUT, ROOT, branch_table, comparison_markdown, img_path, verdict_markdown  # noqa: E402
from inference import Engine, draw  # noqa: E402

ENGINE = Engine()


def recognise(image, tau_class, tau_make, show_others):
    if image is None:
        return None, "Upload an image of a car first.", None, [], ""
    rgb = np.asarray(image)[:, :, :3]
    res = ENGINE.predict(rgb, tau_class, tau_make)
    annotated = draw(rgb, res, show_others)
    if res.level == "no_car":
        return annotated, verdict_markdown(res), None, [], ""
    return (annotated, verdict_markdown(res), {n: p for n, p in res.top5},
            [(img, name) for name, img in res.stages.items()], branch_table(res))


with gr.Blocks(title="Fine-Grained Vehicle Recognition") as demo:
    gr.Markdown("# 🚗 Intelligent Fine-Grained Vehicle Recognition\n"
                "Detects a car, then identifies its **make, model and year** among 196 Stanford Cars classes, "
                "combining classical computer vision with deep learning.")
    with gr.Tab("Recognise"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Upload a car photo", sources=["upload", "clipboard"])
                with gr.Accordion("Confidence thresholds", open=False):
                    tau_c = gr.Slider(0.3, 0.99, value=ENGINE.cfg["tau_class"], step=0.01,
                                      label="τ model: min confidence for a full make/model/year answer")
                    tau_m = gr.Slider(0.3, 0.99, value=ENGINE.cfg["tau_make"], step=0.01,
                                      label="τ make: min confidence for a manufacturer-only answer")
                    others = gr.Checkbox(value=True, label="Show other detected vehicles (grey)")
                btn = gr.Button("Recognise", variant="primary")
                examples_dir = ROOT / "examples"
                if examples_dir.exists():
                    ex = sorted(str(p) for p in examples_dir.glob("*.jpg"))[:8]
                    if ex:
                        gr.Examples(ex, inputs=inp)
            with gr.Column(scale=1):
                out_img = gr.Image(label="Detection + prediction", type="numpy")
                verdict = gr.Markdown()
                top5 = gr.Label(num_top_classes=5, label="Top-5 (calibrated hybrid probabilities)")
        gr.Markdown("### Classical vs deep vs hybrid on this image")
        branch_md = gr.Markdown()
        gr.Markdown("### Intermediate image-processing results (classical branch)")
        stages = gr.Gallery(columns=4, height="auto", object_fit="contain", label="Processing stages")
        btn.click(recognise, [inp, tau_c, tau_m, others], [out_img, verdict, top5, stages, branch_md])
    with gr.Tab("Model comparison"):
        gr.Markdown(comparison_markdown())
        for title, names in [("Calibration and accuracy vs coverage", ["reliability_coverage.png", "hybrid/reliability_coverage.png"]),
                             ("Robustness under corruptions", ["robustness.png", "robustness/robustness.png"]),
                             ("Classical pipeline stages on test images", ["stages_test_samples.png", "classical/stages_test_samples.png"])]:
            p = img_path(*names)
            if p:
                gr.Markdown(f"### {title}")
                gr.Image(p, show_label=False, interactive=False)
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
