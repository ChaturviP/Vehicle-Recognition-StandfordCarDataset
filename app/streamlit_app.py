"""Streamlit front end (same engine and content as the Gradio app), for Streamlit Community Cloud.

Run locally from the repo root:   streamlit run app/streamlit_app.py
Deployed from the folder built by scripts/build_streamlit.py.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

import ui  # noqa: E402
from content import ABOUT, _metrics, comparison_markdown, img_path  # noqa: E402
from inference import Engine, draw  # noqa: E402

st.set_page_config(page_title="Fine-Grained Vehicle Recognition", page_icon=":material/directions_car:",
                   layout="wide")

# Example photos (Wikimedia Commons, CC BY-SA 4.0; see examples/CREDITS.md).
EXAMPLES = [
    ("tesla_model_s.jpg", "Tesla Model S", "In the 196 classes"),
    ("jeep_wrangler.jpg", "Jeep Wrangler", "In the 196 classes"),
    ("ford_mustang_2024.jpg", "2024 Ford Mustang", "Newer than the dataset"),
]
EXAMPLE_DIR = next((d for d in (HERE / "examples", ROOT / "examples") if d.exists()), None)


@st.cache_resource(show_spinner="Loading models (first visit only)…")
def get_engine() -> Engine:
    return Engine(threads=2)


@st.cache_data(show_spinner=False, max_entries=24)
def thumbnail(path: str) -> Image.Image:
    im = Image.open(path)
    im.thumbnail((480, 480))
    return im.convert("RGB")


@st.cache_data(show_spinner=False, max_entries=12)
def predict(data: bytes, tau_c: float, tau_m: float):
    """Cached on the image bytes and thresholds, so toggles and reruns don't recompute."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    rgb = np.asarray(img)
    return rgb, ENGINE.predict(rgb, tau_c, tau_m)


ENGINE = get_engine()
NAME_TO_ID = {n: i for i, n in enumerate(ENGINE.names)}
DEFAULTS = {"tau_c": float(ENGINE.cfg["tau_class"]), "tau_m": float(ENGINE.cfg["tau_make"])}
for k, v in {"up_key": 0, "example": None}.items():
    st.session_state.setdefault(k, v)


def meta_for(class_name: str):
    i = NAME_TO_ID.get(class_name)
    return ENGINE.meta.get(i) if i is not None else None


def pick_example(fname: str):
    st.session_state.example = fname
    st.session_state.up_key += 1      # clears any uploaded file so the example is what's shown


def clear_example():
    st.session_state.example = None


st.html(ui.CSS)

_h = _metrics("hybrid_full")
facts = [("196", "car classes it can name", {"to": 196}), ("INT8", "ConvNeXt-Base, running on CPU", {})]
if _h:
    _t1 = _h["classification"]["top1"] * 100
    facts.insert(1, (f"{_t1:.1f}%", f"top-1 on {_h['n_images']:,} unseen test images",
                     {"to": f"{_t1:.1f}", "dec": 1, "suf": "%"}))
st.html(ui.hero(facts), unsafe_allow_javascript=True)

# Decision thresholds are fixed to the values selected on the validation split
# (models/hybrid_config.json); they are not user-adjustable on the deployed site.
tau_c, tau_m = DEFAULTS["tau_c"], DEFAULTS["tau_m"]
show_others = True

tab_rec, tab_cmp, tab_about = st.tabs(["Recognise", "Model comparison", "About"])

with tab_rec:
    up = st.file_uploader("Upload a car photo", type=["jpg", "jpeg", "png", "webp", "bmp"],
                          key=f"up{st.session_state.up_key}",
                          help="Works best when one car is clearly visible. Up to 25 MB.")
    if EXAMPLE_DIR is not None:
        with st.container(key="examples"):
            st.caption("No photo to hand? Try one of these.")
            cols = st.columns(len(EXAMPLES))
            for col, (fname, label, note) in zip(cols, EXAMPLES):
                with col:
                    st.image(thumbnail(str(EXAMPLE_DIR / fname)), width="stretch")
                    active = st.session_state.example == fname and up is None
                    st.button(label, key=f"ex_{fname}", on_click=pick_example, args=(fname,),
                              type="primary" if active else "secondary", width="stretch")
                    st.html(f'<p class="p5-note">{ui.esc(note)}</p>')

    data, source_name = None, None
    if up is not None:
        data, source_name = up.getvalue(), up.name
    elif st.session_state.example and EXAMPLE_DIR is not None:
        data, source_name = (EXAMPLE_DIR / st.session_state.example).read_bytes(), st.session_state.example

    if data is None:
        st.html(ui.empty_state())
    else:
        try:
            with st.spinner("Detecting and classifying…"):
                rgb, res = predict(data, tau_c, tau_m)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error(f"Couldn't read **{source_name}** as an image. Upload a JPG, PNG, WebP or BMP photo.",
                     icon=":material/broken_image:")
            res = None
        if res is not None:
            with st.container(key="result"):
                st.html(ui.section("01", "Result", "The detected car, the answer the confidence policy allows, "
                                   "and the five closest classes."))
                left, right = st.columns([3, 2], gap="large")
                with left:
                    cap = ("Your photo: no car was detected" if res.level == "no_car"
                           else "Detected car with the hybrid prediction")
                    if res.other_boxes and show_others:
                        cap += "; grey boxes are other vehicles, ignored"
                    st.image(draw(rgb, res, show_others), caption=cap, width="stretch")
                    st.html(ui.legend())
                    if st.session_state.example and up is None:
                        st.button("Clear example", on_click=clear_example, icon=":material/close:")
                with right:
                    st.html(ui.verdict(res, meta_for, tau_c, tau_m))
                    if res.top5:
                        st.html(ui.ranked(res.top5, "Top-5 candidates · calibrated hybrid probability"))
                if res.level != "no_car":
                    st.html(ui.section("02", "Classical vs deep vs hybrid on this image",
                                       "The same detected crop, scored by each approach. The hybrid row gives the answer above."))
                    st.html(ui.branches(res))
                    st.html(ui.section("03", "Inside the classical branch",
                                       "What the handcrafted pipeline sees before it extracts colour, texture, edge "
                                       "and shape features."))
                    st.html(ui.stages(res.stages))

with tab_cmp:
    cards = [("Classical", "Handcrafted colour, texture, HOG and shape features into an RBF SVM", "classical_svm_full"),
             ("Deep", "ConvNeXt-Base at 448 px on the detector's crops", "convnext_base_448_detcrop"),
             ("Hybrid", "Calibrated deep scores, with the SVM fused in when the network is unsure", "hybrid_full")]
    cards = [(t, d, m) for t, d, n in cards if (m := _metrics(n))]
    if cards:
        st.html(ui.section("01", "Three approaches, one test set",
                           "Scored on the 8,041 Stanford Cars test images none of the models saw in training. "
                           "Hybrid keeps the network's accuracy and is far better calibrated."))
        best = min(cards, key=lambda r: r[2]["classification"]["ece"])[0]
        st.html(ui.compare_cards(cards, best))
    st.markdown(comparison_markdown())
    T = ENGINE.cfg.get("temperature")
    plots = [
        ("Calibration and accuracy vs coverage", ["reliability_coverage.png", "hybrid/reliability_coverage.png"],
         "Left: predicted confidence against actual accuracy; the dashed diagonal is perfect calibration. "
         + (f"Temperature scaling (T = {T:.2f}) " if T else "Temperature scaling ")
         + "brings the curve onto the diagonal. Right: accuracy on the images the system chooses to answer, "
           "as it answers more of them."),
        ("Robustness under corruptions", ["robustness.png", "robustness/robustness.png"],
         "Top-1 accuracy on corrupted test crops (lighting, noise, blur, occlusion, low resolution). "
         "Orange bars add the classical CLAHE/gamma preprocessing in front of the network."),
        ("Classical pipeline stages on test images", ["stages_test_samples.png", "classical/stages_test_samples.png"],
         "Each row is one test image passing through the seven classical stages."),
    ]
    with st.container(key="plots"):
        for title, names, caption in plots:
            p = img_path(*names)
            if p:
                st.subheader(title)
                st.image(p, width="stretch")
                st.caption(caption)

with tab_about:
    st.html(ui.section("01", "How a photo becomes an answer"))
    st.html(ui.pipeline())
    st.markdown(ABOUT)
    arch = img_path("architecture.png") or next(
        (str(p) for p in (ROOT / "docs" / "architecture.png",) if p.exists()), None)
    if arch:
        with st.container(key="arch"):
            st.subheader("System architecture")
            st.image(arch, width="stretch")

st.html(ui.footer())
