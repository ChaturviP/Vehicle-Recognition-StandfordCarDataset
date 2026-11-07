import numpy as np

from src.classical.features import GROUPS, extract
from src.classical.preprocess import WORK_H, WORK_W, run_pipeline, visualize

EXPECTED = {"color": 48, "lbp": 40, "glcm": 12, "hog": 3780, "edge": 17, "shape": 12, "geom": 2}


def _image():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    img[120:260, 80:420] = (40, 160, 40)
    return img


def test_feature_dims_and_finite():
    st = run_pipeline(_image(), (80, 120, 420, 260))
    feats = extract(st)
    assert set(feats) == set(GROUPS)
    for g, v in feats.items():
        assert v.shape == (EXPECTED[g],), g
        assert np.isfinite(v).all(), g


def test_stages_shapes_and_ablation_switches():
    st = run_pipeline(_image(), (80, 120, 420, 260), use_clahe=False, use_gamma=False, use_grabcut=False)
    assert st.mask.shape == (WORK_H, WORK_W) and st.gamma == 1.0
    assert len(visualize(st)) == 7


def test_degenerate_box_does_not_crash():
    st = run_pipeline(_image(), (10, 10, 14, 13))
    assert np.isfinite(np.concatenate(list(extract(st).values()))).all()
