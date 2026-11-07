import numpy as np

from src.hybrid.build_hybrid import fuse
from src.hybrid.confidence import Policy, calibrated_probs, evaluate_policy, fit_temperature, make_probs


def test_temperature_scales_with_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 2000)
    logits = rng.normal(size=(2000, 10))
    logits[np.arange(2000), y] += 2.0
    T1, T3 = fit_temperature(logits, y), fit_temperature(logits * 3, y)
    assert abs(T3 / T1 - 3) < 0.05  # sharpening logits 3x needs a 3x larger temperature


def test_make_probs_sum_and_policy_levels():
    c2m = np.array([0, 0, 1, 1])
    p = np.array([[0.95, 0.03, 0.01, 0.01],   # confident model-level
                  [0.50, 0.45, 0.03, 0.02],   # unsure model, sure make -> make
                  [0.30, 0.20, 0.30, 0.20]])  # unsure both -> uncertain
    mp = make_probs(p, c2m, 2)
    np.testing.assert_allclose(mp.sum(1), 1)
    levels = Policy(0.9, 0.9).decide(p, mp)
    assert list(levels) == ["model", "make", "uncertain"]
    res = evaluate_policy(Policy(0.9, 0.9), p, mp, np.array([0, 1, 2]), np.array([0, 0, 1]))
    assert res["model"]["accuracy"] == 1.0 and res["make"]["accuracy"] == 1.0


def test_fuse_identity_and_normalised():
    rng = np.random.default_rng(1)
    pd_, pc = rng.dirichlet(np.ones(5), 4), rng.dirichlet(np.ones(5), 4)
    np.testing.assert_allclose(fuse(pd_, pc, 0.0, 1.0), pd_)
    np.testing.assert_allclose(fuse(pd_, pc, 0.3, 0.7).sum(1), 1)
    np.testing.assert_allclose(calibrated_probs(np.log(pd_), 1.0), pd_, rtol=1e-6)
