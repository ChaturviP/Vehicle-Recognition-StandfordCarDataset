import numpy as np

from src.eval import metrics as M


def test_topk():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 1])
    assert M.topk_accuracy(probs, y, 1) == 1 / 3
    assert M.topk_accuracy(probs, y, 2) == 1.0


def test_iou():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
    np.testing.assert_allclose(M.box_iou(a, b), [1.0, 50 / 150])


def test_ap_perfect_and_miss():
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    ap = M.average_precision(gt, np.array([0, 1]), gt.astype(float), np.array([0.9, 0.8]))
    assert abs(ap - 1.0) < 1e-9
    bad = np.array([[20, 20, 30, 30], [0, 0, 10, 10]], float)
    ap = M.average_precision(gt, np.array([0, 1]), bad, np.array([0.9, 0.8]))
    assert 0 < ap < 1


def test_ece_perfectly_calibrated_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    ece, _ = M.expected_calibration_error(probs, np.array([0, 1]))
    assert ece == 0.0


def test_coverage_curve_monotone():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=200)
    y = rng.integers(0, 5, 200)
    cov = [r["coverage"] for r in M.coverage_curve(probs, y)]
    assert cov == sorted(cov, reverse=True)
