"""Handcrafted feature groups computed from the classical pipeline stages.

Groups (used for the feature-group ablation):
    color  - HSV histograms inside the GrabCut mask                      (48)
    lbp    - uniform LBP histograms on a 2x2 grid                        (40)
    glcm   - Haralick/GLCM properties at 2 distances, angle-averaged     (12)
    hog    - HOG on the 256x128 grayscale crop                           (3780)
    edge   - Canny edge density, global + 4x4 grid                       (17)
    shape  - contour solidity/extent/aspect/circularity + log Hu moments (12)
    geom   - original box aspect ratio, box area share of the image      (2)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from src.classical.preprocess import WORK_H, WORK_W, Stages

GROUPS = ("color", "lbp", "glcm", "hog", "edge", "shape", "geom")


def color_feats(st: Stages) -> np.ndarray:
    hsv = cv2.cvtColor(st.enhanced, cv2.COLOR_BGR2HSV)
    m = st.mask if st.mask.any() else None
    hists = [cv2.calcHist([hsv], [0], m, [16], [0, 180]),
             cv2.calcHist([hsv], [1], m, [16], [0, 256]),
             cv2.calcHist([hsv], [2], m, [16], [0, 256])]
    out = np.concatenate([h.ravel() for h in hists])
    return out / max(out.sum() / 3, 1.0)


def lbp_feats(st: Stages) -> np.ndarray:
    lbp = local_binary_pattern(st.gray, P=8, R=1, method="uniform")  # values 0..9
    feats = []
    for gy in range(2):
        for gx in range(2):
            cell = lbp[gy * WORK_H // 2:(gy + 1) * WORK_H // 2, gx * WORK_W // 2:(gx + 1) * WORK_W // 2]
            h, _ = np.histogram(cell, bins=10, range=(0, 10))
            feats.append(h / max(h.sum(), 1))
    return np.concatenate(feats)


def glcm_feats(st: Stages) -> np.ndarray:
    q = (st.gray // 8).astype(np.uint8)  # 32 grey levels keeps the matrix small
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props])  # (6 props x 2 dists)


def hog_feats(st: Stages) -> np.ndarray:
    return hog(st.gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def edge_feats(st: Stages) -> np.ndarray:
    e = st.edges > 0
    grid = [e[gy * WORK_H // 4:(gy + 1) * WORK_H // 4, gx * WORK_W // 4:(gx + 1) * WORK_W // 4].mean()
            for gy in range(4) for gx in range(4)]
    return np.array([e.mean(), *grid], np.float32)


def shape_feats(st: Stages) -> np.ndarray:
    c = st.contour
    if c is None or cv2.contourArea(c) < 1:
        return np.zeros(12, np.float32)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
    x, y, w, h = cv2.boundingRect(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return np.array([
        area / hull_area,                         # solidity
        area / max(w * h, 1),                     # extent
        max(rw, rh) / max(min(rw, rh), 1e-6),     # elongation of rotated rect
        4 * np.pi * area / max(perim ** 2, 1e-6),  # circularity
        float(st.mask.mean() / 255),              # foreground share of crop
        *hu,
    ], np.float32)


def geom_feats(st: Stages) -> np.ndarray:
    m = st.meta
    return np.array([m["orig_box_w"] / max(m["orig_box_h"], 1),
                     (m["orig_box_w"] * m["orig_box_h"]) / max(m["img_w"] * m["img_h"], 1)], np.float32)


_FUNCS = {"color": color_feats, "lbp": lbp_feats, "glcm": glcm_feats, "hog": hog_feats,
          "edge": edge_feats, "shape": shape_feats, "geom": geom_feats}


def extract(st: Stages, groups=GROUPS) -> dict[str, np.ndarray]:
    return {g: _FUNCS[g](st).astype(np.float32) for g in groups}
