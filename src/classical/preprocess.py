"""Classical image preprocessing and region-of-interest extraction.

Every function takes/returns OpenCV BGR uint8 images. `run_pipeline` returns all
intermediate stages so the app can display them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src import config

# Fixed working size for feature extraction (width, height). Cars are wide, so 2:1.
WORK_W, WORK_H = 256, 128


def clahe_lab(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited histogram equalisation on the L channel only (keeps colours)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def auto_gamma(img: np.ndarray, target: float = 0.5) -> tuple[np.ndarray, float]:
    """Gamma-correct so the mean brightness moves toward `target` (0-1)."""
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    mean = float(np.clip(mean, 0.05, 0.95))
    gamma = float(np.clip(np.log(target) / np.log(mean), 0.4, 2.5))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut), gamma


def denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def padded_crop(img: np.ndarray, box, pad: float = config.CROP_PAD) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the padded box; also return the original box in crop coordinates."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    cx2, cy2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    crop = img[cy1:cy2, cx1:cx2]
    inner = (int(x1) - cx1, int(y1) - cy1, int(x2) - cx1, int(y2) - cy1)
    return crop, inner


def grabcut_mask(img: np.ndarray, rect: tuple[int, int, int, int], iters: int = 3) -> np.ndarray:
    """Foreground (car) mask from GrabCut seeded with the detection box. uint8 {0, 255}."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    x1, y1 = max(1, x1), max(1, y1)
    x2, y2 = min(w - 2, x2), min(h - 2, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if fg.mean() < 255 * 0.05:  # GrabCut collapsed: fall back to the box itself
        fg[y1:y2, x1:x2] = 255
    return fg


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


@dataclass
class Stages:
    """Intermediate results, all at the fixed working size (WORK_W x WORK_H)."""
    crop: np.ndarray
    clahe: np.ndarray
    gamma_img: np.ndarray
    gamma: float
    enhanced: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    edges: np.ndarray
    contour: np.ndarray | None
    box_in_crop: tuple[int, int, int, int]
    meta: dict = field(default_factory=dict)


def run_pipeline(img_bgr: np.ndarray, box, use_clahe: bool = True, use_gamma: bool = True,
                 use_grabcut: bool = True) -> Stages:
    crop, inner = padded_crop(img_bgr, box)
    ch, cw = crop.shape[:2]
    sx, sy = WORK_W / cw, WORK_H / ch
    crop = cv2.resize(crop, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    inner = (int(inner[0] * sx), int(inner[1] * sy), int(inner[2] * sx), int(inner[3] * sy))

    x = clahe_lab(crop) if use_clahe else crop
    clahe_img = x
    if use_gamma:
        x, gamma = auto_gamma(x)
    else:
        gamma = 1.0
    gamma_img = x
    enhanced = denoise(x)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    if use_grabcut:
        mask = grabcut_mask(enhanced, inner)
    else:
        mask = np.zeros((WORK_H, WORK_W), np.uint8)
        mask[inner[1]:inner[3], inner[0]:inner[2]] = 255
    edges = auto_canny(gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    return Stages(crop, clahe_img, gamma_img, gamma, enhanced, gray, mask, edges, contour, inner,
                  meta={"orig_box_w": float(box[2]) - float(box[0]), "orig_box_h": float(box[3]) - float(box[1]),
                        "img_w": img_bgr.shape[1], "img_h": img_bgr.shape[0]})


def visualize(st: Stages) -> dict[str, np.ndarray]:
    """RGB images of each stage for display (app / docs)."""
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    contour_img = st.enhanced.copy()
    if st.contour is not None:
        cv2.drawContours(contour_img, [st.contour], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(st.contour)
        cv2.drawContours(contour_img, [hull], -1, (255, 128, 0), 1)
    x1, y1, x2, y2 = st.box_in_crop
    cv2.rectangle(contour_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    masked = cv2.bitwise_and(st.enhanced, st.enhanced, mask=st.mask)
    return {
        "1. ROI crop": rgb(st.crop),
        "2. CLAHE (LAB-L)": rgb(st.clahe),
        f"3. Gamma (γ={st.gamma:.2f}) + bilateral": rgb(st.enhanced),
        "4. Canny edges": st.edges,
        "5. GrabCut mask": st.mask,
        "6. Foreground": rgb(masked),
        "7. Contour + convex hull": rgb(contour_img),
    }
