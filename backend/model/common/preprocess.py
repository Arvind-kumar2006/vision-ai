"""One preprocessing module for both features (OpenCV).

Steps, each switchable so the OCR comparison (scripts/compare_ocr.py) can measure what helps:
    exif         rotate the photo the way the camera held it (always on: free and never hurts)
    resize       long side to RESIZE_LONG px (small photos upscaled, huge ones reduced)
    perspective  find the page / label outline (largest 4-corner shape covering >= 30 % of the photo) and
                 flatten it; skipped when no such outline is found (most label photos are already crops)
    clahe        local contrast (CLAHE on the lightness channel), for dim or uneven light

Which steps are on is decided per feature by the comparison on our eval photos (LABEL_STEPS,
PRESCRIPTION_STEPS; project_review/ADR-005).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

RESIZE_LONG = 1600
MIN_PAGE_AREA = 0.30
ALL_STEPS = ("exif", "resize", "perspective", "clahe")
# Chosen from scripts/compare_ocr.py on our eval photos (ADR-005): CLAHE and perspective helped the label photos
# (73.2 % -> 81.2 % of values) but CLAHE cost medicine names on the handwritten prescription, so it's per feature.
LABEL_STEPS: Tuple[str, ...] = ("exif", "resize", "perspective", "clahe")
PRESCRIPTION_STEPS: Tuple[str, ...] = ("exif", "resize")
DEFAULT_STEPS = PRESCRIPTION_STEPS


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_page(bgr: np.ndarray):
    """4 corners of the largest page-like outline, or None."""
    h, w = bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) >= MIN_PAGE_AREA * area and cv2.isContourConvex(approx):
            return _order(approx.reshape(4, 2).astype("float32") / min(scale, 1.0))
    return None


def flatten(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(corners, dst), (width, height))


def clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(image: Image.Image, steps: Iterable[str] = DEFAULT_STEPS) -> Tuple[np.ndarray, Dict[str, object]]:
    """PIL image -> (BGR array for the OCR, {"steps": [...applied], "size": (w, h)})."""
    steps = set(steps)
    applied: List[str] = []
    if "exif" in steps:
        image = ImageOps.exif_transpose(image)
        applied.append("exif")
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if "perspective" in steps:
        corners = find_page(bgr)
        if corners is not None:
            bgr = flatten(bgr, corners)
            applied.append("perspective")
    if "resize" in steps:
        h, w = bgr.shape[:2]
        scale = RESIZE_LONG / max(h, w)
        if abs(scale - 1) > 0.05:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            applied.append("resize")
    if "clahe" in steps:
        bgr = clahe(bgr)
        applied.append("clahe")
    return bgr, {"steps": applied, "size": (bgr.shape[1], bgr.shape[0])}


def prescription_variants(bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Return conservative OCR alternatives for difficult handwritten prescriptions.

    The normal prescription image must remain untouched: CLAHE and thresholding can
    improve faint pen strokes, but can also destroy names.  Callers therefore use
    these only as a fallback after the normal pass has failed to find a medicine
    section.  All variants keep the original dimensions and colour-channel contract
    expected by :class:`OCREngine`.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = clahe(bgr)
    # Otsu is less aggressive than a fixed threshold for photos with shadows.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Preserve a 3-channel BGR input for RapidOCR and avoid changing the default path.
    return [("clahe", enhanced), ("otsu", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))]
