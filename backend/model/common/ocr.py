"""On-device OCR with RapidOCR (PaddleOCR models run on onnxruntime). One engine for both features.

Models are files in backend/weights/ocr/ (scripts/setup_ocr_models.py downloads them once). Nothing is ever
downloaded at request time: a missing model file is an error ("run scripts/setup_ocr_models.py").

    det        text-box detection (PP-OCRv6 / PP-OCRv5)
    rec        recognition, Latin script (English labels and prescriptions)
    rec_hindi  optional second recognizer, Devanagari (PP-OCRv5): every box is read by both and the reading
               with the higher confidence is kept, so printed Hindi ("रजि. क्र.") and Hindi timing words
               ("रात") come out as Hindi instead of Latin garbage.

Two profiles keep the two features independent:

    label         current behaviour: RapidOCR's per-line 0/180 classifier is used, and RapidOCR's default
                  recognition score cutoff applies.
    prescription  no per-line classifier (it damages handwriting), and every non-empty recognition candidate
                  is kept (``text_score=0``) so a low score is evidence, not automatic deletion.

Output rows are text boxes grouped into lines (left to right), each with a confidence. Every box also carries
its candidate readings and where they came from:
    {"ok", "rows": [{"text", "confidence", "box", "boxes"}], "boxes": [...], "lines": [str],
     "profile", "rotated", "ms", "error"}

Candidate boxes are internal evidence only; no image bytes or crop URLs are returned.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger("common.ocr")

MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "ocr"
CLS_MODEL = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
# Sets compared in scripts/compare_ocr.py; DEFAULT chosen from that comparison (ADR-005).
MODEL_SETS = {
    "v6-small": ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx"),
    "v6-medium": ("PP-OCRv6_det_medium.onnx", "PP-OCRv6_rec_medium.onnx"),
    "v5-mobile": ("ch_PP-OCRv5_det_mobile.onnx", "en_PP-OCRv5_rec_mobile.onnx"),
    "v5-server": ("ch_PP-OCRv5_det_server.onnx", "ch_PP-OCRv5_rec_server.onnx"),
}
HINDI_REC = "devanagari_PP-OCRv5_rec_mobile.onnx"
DEFAULT_SET = os.getenv("OCR_MODELS", "v6-small")
DEFAULT_HINDI = os.getenv("OCR_HINDI", "1") not in ("0", "false", "no")
PROFILE_LABEL = "label"
PROFILE_PRESCRIPTION = "prescription"
PROFILES = (PROFILE_LABEL, PROFILE_PRESCRIPTION)
LABEL_TEXT_SCORE = 0.5     # RapidOCR's default; passed explicitly because the engine remembers the last value
SAMPLE_BOXES = 6           # sideways photos: boxes read both ways to choose the quarter turn
UPSIDE_DOWN_BELOW = 0.75   # below this, an upside-down photo is checked
SETUP_HINT = "OCR models are missing: run `venv/bin/python scripts/setup_ocr_models.py`"


def _model(name: str) -> str:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{SETUP_HINT} ({name} not in {MODEL_DIR})")
    return str(path)


# Text-box detector sensitivity: RapidOCR's defaults 0.3 / 0.5. Lower values (0.2 / 0.3) found one more faint
# ballpoint line on one prescription but added noise boxes that lowered the label eval from 94.6 % to 90.2 % and
# the prescription evals too (2026-11-13), so the defaults stay. Override with OCR_DET_THRESH / OCR_DET_BOX_THRESH.
DET_THRESH = float(os.getenv("OCR_DET_THRESH", "0.3"))
DET_BOX_THRESH = float(os.getenv("OCR_DET_BOX_THRESH", "0.5"))


def _engine(det: str, rec: str):
    from rapidocr import RapidOCR
    return RapidOCR(params={"Det.model_path": _model(det), "Rec.model_path": _model(rec),
                            "Cls.model_path": _model(CLS_MODEL), "Global.log_level": "error",
                            "Det.thresh": DET_THRESH, "Det.box_thresh": DET_BOX_THRESH})


def _crop(img: np.ndarray, quad: np.ndarray, rotate_tall: bool = False) -> np.ndarray:
    """The crop inside a detected quad. Tall crops are NOT rotated implicitly: aspect ratio alone is not
    evidence that handwriting is sideways. Callers that genuinely need a quarter turn must ask for it
    (``rotate_tall=True``) and keep both readings as candidates."""
    quad = quad.astype("float32")
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (max(w, 1), max(h, 1)),
                               borderMode=cv2.BORDER_REPLICATE)
    return np.rot90(crop) if rotate_tall and h >= 1.5 * w else crop


def group_rows(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Text boxes -> lines. A box joins a line when it overlaps the line vertically by more than half of
    the smaller height; boxes in a line are ordered left to right."""
    rows: List[Dict[str, Any]] = []
    for b in sorted(boxes, key=lambda b: (b["box"][1] + b["box"][3]) / 2):
        y0, y1 = b["box"][1], b["box"][3]
        best = None
        for r in rows:
            overlap = min(y1, r["y1"]) - max(y0, r["y0"])
            if overlap > 0.5 * min(y1 - y0, r["y1"] - r["y0"]):
                best = r
                break
        if best is None:
            rows.append({"y0": y0, "y1": y1, "items": [b]})
        else:
            best["items"].append(b)   # the line keeps its first box's band, so lines never merge
    out = []
    for r in sorted(rows, key=lambda r: r["y0"]):
        items = sorted(r["items"], key=lambda b: b["box"][0])
        out.append({"text": " ".join(b["text"] for b in items),
                    "confidence": round(float(np.mean([b["confidence"] for b in items])), 3),
                    "box": [min(b["box"][0] for b in items), min(b["box"][1] for b in items),
                            max(b["box"][2] for b in items), max(b["box"][3] for b in items)],
                    "boxes": items})
    return out


class OCREngine:
    def __init__(self, models: str = DEFAULT_SET, hindi: bool = DEFAULT_HINDI) -> None:
        self.models, self.hindi = models, hindi
        self.name = f"RapidOCR {models}" + (" + Devanagari" if hindi else "")
        self._main = self._hi = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    def load(self) -> bool:
        if self._main is not None:
            return True
        with self._lock:
            if self._main is None:
                try:
                    det, rec = MODEL_SETS[self.models]
                    self._main = _engine(det, rec)
                    if self.hindi:
                        self._hi = _engine(det, HINDI_REC)
                except Exception as exc:   # missing model files, broken install
                    self.error = str(exc)
                    log.error("OCR unavailable: %s", exc)
                    return False
        return True

    @property
    def available(self) -> bool:
        return self.load()

    @staticmethod
    def _check_profile(profile: str) -> str:
        if profile not in PROFILES:
            raise ValueError(f"unknown OCR profile {profile!r}; expected one of {PROFILES}")
        return profile

    def _full(self, bgr: np.ndarray, profile: str = PROFILE_LABEL):
        """Full OCR pass. The prescription profile deliberately does not use RapidOCR's per-line 0/180
        classifier (it rotates handwriting crops) and keeps every non-empty recognition candidate
        (``text_score=0``). The label profile is unchanged."""
        self._check_profile(profile)
        # RapidOCR remembers use_* from the previous call, so always pass all three.
        if profile == PROFILE_PRESCRIPTION:
            return self._main(bgr, use_det=True, use_cls=False, use_rec=True, text_score=0.0)
        return self._main(bgr, use_det=True, use_cls=True, use_rec=True, text_score=LABEL_TEXT_SCORE)

    def _rec_only(self, crop: np.ndarray, profile: str):
        """Recognition-only helper used by page-orientation probes."""
        kwargs = {"use_det": False, "use_cls": False, "use_rec": True,
                  "text_score": 0.0 if profile == PROFILE_PRESCRIPTION else LABEL_TEXT_SCORE}
        return self._main(crop, **kwargs)

    def _upright(self, bgr: np.ndarray, profile: str = PROFILE_LABEL):
        """(image turned upright, OCR result, degrees turned). A cheap detection-only pass first: when most
        text boxes are tall the photo is sideways, so the quarter turns are read (stopping at the first one
        that reads clearly). Otherwise a low-confidence reading tries a half turn (upside down).

        Page-level rotation is separate from the per-line classifier. The latter is off in the prescription
        profile; this method only decides whether the whole page should be turned."""
        self._check_profile(profile)
        det = self._main(bgr, use_det=True, use_cls=False, use_rec=False)
        quads = [np.asarray(q) for q in (det.boxes if det.boxes is not None else [])]
        tall = sum(1 for q in quads if (q[:, 1].max() - q[:, 1].min()) > 1.5 * (q[:, 0].max() - q[:, 0].min()))
        mean = self._mean_quality
        if quads and tall > 0.5 * len(quads):
            # Sideways: read the largest boxes turned both ways (recognition only, cheap) and pick the better turn.
            big = sorted(quads, key=lambda q: -(np.ptp(q[:, 0]) * np.ptp(q[:, 1])))[:SAMPLE_BOXES]
            scores = {}
            for code, deg in ((cv2.ROTATE_90_CLOCKWISE, 90), (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)):
                vals = []
                for q in big:
                    x0, y0 = max(0, int(q[:, 0].min())), max(0, int(q[:, 1].min()))
                    crop = cv2.rotate(bgr[y0:int(q[:, 1].max()) + 1, x0:int(q[:, 0].max()) + 1], code)
                    r = self._rec_only(crop, profile)
                    vals.append(float(r.scores[0]) if r.txts else 0.0)
                scores[deg] = (float(np.mean(vals)), code)
            deg = max(scores, key=lambda d: scores[d][0])
            turned = cv2.rotate(bgr, scores[deg][1])
            return turned, self._full(turned, profile), deg
        res = self._full(bgr, profile)
        if res.txts and mean(res, profile) < UPSIDE_DOWN_BELOW:
            turned = cv2.rotate(bgr, cv2.ROTATE_180)
            r = self._full(turned, profile)
            if mean(r, profile) > mean(res, profile) + 0.05:
                return turned, r, 180
        return bgr, res, 0

    @staticmethod
    def _mean_quality(res, profile: str) -> float:
        """Page-orientation quality. Label profile keeps the old mean of all recognition scores. The
        prescription profile ignores low-score noise for this decision only (the candidates themselves are
        still kept); otherwise a flood of 0.1-0.4 candidates can make an upright page look upside down."""
        scores = [float(s) for s in (getattr(res, "scores", None) or ()) if str(s).strip()]
        if not scores:
            return 0.0
        if profile == PROFILE_LABEL:
            return float(np.mean(scores))
        strong = [s for s in scores if s >= 0.5]
        return float(np.mean(strong)) if strong else 0.0

    def read(self, bgr: np.ndarray, hindi: Optional[bool] = None, profile: str = PROFILE_LABEL,
             preprocessing: Optional[str] = None) -> Dict[str, Any]:
        """hindi=None -> the engine's setting; False skips the Devanagari pass (food labels).
        profile='label' keeps the old behaviour. profile='prescription' keeps low-score candidates and
        turns off the per-box orientation classifier. preprocessing is only metadata on candidates."""
        t = time.perf_counter()
        self._check_profile(profile)
        if not self.load():
            return {"ok": False, "rows": [], "boxes": [], "lines": [], "profile": profile,
                    "ms": 0, "error": self.error}
        bgr, res, rotated = self._upright(bgr, profile)
        boxes = []
        quads = res.boxes if res.boxes is not None else []
        for quad, text, score in zip(quads, res.txts or (), res.scores or ()):
            text, score = str(text).strip(), float(score)
            q = np.asarray(quad)
            candidates = [{"text": text, "confidence": round(score, 3),
                           "engine": f"RapidOCR {self.models}", "orientation": str(rotated)}]
            selected = 0
            if self._hi is not None and hindi is not False:
                hi_kwargs = {"use_det": False, "use_cls": False, "use_rec": True,
                             "text_score": 0.0 if profile == PROFILE_PRESCRIPTION else LABEL_TEXT_SCORE}
                hi = self._hi(_crop(bgr, q, rotate_tall=False), **hi_kwargs)
                if hi.txts and str(hi.txts[0]).strip():
                    h_text, h_score = str(hi.txts[0]).strip(), float(hi.scores[0])
                    candidates.append({"text": h_text, "confidence": round(h_score, 3),
                                       "engine": "RapidOCR Devanagari", "orientation": str(rotated)})
                    if h_score > score + 0.05:   # Devanagari reading is clearly better
                        text, score, selected = h_text, h_score, len(candidates) - 1
            if not text:
                continue                       # truly empty recognition is not a candidate
            boxes.append({
                "id": f"B{len(boxes) + 1}",
                "text": text,
                "raw": text,                   # selected OCR before any cleanup in the pipeline
                "confidence": round(score, 3),
                "box": [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())],
                "engine": candidates[selected]["engine"],
                "orientation": str(rotated),
                "preprocessing": preprocessing or "",
                "selected": selected,
                "candidates": candidates,
            })
        rows = group_rows(boxes)
        for n, row in enumerate(rows, 1):
            row["id"] = f"R{n}"
        return {"ok": bool(rows), "rows": rows, "boxes": boxes, "lines": [r["text"] for r in rows],
                "profile": profile, "rotated": rotated, "ms": int((time.perf_counter() - t) * 1000),
                "error": None if rows else "no text found"}


_DEFAULT: Optional[OCREngine] = None


def default_engine() -> OCREngine:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OCREngine()
    return _DEFAULT
