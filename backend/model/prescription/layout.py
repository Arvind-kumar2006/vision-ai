"""Spatial row reconstruction and OCR-candidate arbitration for prescriptions (Phase 2A).

Phase 1 keeps every non-empty OCR reading, including low-confidence and Devanagari alternatives. This
module decides how those readings become medicine rows. It never calls the medicine database, never
invents text, and never uses fixed pixel coordinates: every threshold is relative to the page width or
the median line height.

The output rows keep the fields the existing redaction/block/LLM path expects (``text``, ``raw``,
``confidence``, ``box``, ``boxes``), plus internal evidence:

    evidence.arbitration   scored candidate readings for each kept box
    evidence.excluded      low-information boxes kept for inspection but not put in the row text
    evidence.group         the geometry used to form the row

Only the existing text/confidence fields go to the LLM. No image bytes or crops are stored.
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# All thresholds are relative: page-width fractions or multiples of the median box height.
VERTICAL_OVERLAP = 0.5       # a box joins a line when it overlaps by half the smaller box height
CENTER_DISTANCE = 0.30       # or when its centre is this close to the line's centre (line heights)
MARGIN_X = 0.6               # a box starting right of this page fraction may be a margin note
MARGIN_GAP = 2.0             # ... if the gap from the previous box is this many line heights
PREFIX_CONFIDENCE = 0.35     # below this a 1-2 character box is not trusted as medicine text
VERIFY_SCORE_GAP = 0.07      # close-scoring candidates that disagree -> verify, not silent selection
VERIFY_CONFIDENCE = 0.6      # confidence ceiling for a row whose selected reading needs verification

FORM_WORD = re.compile(
    r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syp|syr|syrup|inj|injection|drops?|oint|"
    r"ointment|cream|gel|sachet)\b|^\s*[tc]\s*[.:]\s*(?=[a-z](?![.]))", re.I)
STRENGTH = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu)\b", re.I)
# Boundary-only evidence for OCR-corrupted units such as ``50ms``. This does not rewrite the strength or
# participate in timing extraction; it only prevents a recognizable medicine row from being treated as prose.
STRENGTH_LIKE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|ms|mgs?)\b", re.I)
FREQ_CODE = re.compile(r"\b(?:OD|BD|BID|TDS|TID|QID|HS|SOS)\b", re.I)
SLOT = r"(?:[0-4]|½|1/2|[\u2460-\u2463]|[xX×oO])"
SLOT_GRID = re.compile(rf"(?<![\w/.]){SLOT}(?:\s*[-—–_~.=]+\s*|\s+){SLOT}(?:\s*[-—–_~.=]+\s*|\s+){SLOT}(?![0-9/.])")
DURATION = re.compile(r"(?:\b|(?<=[x×]))\d+(?:\.\d+)?\s*(?:d|dy|day|days|w|wk|wks|week|weeks|m|mo|mon|mons|month|months|mth|mths|"
                      r"y|yr|yrs|year|years)\b|(?:\b|(?<=[x×]))\d+\s*[dwmy][a-z]{0,5}\b|\b(?:one|two|three|four|five|six|seven|"
                      r"eight|nine|ten|eleven|twelve|fifteen|thirty)\s+(?:day|days|week|weeks|month|months|year|years)\b",
                       re.I)
MARKER = re.compile(r"^\s*(?:[\u2460-\u2473]|\(?\d{1,2}[).]|\d{1,2}\s*-\s*(?=[A-Za-z]))\s*$")
NUMERIC_MARKER = re.compile(r"^\s*\d{1,2}\s*$")
MARKER_PREFIX = re.compile(r"^\s*(?:[\u2460-\u2473]|\(?\d{1,2}[).]|\d{1,2}\s*-\s*|\d{1,2}(?=\s+[A-Za-z]))\s*")
FORM_PREFIX = re.compile(
    r"^\s*(?:tablets?|tabs?|capsules?|caps?|syrup|syp|syr|injection|inj|drops?|ointment|oint|cream|gel|sachet)"
    r"(?:\b[.:,]?\s*|(?=[A-Z]))|^\s*[tc]\s*[.:]\s*(?![tc]\s*[.:])(?=[a-z])",
    re.I,
)
CONTINUATION_WORDS = {
    "after", "before", "breakfast", "dinner", "evening", "food", "morning", "night", "noon",
    "review", "continue", "advice", "between", "daily", "needed", "times", "with",
    "meal", "meals", "massage", "apply", "local", "application", "once", "twice", "thrice", "then",
}
# Dosage forms written after the name ("Hexigel gum paint", "Colgate Plax mouthwash", "Nasoclear nasal drops").
FORM_ANYWHERE = re.compile(r"\b(?:tablet|capsule|syrup|syp|injection|inj|drops?|ointment|cream|gel|powder|lotion|spray|"
                           r"inhaler|respules?|suppository|supp|gargles?|mouth\s*wash|gum\s*pa[a-z]{1,3}|sachet|suspension)\b",
                           re.I)
# Section words written in front of the first medicine on the same line: "Rx, Cap. X", "Adv: Hexigel gum paint".
SECTION_PREFIX = re.compile(r"^\s*(?:r\s*x|℞|adv|advice|advised)\s*[.,:;-]*\s+(?=\S)", re.I)
ALNUM = re.compile(r"[a-z0-9]+", re.I)


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _alnum(text: str) -> str:
    return "".join(ALNUM.findall((text or "").lower()))


def _valid_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return [x0, y0, x1, y1]


def line_height(boxes: Sequence[Dict[str, Any]]) -> float:
    """Median height of the boxes that carry geometry; 0 when there is no geometry."""
    hs = sorted(
        b["box"][3] - b["box"][1]
        for b in boxes
        if _valid_box(b.get("box")) is not None and b["box"][3] > b["box"][1]
    )
    return float(median(hs)) if hs else 0.0


def _structure(text: str) -> Tuple[float, List[str]]:
    """How medicine-like a reading is, without using any medicine name or the database."""
    t = text or ""
    score, why = 0.0, []
    if FORM_WORD.search(t):
        score += 0.30
        why.append("dosage form")
    if STRENGTH.search(t):
        score += 0.25
        why.append("strength")
    if FREQ_CODE.search(t) or SLOT_GRID.search(t):
        score += 0.25
        why.append("frequency")
    if DURATION.search(t):
        score += 0.20
        why.append("duration")
    if MARKER.match(t):
        score += 0.15
        why.append("numbered entry")
    words = [w for w in re.findall(r"[A-Za-z]{3,}", t) if w.lower() not in {"tab", "cap", "syp", "inj", "mg", "ml"}]
    if words:
        score += 0.10
        why.append("name-like word")
    return min(1.0, score), why


def _entry_parts(text: str) -> Tuple[str, bool, bool, bool]:
    """Return (candidate name text, numbered marker, form prefix, short prefix).

    The short-prefix signal is intentionally weak. It only contributes when the rest of the row has
    medicine-like evidence (normally a strength), so arbitrary instruction lines are not promoted to
    medicine entries.
    """
    t = _clean(text)
    t = re.sub(r"^\s*[\W_]+(?=\d|[A-Za-z\u2460-\u2473])", "", t, count=1)
    t = SECTION_PREFIX.sub("", t, count=1)
    # OCR often leaves a one-letter tick/bullet before the actual numbered/form prefix (``X 2) Tab``).
    t = re.sub(r"^\s*[A-Za-z]\s+(?=(?:\(?\d{1,2}[).]|\d{1,2}\s*-|(?:tab|cap|syp|syr|inj|capsule|tablet)\b))", "", t, count=1, flags=re.I)
    # Dates can precede a numbered medicine row. They are structural prefixes, not medicine names.
    t = re.sub(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*", "", t, count=1)
    numbered = bool(MARKER_PREFIX.match(t))
    if numbered:
        t = MARKER_PREFIX.sub("", t, count=1).strip()
    # Numeric OCR/date fragments before a recognized form are structural prefixes, not names.
    parts = t.split()
    while len(parts) > 1 and not FORM_PREFIX.match(t) and re.fullmatch(r"[\d|/._-]+", parts[0] or ""):
        parts.pop(0)
        t = " ".join(parts)
    form = bool(FORM_PREFIX.match(t))
    if form:
        t = FORM_PREFIX.sub("", t, count=1).strip()
    short_prefix = False
    first, _, rest = t.partition(" ")
    # Only when a name follows ("X flagyl 400"). "Pan 40mg" / "Azi 500mg": the short word IS the name.
    if rest and re.fullmatch(r"[A-Za-z]{1,3}[.:]?,?", first) and re.match(r"[A-Za-z]{3,}", rest.strip()) \
            and (STRENGTH_LIKE.search(t) or (not form and not numbered and FORM_PREFIX.match(rest.strip()))):
        short_prefix = True
        t = rest.strip()
        if FORM_PREFIX.match(t):          # "afe Tab. Augmentin 625mg": a margin word, then the dosage form
            form, short_prefix = True, False
            t = FORM_PREFIX.sub("", t, count=1).strip()
    return t, numbered, form, short_prefix


def entry_boundary_decision(row: Dict[str, Any], previous: Optional[Dict[str, Any]] = None,
                            line_h: float = 0.0, page_width: float = 0.0) -> Dict[str, Any]:
    """Explain whether a reconstructed row opens a medicine entry.

    This is deliberately not a name extractor and never consults the medicine database. A new-entry
    decision requires combined evidence: a marker/form or strength, plus name-like content. Spatial
    separation and OCR confidence are supporting signals only. ``verify`` is returned for ambiguous
    rows so callers can retain the row without silently forcing a new block.
    """
    text = _clean(row.get("main_text") or row.get("text"))
    candidate, numbered, form, short_prefix = _entry_parts(text)
    lower_words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", candidate)]
    name_words = [w for w in lower_words if w not in CONTINUATION_WORDS and w not in {
        "tablet", "tablets", "capsule", "capsules", "syrup", "syr", "injection", "powder", "months",
        "month", "weeks", "week", "days", "day", "years", "year", "mg", "mcg", "ml",
    }]
    strength = bool(STRENGTH_LIKE.search(text))
    frequency = bool(FREQ_CODE.search(text) or SLOT_GRID.search(text))
    duration = bool(DURATION.search(text))
    # Handwritten rows often lose the unit/prefix in OCR (e.g. ``Aduent 0.81``
    # or ``Hirzole cream``).  These are row-level signals only; the candidate is
    # still shown for verification and is never treated as a confirmed medicine.
    numeric_row = bool(re.search(r"\d", text)) and bool(name_words) and not duration
    form_anywhere = bool(FORM_ANYWHERE.search(text)) and bool(name_words)
    continuation_only = not name_words and not strength and not form and not numbered and (frequency or duration)

    score = 0.0
    reasons: List[str] = []
    if form:
        score += 0.42
        reasons.append("dosage-form prefix")
    if numbered:
        score += 0.25
        reasons.append("numbered marker")
    if short_prefix:
        score += 0.10
        reasons.append("short OCR prefix with strength")
    if name_words:
        score += 0.24
        reasons.append("name-like content")
    if strength:
        score += 0.25
        reasons.append("strength")
    if frequency:
        score += 0.12
        reasons.append("frequency/dose pattern")
    if duration and not name_words:
        score -= 0.16
        reasons.append("duration-like continuation")
    if continuation_only:
        score -= 0.20
        reasons.append("continuation-only pattern")

    if previous and line_h and row.get("box") and previous.get("box"):
        gap = float(row["box"][1]) - float(previous["box"][3])
        relative_gap = gap / line_h
        if relative_gap >= 0.75:
            score += 0.10
            reasons.append("vertical separation")
        elif relative_gap < -0.25:
            score -= 0.05
            reasons.append("overlapping neighboring row")
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence >= 0.75:
        score += 0.05
        reasons.append("OCR confidence")

    score = round(max(0.0, min(1.0, score)), 3)
    # Explicit dosage forms retain the established behavior. Other rows need combined evidence;
    # a close score is exposed as verify instead of becoming a new block silently.
    if form and not re.match(r"^\s*on\s+t\w{1,2}\b", text, re.I):
        decision = "new medicine"
    elif (numeric_row or form_anywhere) and not continuation_only:
        decision = "new medicine"
        reasons.append("generic medicine-row evidence")
    elif score >= 0.52 and (name_words and (strength or numbered or short_prefix)):
        decision = "new medicine"
    elif score >= 0.40 and name_words and (strength or numbered or short_prefix or frequency):
        decision = "verify"
    else:
        decision = "continuation"
    return {
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "evidence": {
            "text": text,
            "candidate": candidate,
            "numbered": numbered,
            "form": form,
            "short_prefix": short_prefix,
            "name_words": name_words,
            "strength": strength,
            "frequency": frequency,
            "duration": duration,
            "line_height": round(float(line_h or 0.0), 3),
            "page_width": round(float(page_width or 0.0), 3),
        },
    }


def _similarity(a: str, b: str) -> float:
    a, b = _alnum(a), _alnum(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _agreement(candidates: Sequence[Dict[str, Any]], index: int) -> float:
    """Best agreement with another reading. One candidate has no second opinion -> 0."""
    text = candidates[index]["text"]
    return max((_similarity(text, c["text"]) for i, c in enumerate(candidates) if i != index and _alnum(c["text"])),
               default=0.0)


def arbitrate_box(box: Dict[str, Any], normalize: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Choose one reading while keeping every reading and the reason for the choice.

    The score is deliberately explainable: OCR confidence first, then medicine-like structure, then
    agreement with another reading. Confidence alone is never sufficient. A close disagreement is
    marked ``verify`` and lowers the effective confidence instead of silently picking a winner.
    """
    raw_candidates = box.get("candidates") or []
    if not raw_candidates:
        raw_candidates = [{"text": box.get("text"), "confidence": box.get("confidence"),
                           "engine": box.get("engine"), "orientation": box.get("orientation")}]
    candidates: List[Dict[str, Any]] = []
    for c in raw_candidates:
        raw = _clean(c.get("raw", c.get("text")))
        if not raw:
            continue
        text = _clean(normalize(raw)) if normalize else raw
        try:
            conf = max(0.0, min(1.0, float(c.get("confidence", box.get("confidence", 0.0)) or 0.0)))
        except (TypeError, ValueError):
            conf = 0.0
        structure, why = _structure(text)
        candidates.append({
            "text": text,
            "raw": raw,
            "confidence": round(conf, 3),
            "engine": c.get("engine") or box.get("engine") or "",
            "orientation": str(c.get("orientation", box.get("orientation", ""))),
            "structure": round(structure, 3),
            "reasons": why,
        })
    if not candidates:
        candidates = [{"text": "", "raw": "", "confidence": 0.0, "engine": box.get("engine") or "",
                       "orientation": str(box.get("orientation", "")), "structure": 0.0, "reasons": []}]

    for i, c in enumerate(candidates):
        c["agreement"] = round(_agreement(candidates, i), 3)
        low_info = len(_alnum(c["text"])) <= 2 and c["structure"] == 0.0
        c["low_information"] = low_info
        c["score"] = round(c["confidence"] + 0.12 * c["structure"] + 0.06 * c["agreement"]
                           - (0.25 if low_info else 0.0), 3)

    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]["score"], -candidates[i]["confidence"], i))
    selected_i = order[0]
    selected = candidates[selected_i]
    second = candidates[order[1]] if len(order) > 1 else None
    close = bool(second and selected["score"] - second["score"] < VERIFY_SCORE_GAP)
    disagree = bool(second and _similarity(selected["text"], second["text"]) < 0.75)
    verify = selected["low_information"] or (close and disagree)
    for i, c in enumerate(candidates):
        c["selected"] = i == selected_i
        c["score"] = round(c["score"], 3)

    effective = selected["confidence"]
    reasons = list(selected["reasons"])
    if verify:
        effective = min(effective, VERIFY_CONFIDENCE)
        reasons.append("close/disagreeing evidence" if close and disagree else "low-information reading")
    if second and second["score"] < selected["score"] and second["confidence"] > selected["confidence"]:
        reasons.append("structure beat a slightly higher OCR score")

    return {
        "id": box.get("id"),
        "text": selected["text"],
        "raw": selected["raw"],
        "confidence": round(effective, 3),
        "ocr_confidence": selected["confidence"],
        "box": list(box["box"]) if _valid_box(box.get("box")) else box.get("box"),
        "engine": selected["engine"],
        "orientation": selected["orientation"],
        "preprocessing": box.get("preprocessing") or "",
        "selected": selected_i,
        "candidates": candidates,
        "verify": verify,
        "reasons": reasons,
    }


def _group_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Boxes into visual lines using vertical overlap or centre distance, never x/y constants."""
    if not boxes:
        return []
    lh = line_height(boxes) or 1.0
    groups: List[Dict[str, Any]] = []
    for b in sorted(boxes, key=lambda b: ((b["box"][1] + b["box"][3]) / 2, b["box"][0])):
        x0, y0, x1, y1 = b["box"]
        cy, h = (y0 + y1) / 2, y1 - y0
        best = None
        for g in groups:
            gy0, gy1 = g["y0"], g["y1"]
            overlap = min(y1, gy1) - max(y0, gy0)
            min_h = min(h, gy1 - gy0)
            centre = abs(cy - (gy0 + gy1) / 2)
            if (min_h > 0 and overlap >= VERTICAL_OVERLAP * min_h) or centre <= CENTER_DISTANCE * lh:
                affinity = (overlap / min_h if min_h > 0 else 0.0) + max(0.0, 1.0 - centre / lh) * 0.2
                if best is None or affinity > best[0]:
                    best = (affinity, g)
        if best is None:
            groups.append({"y0": y0, "y1": y1, "boxes": [b]})
        else:
            best[1]["boxes"].append(b)
    for g in groups:
        g["boxes"].sort(key=lambda b: b["box"][0])
    return sorted(groups, key=lambda g: g["y0"])


def split_row_boxes(boxes: Sequence[Dict[str, Any]], page_width: float, lh: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(main-column boxes, margin boxes) for one visual row. Thresholds are page-relative."""
    ordered = sorted(boxes, key=lambda b: b["box"][0])
    if not ordered or not page_width or not lh:
        return list(ordered), []
    main, margin = [], []
    prev_x1: Optional[float] = None
    for b in ordered:
        x0 = b["box"][0]
        far = prev_x1 is None or x0 - prev_x1 > MARGIN_GAP * lh
        if margin or (x0 >= MARGIN_X * page_width and far):
            margin.append(b)
        else:
            main.append(b)
        prev_x1 = b["box"][2]
    return main, margin


def content_confidence(boxes: Sequence[Dict[str, Any]], fallback: float = 0.0) -> float:
    values = [float(b.get("confidence") or 0.0) for b in boxes
              if b.get("role") not in ("marker", "noise") and float(b.get("confidence") or 0.0) > 0]
    if not values:
        values = [float(b.get("confidence") or 0.0) for b in boxes if float(b.get("confidence") or 0.0) > 0]
    return round(sum(values) / len(values), 3) if values else round(float(fallback or 0.0), 3)


def _row_from_group(group: Dict[str, Any], page_width: float, lh: float, index: int) -> Dict[str, Any]:
    boxes = group["boxes"]
    # A low-confidence, one/two-character box is evidence, but not medicine text. A numbered marker is
    # the exception: it can start an entry, so it stays in the text while being kept out of confidence.
    for b in boxes:
        b["role"] = "content"
        text = b.get("text") or ""
        if len(_alnum(text)) <= 2 and float(b.get("confidence") or 0) < PREFIX_CONFIDENCE:
            b["role"] = "marker" if MARKER.match(text) or NUMERIC_MARKER.match(text) else "noise"
    included = [b for b in boxes if b["role"] != "noise"]
    excluded = [b for b in boxes if b["role"] == "noise"]
    main_boxes, margin_boxes = split_row_boxes(included, page_width, lh)
    text = " ".join(b["text"] for b in included).strip()
    raw = " ".join(b.get("raw", b["text"]) for b in included).strip()
    main_text = " ".join(b["text"] for b in main_boxes).strip()
    margin_text = " ".join(b["text"] for b in margin_boxes).strip()
    confidence = content_confidence(main_boxes) if main_boxes else content_confidence(included)
    all_boxes = included + excluded
    return {
        "id": f"L{index}",
        "text": text,
        "raw": raw,
        "confidence": confidence,
        "box": [min(b["box"][0] for b in all_boxes), min(b["box"][1] for b in all_boxes),
                max(b["box"][2] for b in all_boxes), max(b["box"][3] for b in all_boxes)],
        "boxes": included,
        "main_boxes": main_boxes,
        "margin_boxes": margin_boxes,
        "main_text": main_text,
        "margin_text": margin_text,
        "evidence": {
            "group": {"y0": group["y0"], "y1": group["y1"], "line_height": round(lh, 2),
                      "page_width": round(float(page_width or 0), 2)},
            "arbitration": [{"id": b.get("id"), "selected": b.get("selected"), "verify": b.get("verify"),
                             "reasons": b.get("reasons"), "candidates": b.get("candidates")} for b in included],
            "excluded": [{"id": b.get("id"), "text": b.get("text"), "confidence": b.get("confidence"),
                          "box": b.get("box"), "reasons": b.get("reasons")} for b in excluded],
        },
    }


def _fallback_rows(rows: Sequence[Dict[str, Any]], normalize: Optional[Callable[[str], str]]) -> List[Dict[str, Any]]:
    """Rows without box geometry (unit tests, or an engine that does not expose boxes) still work."""
    out = []
    for i, r in enumerate(rows, 1):
        raw = _clean(r.get("raw", r.get("text")))
        text = _clean(normalize(raw)) if normalize else raw
        boxes = []
        for b in r.get("boxes") or []:
            b = dict(b)
            b_raw = _clean(b.get("raw", b.get("text")))
            b["raw"] = b_raw
            b["text"] = _clean(normalize(b_raw)) if normalize else b_raw
            b.setdefault("role", "content")
            boxes.append(b)
        out.append({**r, "id": r.get("id") or f"L{i}", "raw": raw, "text": text, "boxes": boxes,
                    "evidence": {"group": {"line_height": 0.0, "page_width": 0.0}, "arbitration": [],
                                 "excluded": []}})
    return out


def reconstruct_rows(boxes: Sequence[Dict[str, Any]], page_width: float,
                     normalize: Optional[Callable[[str], str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """OCR boxes (or already-rowed items) -> (spatially reconstructed rows, phase timings).

    ``boxes`` is normally ``ocr["boxes"]``. If it contains row dicts (items with a ``boxes`` key) or
    has no usable geometry, the function falls back to those rows so existing callers/tests keep working.
    """
    t0 = time.perf_counter()
    items = list(boxes or [])
    if not items:
        return [], {"layout_ms": 0, "arbitration_ms": 0, "group_ms": 0}
    if any("boxes" in item for item in items):
        return _fallback_rows(items, normalize), {
            "layout_ms": int((time.perf_counter() - t0) * 1000), "arbitration_ms": 0,
            "group_ms": int((time.perf_counter() - t0) * 1000)}

    t = time.perf_counter()
    evidence = [arbitrate_box(b, normalize) for b in items if _valid_box(b.get("box")) is not None]
    arbitration_ms = int((time.perf_counter() - t) * 1000)
    if not evidence:
        return [], {"layout_ms": int((time.perf_counter() - t0) * 1000), "arbitration_ms": arbitration_ms,
                    "group_ms": 0}

    t = time.perf_counter()
    lh = line_height(evidence) or 1.0
    rows = [_row_from_group(g, page_width, lh, i) for i, g in enumerate(_group_boxes(evidence), 1)]
    group_ms = int((time.perf_counter() - t) * 1000)
    return rows, {"layout_ms": int((time.perf_counter() - t0) * 1000), "arbitration_ms": arbitration_ms,
                  "group_ms": group_ms}
