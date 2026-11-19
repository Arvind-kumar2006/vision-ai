"""Prescription pipeline (ADR-005):

    image -> preprocessing (common/preprocess.py) -> RapidOCR, English + Devanagari (common/ocr.py)
          -> no text: "Couldn't read this photo. Try a flatter, brighter photo."
          -> redact header / patient / doctor lines (common/redact.py; they never leave this machine)
          -> AI text model lists the medicines: name as written, alternatives, strength, form (extract.py;
             OpenRouter free, Groq backup). Both busy: names from the OCR lines only + "AI busy ..."
          -> Indian medicine dataset (medicine_db.py): rapidfuzz match, strength narrows the variant,
             matched / verify / not found, "What it's for" by ingredient (uses.py)
          -> PrescriptionResult

Goal (2026-11-17): identify each medicine correctly, with its strength and what it is for. How often, how long,
dose counts, dates and follow-up are not extracted.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from model.common.llm import AI_BUSY_NOTE, LLM, AIBusy, default_llm
from model.common.ocr import OCREngine, default_engine
from model.common.preprocess import PRESCRIPTION_STEPS, preprocess, prescription_variants
from model.common.redact import MEDICINE_START, lines_for_ai

from .extract import (basic_medicines, build_blocks, extract, fix_ocr, ground, layout_candidates, missed_by_ai,
                      missed_line_gaps, split_merged_entries, supplement_layout_candidates)
from .layout import reconstruct_rows
from .medicine_db import match_all
from .schemas import MedicineItem, OcrLine, PrescriptionResult

log = logging.getLogger("prescription.pipeline")

UNREADABLE = "Couldn't read this photo. Try a flatter, brighter photo."
NO_MEDICINE_LINES = ("Text was read, but no medicine lines were recognised. Medicines are found from lines that start "
                     "with Tab, Cap, Syp, Inj, T., C. or a number like 1). Try a closer, flatter photo of the medicines.")
PRIVACY_NOTE = "Patient and doctor details stay on this machine; only the medicine lines are sent to the AI."
CHECK_BELOW = 0.85          # OCR confidence of a medicine's line below this -> its strength is marked "check"


def _source_row(med: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from .extract import grounding_score
    scored = [(max(grounding_score(n, r["text"]) for n in [med.get("name", ""), *(med.get("alternatives") or [])]), r)
              for r in rows]
    best = max(scored, key=lambda x: x[0], default=(0, None))
    return best[1] if best[0] >= 0.6 else None


def mark_checks(med: Dict[str, Any], rows: List[Dict[str, Any]], ocr_text: str) -> None:
    """The strength is marked "check" when its line was read with low confidence, or when its number is not in
    the text that was read."""
    row = _source_row(med, rows)
    low = row is None or float(row.get("confidence") or 0) < CHECK_BELOW
    digits_in_text = set(re.findall(r"\d+(?:\.\d+)?", ocr_text))
    checks = [c for c in med.get("checks") or [] if c == "name"]
    nums = re.findall(r"\d+(?:\.\d+)?", str(med.get("strength") or ""))
    if nums and (low or any(n not in digits_in_text for n in nums)):
        checks.append("strength")
    med["checks"] = checks
    med["as_written"] = row["text"] if row else None
    med["ocr_confidence"] = round(float(row.get("confidence") or 0), 2) if row else None


class PrescriptionPipeline:
    def __init__(self, ocr: Optional[OCREngine] = None, llm: Optional[LLM] = None, steps=PRESCRIPTION_STEPS) -> None:
        self.ocr = ocr or default_engine()
        self.llm = llm if llm is not None else default_llm()
        self.steps = tuple(steps)

    def run(self, image) -> PrescriptionResult:
        t0 = time.perf_counter()
        timings: Dict[str, int] = {}
        rid = uuid.uuid4().hex[:12]
        bgr, info = preprocess(image, self.steps)
        timings["preprocess_ms"] = int((time.perf_counter() - t0) * 1000)
        ocr = self.ocr.read(bgr, hindi=True, profile="prescription",
                            preprocessing="+".join(self.steps))
        timings["ocr_ms"] = ocr.get("ms", 0)
        # Handwritten pages are especially sensitive to shadows and faint ink.  Do
        # not pay for extra passes on normal images; retry only when the first OCR
        # result has no medicine-section evidence.  The alternatives are local
        # image transforms, not guesses, and the selected OCR still goes through
        # the same redaction and grounding checks below.
        if ocr.get("ok"):
            initial_rows, _ = lines_for_ai(ocr.get("rows") or [])
        else:
            initial_rows = []
        if not initial_rows:
            best = ocr
            best_score = self._ocr_evidence(best)
            for name, variant in prescription_variants(bgr):
                candidate = self.ocr.read(variant, hindi=True, profile="prescription",
                                          preprocessing="+".join(self.steps) + "+" + name)
                score = self._ocr_evidence(candidate)
                if score > best_score:
                    best, best_score = candidate, score
            if best is not ocr:
                ocr = best
                timings["ocr_ms"] = int(timings["ocr_ms"] + best.get("ms", 0))
        rows = ocr["rows"]
        if ocr["ok"]:
            # Phase 2A: rebuild visual rows from the OCR boxes, arbitrate their candidate readings,
            # and keep low-information boxes as evidence instead of medicine text.
            t = time.perf_counter()
            rows, layout_stats = reconstruct_rows(ocr.get("boxes") or rows, bgr.shape[1], normalize=fix_ocr)
            recovered = self._recover_missed_lines(bgr, ocr, rows)
            if recovered:
                ocr = {**ocr, "boxes": ocr["boxes"] + recovered}
                rows, layout_stats = reconstruct_rows(ocr["boxes"], bgr.shape[1], normalize=fix_ocr)
                timings["recover_ms"] = int((time.perf_counter() - t) * 1000)
            timings["layout_ms"] = layout_stats["layout_ms"]
            timings["arbitration_ms"] = layout_stats["arbitration_ms"]
            timings["group_ms"] = layout_stats["group_ms"]
        base = dict(id=rid, preprocessing=info["steps"], ocr_engine=self.ocr.name)
        if not ocr["ok"]:
            timings["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return PrescriptionResult(**base, status="unreadable", message=UNREADABLE, timings=timings,
                                      review_notes=[UNREADABLE])
        med_rows, withheld = lines_for_ai(rows)
        med_rows = split_merged_entries(med_rows)
        sent = {id(r.get("_src", r)) for r in med_rows}
        ocr_lines = [OcrLine(text=r["raw"], confidence=r["confidence"], sent_to_ai=id(r) in sent) for r in rows]
        ocr_text = "\n".join(r["text"] for r in med_rows)
        raw_text = "\n".join(r["raw"] for r in med_rows)   # numbers are checked against what was actually read
        notes: List[str] = [PRIVACY_NOTE]
        status, message, ai = "ok", None, None
        extracted: Dict[str, Any] = {"medicines": [], "already_taking": []}
        if not med_rows:
            notes.append(NO_MEDICINE_LINES)
            status, message = "no_medicines", NO_MEDICINE_LINES
        else:
            t = time.perf_counter()
            layout = build_blocks(med_rows, bgr.shape[1])
            try:
                extracted = extract(self.llm, layout)
                ai = extracted.pop("ai")
                listed = extracted["medicines"] + [{"name": t} for t in extracted["already_taking"]]
                for m in missed_by_ai(listed, med_rows):   # safety net: never drop a known medicine
                    extracted["medicines"].append(m)
                    notes.append(f"“{m['name']}” is in the text but the AI didn't list it. Check it on the prescription.")
                for m in supplement_layout_candidates(extracted["medicines"], med_rows):
                    extracted["medicines"].append(m)
                    notes.append(f"“{m['name']}” is a medicine-like line the AI could not structure. Check it on the prescription.")
            except (AIBusy, ValueError) as exc:
                log.info("Prescription %s: AI not used (%s)", rid, exc)
                meds, taking = basic_medicines(med_rows)
                meds += supplement_layout_candidates(meds, med_rows)
                extracted.update(medicines=meds, already_taking=taking)
                status, message = "ai_busy", AI_BUSY_NOTE
                notes.append(AI_BUSY_NOTE)
            timings["ai_ms"] = int((time.perf_counter() - t) * 1000)
        meds = extracted["medicines"]
        notes += ground(meds, ocr_text)
        # If the AI supplied a medicine name that is not grounded in the OCR, do
        # not let it replace visible rows. When layout produced enough candidates,
        # remove the unsupported AI row and keep the OCR-spelled Verify rows. This
        # prevents plausible but unrelated database matches (for example
        # Levofloxacin) from hiding the handwritten medicine that was actually read.
        ungrounded = [m for m in meds if m.get("in_ocr_text") is False]
        if ungrounded and len(layout_candidates(med_rows)) >= len(meds):
            meds = [m for m in meds if m.get("in_ocr_text") is not False]
            meds += supplement_layout_candidates(meds, med_rows)
            notes.append("An AI medicine name was not found in the OCR text; the visible OCR rows were kept for verification.")
        for m in meds:
            mark_checks(m, med_rows, raw_text)
            m.pop("block", None)
        t = time.perf_counter()
        matched, match_notes = match_all(meds) if meds else ([], [])
        timings["match_ms"] = int((time.perf_counter() - t) * 1000)
        notes += match_notes
        for m in matched:
            # A name the AI gave that is not in the text that was read is never shown as confirmed, however well
            # it matches the database ("Risperidone" on a page where nothing like it was written).
            if m.get("in_ocr_text") is False and m.get("match_status") in ("matched", "known_brand"):
                m["match_status"] = "verify"
                m["confidence_level"] = "low"
        if any("strength" in (m.get("checks") or []) for m in matched):
            notes.append("Strengths marked “check” were hard to read. Compare them with the prescription.")
        if med_rows and not meds and status == "ok":
            notes.append(NO_MEDICINE_LINES)
        timings["total_ms"] = int((time.perf_counter() - t0) * 1000)
        result = PrescriptionResult(
            **base, status=status, message=message,
            medicines=[MedicineItem(**m) for m in matched], already_taking=extracted["already_taking"],
            review_notes=list(dict.fromkeys(notes)), ai=ai, ocr_lines=ocr_lines, lines_sent_to_ai=len(med_rows) if ai else 0,
            lines_kept_private=len(rows) - (len(med_rows) if ai else 0), timings=timings)
        log.info("Prescription %s: %d medicine(s), status=%s, %d ms (ocr %d, ai %d)", rid, len(matched), status,
                 timings["total_ms"], timings["ocr_ms"], timings.get("ai_ms", 0))
        return result

    MAX_RECOVERY_STRIPS = 3
    STRIP_PAD = 20

    def _recover_missed_lines(self, bgr, ocr: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-read the strip above an orphan schedule line (see missed_line_gaps). The detector sometimes skips a
        whole handwritten line on a full page but finds it in a narrow strip. Only medicine-style lines ("Tab. X",
        "1) X") are added; they then go through the same redaction, AI and grounding as every other line."""
        if ocr.get("rotated") or not ocr.get("boxes"):
            return []    # box coordinates are in the turned image; the strip would be cut from the wrong place
        import cv2
        out: List[Dict[str, Any]] = []
        for y0, y1 in missed_line_gaps(rows)[: self.MAX_RECOVERY_STRIPS]:
            top, bottom = max(0, int(y0)), min(bgr.shape[0], int(y1) + 1)
            if bottom - top < 8:
                continue
            pad = self.STRIP_PAD
            strip = cv2.copyMakeBorder(bgr[top:bottom], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            read = self.ocr.read(strip, hindi=False, profile="prescription",
                                 preprocessing="+".join(self.steps) + "+strip")
            for b in read.get("boxes") or []:
                if not MEDICINE_START.search(b["text"]):
                    continue
                x0, by0, x1, by1 = b["box"]
                out.append({**b, "id": f"S{len(out) + 1}",
                            "box": [x0 - pad, by0 - pad + top, x1 - pad, by1 - pad + top]})
        if out:
            log.info("Recovered %d line(s) the detector missed: %s", len(out), [b["text"] for b in out])
        return out

    @staticmethod
    def _ocr_evidence(ocr: Dict[str, Any]) -> tuple:
        """Rank OCR alternatives by prescription-like evidence, not medicine names."""
        rows = ocr.get("rows") or []
        text = "\n".join(str(r.get("text") or "") for r in rows)
        # These are format clues only.  Medicine names are deliberately not used,
        # so the fallback cannot favour a database name or invent a row.
        markers = len(re.findall(r"(?:\b(?:tab|cap|syp|inj|drop|cream|ointment)\b|(?:^|\n)\s*\d\s*[).])",
                                text, re.I))
        strengths = len(re.findall(r"\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu)\b", text, re.I))
        schedules = len(re.findall(r"\b\d\s*[-–—]\s*\d\s*[-–—]\s*\d\b|\b(?:OD|BD|TDS|QID|SOS|HS)\b",
                                  text, re.I))
        rows_with_text = sum(bool(str(r.get("text") or "").strip()) for r in rows)
        confidence = sum(float(r.get("confidence") or 0) for r in rows)
        return (markers * 100 + strengths * 20 + schedules * 20, rows_with_text, confidence)
