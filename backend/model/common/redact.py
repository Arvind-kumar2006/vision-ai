"""What leaves the machine: only the medicine section of a prescription, without personal or clinical details.

The page is read on this machine (RapidOCR, ADR-005). Before any text goes to the AI text model:

1. The medicine section starts at the first medicine marker ("Rx", "Tab", "1)") and ENDS at the first
   signature / footer line ("Dr ...", "Regd", "Signature", "In emergency", an address, phone or e-mail line).
2. A line with personal details (name titles, age, sex, weight, phone, address, registration, "Dr.") is
   withheld entirely, so a name split across two text boxes can never leak.
3. Clinical history notes (diagnoses, family / personal history: "c/o", "h/o", "known case", "Chr ...", "DM",
   "HTN", "mother expired", ...) are personal information too. They are often written in a margin on the same
   row as a medicine's timing, so they are removed box by box: the matching box and everything to its right on
   that row (margin notes sit right of the medicine column). Their numbers never reach the AI or the rules, so
   they can't be read as a duration.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

FORM = r"(?:tab|tabs|tablet|cap|caps|capsule|syp|syr|syrup|inj|injection|drops?|oint|ointment|cream|gel|sachet)"
# Bullets / brackets / dashes written before an entry: "- T. X", "[- T. X", "• C. X", "(1) Tab X"
LEAD = r"^[\s\-–—•·*>\[\](){}|.,:~_]*"
# Abbreviated dosage forms used on Indian prescriptions, only at the start of a line: "T." = tablet, "C." = capsule
ABBR_FORM = r"[tc]\s*[.:]\s*(?=[a-z](?![.]))"   # not "T.C." (an abbreviation like a lab test)
MEDICINE_START = re.compile(r"(?:\b" + FORM + r"\b\.?|\brx\b|℞|^\s*\d{1,2}\s*\)\s*\S|^\s*\d{1,2}\s*\.\s+(?!\d)\S|" + LEAD + ABBR_FORM + ")", re.I)
# A prescription row does not always contain a dosage-form marker.  These are
# evidence signals only; they never make a row a medicine by themselves.
PRESCRIPTION_EVIDENCE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b|"
    r"\b\d\s*[-–—]\s*\d\s*[-–—]\s*\d\b|"
    r"\b(?:od|bd|bid|tds|tid|qid|hs|sos)\b)", re.I)
PERSONAL = re.compile(
    r"\b(?:name|patient|age|aged|sex|gender|male|female|wt|weight|kgs?|yrs?|years?|mobile|mob|ph|phone|tel|"
    r"contact|address|addr|road|nagar|colony|street|district|dist|pin|reg|regd|registration|uhid|ipd|opd|"
    r"father|husband|w/o|s/o|d/o|c/o|mr|mrs|ms|smt|shri|sri|dr|doctor|consultant|regn)\b"
    r"|\bpt\s*(?:[:.]|name\b)"           # "Pt: ..." / "Pt name"; a bare "PT" is a variant ("Zerodol PT")
    r"|\b\d{2,3}\s*/\s*[mf]\b"            # "72/M"
    r"|\b\d{5}[-\s]?\d{5}\b|\b\d{10}\b"   # phone numbers
    r"|\b\d{6}\b",                        # PIN codes
    re.I)
# The medicine section ends at the signature / footer.
SECTION_END = re.compile(
    r"^\W*(?:dr\b\.?|regd\b|reg\.?\s*no\b|signature\b|sign\b|in\s+(?:case\s+of\s+)?emergency\b|emergency\b|"
    r"plot\s*(?:no\b|#)|address\b|consultant\b|regn\b|tel\b|telephone\b|phone\b|mobile\b|e-?mail\b|"
    r"please\s+bring|facility\s+available|rest\s+for\b)"
    r"|\b\d{5}[-\s]?\d{5}\b|\b\d{3,4}[-\s]\d{7,8}\b|\b[\w.]+@[\w.]+\.\w+\b",   # phone, landline, e-mail anywhere
    re.I)
# Clinical history: markers, relations, events, common diagnoses and their abbreviations.
CLINICAL_EXACT = {
    "h/o", "c/o", "k/c/o", "kco", "hx", "dx", "chr", "chronic", "history", "diagnosis", "impression", "known",
    "family", "past", "complaints", "mother", "father", "brother", "sister", "wife", "husband", "son", "daughter",
    "grandfather", "grandmother", "uncle", "aunt", "expired", "died", "death", "deceased", "suicide", "surgery",
    "operated", "admitted", "accident", "dm", "t2dm", "htn", "cad", "ihd", "ckd", "copd", "tb", "hiv", "bpad",
    "ocd", "mdd", "gad", "ptsd", "adhd",
    }
CLINICAL_LONG = ["schizophrenia", "depression", "bipolar", "psychosis", "epilepsy", "hypertension", "diabetes",
                 "hypothyroidism", "hyperthyroidism", "dementia", "parkinson", "migraine", "asthma", "tuberculosis",
                 "expired", "diagnosis", "history", "chronic", "grandmother", "grandfather", "relieved"]
ALLERGY_WORDS = ("allergic", "allergy", "allergies", "intolerant")
# A row that continues an allergy list: comma separated names, or a name list ending in a full stop.
ALLERGY_LIST_ROW = re.compile(r",|[A-Za-z]{3,}\s*\.(?:\s|$)")
ALLERGY_LIST_MAX_ROWS = 4


def mentions_allergy(text: str) -> bool:
    """'allergic to', OCR-misread 'allengic' / 'alergy' too (an allergy list must never become a prescription)."""
    return any(len(w) >= 6 and SequenceMatcher(None, w, a).ratio() >= FUZZY_MIN
               for w in _words(text) for a in ALLERGY_WORDS)

FUZZY_MIN = 0.8   # a misread long word still counts ("Schisophrenia"), short words must match exactly


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z/]*", (text or "").lower())


def is_clinical(text: str) -> bool:
    for w in _words(text):
        if w in CLINICAL_EXACT or w.strip("/") in CLINICAL_EXACT:
            return True
        if len(w) >= 6 and any(SequenceMatcher(None, w, k).ratio() >= FUZZY_MIN for k in CLINICAL_LONG):
            return True
    return False


def _known_medicine_row(text: str) -> bool:
    """Use the local medicine index as a boundary hint, never as extracted truth.

    OCR often reads ``Tab Flagyl 400`` as ``X flagyl 400 101``.  The database is
    already on-device and can identify that row as medicine-like, but the actual
    name/value still has to be grounded by the OCR and structured by the pipeline.
    """
    # A bare dictionary match is too permissive (ordinary words such as a clinic
    # or surname can resemble a brand).  Require prescription context when this
    # helper is used as a section-boundary signal.
    value = text or ""
    if re.search(r"\b(?:timings?|clinic|hospital|morning|evening|afternoon|age|years?|date|rest|facility|"
                 r"temp|temperature|fever|since|review|problem|pulse)\b|\b(?:am|pm)\b|\d\s*:\s*\d", value, re.I):
        return False
    # "C Neurokind plus": a capsule / tablet letter whose dot the OCR dropped, before a capitalised name
    abbreviated = re.match(r"^\W*[TC]\s+[A-Z][A-Za-z]{3,}", value)
    if not abbreviated and not re.search(r"\b\d{2,}\b|\b(?:od|bd|bid|tds|tid|qid|hs|sos)\b", value, re.I):
        return _generic_medicine_row(value)
    try:
        from model.prescription.medicine_db import match_medicine
        words = re.findall(r"[a-z][a-z-]{2,}", value.lower())
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                candidate = " ".join(words[i:i + size])
                result = match_medicine(candidate)
                if result.get("match_status") in ("matched", "known_brand") and \
                        float(result.get("match_score") or 0) >= 90:
                    return True
    except Exception:
        # Boundary detection must remain usable when the optional medicine index
        # cannot load; the OCR evidence rules still provide a safe fallback.
        return False
    return False


def _generic_medicine_row(text: str) -> bool:
    """A line naming a medicine by its ingredient ("LIPOSOMAL AMPHOTERICIN B"), even without a number: long
    ingredient names (8+ letters, near-exact) do not occur in letterheads or patient details."""
    try:
        from model.prescription.medicine_db import match_generic
        hit = match_generic(text)
    except Exception:
        return False
    return bool(hit and hit["match_score"] >= 95 and len(hit["db_info"]["composition"].replace(" ", "")) >= 8)


def _without_clinical(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """The row with clinical-note boxes (and everything right of them) removed; None if nothing is left."""
    boxes = row.get("boxes")
    if not boxes:
        return None if is_clinical(row["text"]) else row
    cut = next((i for i, b in enumerate(boxes) if is_clinical(b["text"])), None)
    if cut is None:
        return row
    kept = boxes[:cut]
    if not kept:
        return None
    return {**row, "text": " ".join(b["text"] for b in kept), "raw": " ".join(b.get("raw", b["text"]) for b in kept),
            "boxes": kept, "_src": row}


def lines_for_ai(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """(prescription-body rows without personal/clinical details, number withheld).

    Dosage-form markers remain the preferred boundary.  If a doctor writes bare
    medicine names (``Dolo 650`` rather than ``Tab Dolo 650``), use strength or
    schedule evidence to find the body. A row is never sent merely because it contains
    an arbitrary number, and the old no-signal behaviour remains private.
    """
    used = [r for r in rows if r.get("used", True) and str(r.get("text") or "").strip()]
    marker_start = next((i for i, r in enumerate(used) if MEDICINE_START.search(r["text"])), None)
    personal_rows = [i for i, r in enumerate(used) if PERSONAL.search(r["text"])]
    # Medicines are written after the patient's details. A row above a name / age line that itself sits above
    # the first medicine marker is letterhead ("Veer Savarkar Marg, Mumbai - 400 016"), not the medicine section.
    above_patient = lambda i: marker_start is not None and any(i < p < marker_start for p in personal_rows)
    evidence = [i for i, r in enumerate(used)
                if not PERSONAL.search(r["text"]) and not is_clinical(r["text"]) and not above_patient(i) and
                (PRESCRIPTION_EVIDENCE.search(r["text"]) or _known_medicine_row(r["text"]))]
    starts = [i for i in (marker_start, *evidence) if i is not None]
    start = min(starts) if starts else None
    if start is None:
        return [], len(used)
    kept: List[Dict[str, Any]] = []
    allergy_rows = 0                    # > 0 while inside an "allergic to: A, B, / C, D." list
    for r in used[start:]:
        if kept and SECTION_END.search(r["text"]):
            break                       # signature / footer: nothing after it is sent
        if mentions_allergy(r["text"]) and (not MEDICINE_START.search(r["text"])
                                            or re.search(r"\bto\b", r["text"], re.I)):
            allergy_rows = 1            # "allergic to ..." (but "Tab Cetirizine for allergy" is a medicine)
            continue
        if allergy_rows:
            ends = (MEDICINE_START.search(r["text"]) or PRESCRIPTION_EVIDENCE.search(r["text"])
                    or not ALLERGY_LIST_ROW.search(r["text"]) or allergy_rows > ALLERGY_LIST_MAX_ROWS)
            if not ends:
                allergy_rows += 1
                continue                # "Naproxen, Amikacin, / Levofloxacin, Cefixime ..." continuation lines
            allergy_rows = 0
        clean = _without_clinical(r)         # clinical notes first (they share words like "yrs", "c/o") ...
        if clean is not None and not PERSONAL.search(clean["text"]):   # ... then any personal detail: whole line
            kept.append(clean)
    return kept, len(used) - len(kept)
