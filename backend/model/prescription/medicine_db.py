"""Local medicine lookup: Indian Medicine Dataset + curated known-brand fallback.

Replaces the old OpenFDA + built-in-list validation. Nothing leaves the machine.

Data
  data/medicines/updated_indian_medicine_data.csv   Indian Medicine Dataset (MIT), ~254k rows
  data/medicines/known_brands.json                  curated fallback (brands missing from the dataset)
  data/cache/medicine_index.pkl                     cleaned index, rebuilt when the CSV changes

Matching, per extracted medicine
  1. Company rule: a manufacturer name is never a medicine (unmatched, no candidates).
  2. Brand similarity (rapidfuzz ratio of the brand token) against every dataset brand.
     The brand score decides the status:
        matched    >= 90 AND >= 5 points ahead of the best DIFFERENT brand
        verify     75-89, or a different brand within 5 points that is a DIFFERENT medicine
     A different brand within 5 points with the same composition (ingredients + strengths) and the same kind of
     form is the same medicine under another name: it does not block "matched"; the best name match is kept
     and the alternatives are listed in a note (2026-11-18).
        unmatched  < 75
  3. Within the winning brand, the variant is chosen with a strength/form bonus
     (e.g. "Dolo 650" -> "Dolo 650 Tablet"); a strength that matches no variant
     downgrades "matched" to "verify".
  4. Not found in the dataset -> curated list -> "known_brand" (never "matched").
  5. The name the reader produced is never overwritten: ``read_as`` keeps it,
     ``suggested_name`` carries the database name when it differs.
  6. Identification guards (2026-11-17, measured with scripts/eval_matcher.py):
     - the brand word is chosen past an OCR-misread form prefix ("Td SERTA" -> serta, not the brand "TD");
     - a misspelt brand (similarity < 100) is "matched" only with corroboration: the written strength is in
       that product, or a written variant word ("PT", "Plus") is; otherwise "verify";
     - a written word after the brand that the product doesn't have ("Colgate Plax") -> "verify";
     - oral-care and other non-medicine products are "unmatched" with a note, never forced onto a drug.
"""
from __future__ import annotations

import csv
import json
import logging
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

log = logging.getLogger("perception.medicine_db")

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = BACKEND_ROOT / "data" / "medicines" / "updated_indian_medicine_data.csv"
KNOWN_BRANDS_PATH = BACKEND_ROOT / "data" / "medicines" / "known_brands.json"
CACHE_PATH = BACKEND_ROOT / "data" / "cache" / "medicine_index.pkl"
INDEX_VERSION = 2  # 2: strengths also parsed from the composition

MATCHED_MIN = 90.0
VERIFY_MIN = 75.0
BRAND_MARGIN = 5.0
KNOWN_BRAND_MIN = 90.0
COMPANY_MIN = 95.0

STRENGTH_BONUS = 5.0
STRENGTH_PENALTY = -5.0
FORM_BONUS = 3.0
TOKEN_BONUS = 2.0

# Dosage-form words -> canonical form.
FORMS = {
    "tablet": "tab", "tablets": "tab", "tab": "tab", "tabs": "tab", "dt": "tab", "chewable": "tab",
    "capsule": "cap", "capsules": "cap", "cap": "cap", "caps": "cap", "softgel": "cap",
    "syrup": "syp", "syp": "syp", "suspension": "syp", "susp": "syp", "liquid": "syp",
    "solution": "sol", "injection": "inj", "inj": "inj", "infusion": "inj",
    "drop": "drop", "drops": "drop", "cream": "cream", "ointment": "oint", "oint": "oint",
    "gel": "gel", "powder": "powder", "sachet": "powder", "granules": "powder",
    "lotion": "lotion", "spray": "spray", "inhaler": "inhaler", "rotacap": "inhaler",
    "respules": "inhaler", "lozenges": "tab", "soap": "soap", "shampoo": "shampoo",
    "paint": "paint", "mouthwash": "mouthwash", "gargle": "mouthwash", "toothpaste": "paste", "paste": "paste",
}
# Forms that tell products apart (a gum paint is never a tablet); tab / cap are not among them.
DISTINCT_FORMS = {"syp", "sol", "inj", "drop", "cream", "oint", "gel", "powder", "lotion", "spray", "inhaler", "soap",
                  "shampoo", "paint", "mouthwash", "paste"}
# Forms that are the same kind of product for this check: a "gum paint" is sold as a mouth gel, a syrup as a
# suspension or solution, an ointment as a cream.
FORM_FAMILY = {"paint": "topical", "gel": "topical", "cream": "topical", "oint": "topical", "lotion": "topical",
               "paste": "topical", "syp": "liquid", "sol": "liquid", "mouthwash": "liquid"}
# OCR readings of a leading "Tab" / "Cap" / "Syp" ("Td SERTA", "Tob Dolo", "7ab Pan"): never the brand when another
# word follows.
FORM_PREFIX_LOOKALIKES = {"t", "td", "tb", "tob", "tah", "tal", "tbl", "tav", "7ab", "9ab", "iab", "c", "cp", "cab",
                          "sy", "syr", "sup", "iny", "inj"}
# Products a prescription may list that are not medicines in the dataset (oral care, devices). Matched against the
# written words; such an item is "unmatched" with a note instead of the nearest drug name ("Colgate" -> Colistin).
NON_MEDICINES = {"colgate", "plax", "pepsodent", "closeup", "sensodyne", "listerine", "oral-b", "oralb",
                 "toothbrush", "brush", "toothpaste", "floss", "dabur", "patanjali", "himalaya"}
# Release/variant modifiers kept in the variant key but never a brand on their own.
MODIFIERS = {"sr", "er", "xr", "cr", "mr", "ds", "dsr", "la", "od", "plus", "forte", "duo", "mps",
             "ls", "kid", "kids", "junior", "max", "advance", "fort", "p", "d", "m", "l", "z"}
# Words a reader may leave in a name that say nothing about the product.
IGNORED_WORDS = {"tablet", "tablets", "capsule", "capsules", "oral", "suspension", "syrup", "with", "and", "only",
                 "after", "before", "food", "daily", "night", "morning", "strip", "strips"}
STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|gm|g|ml|iu|%|kg)?(?![a-z])")
COMPANY_SUFFIXES = {"pharmaceuticals", "pharmaceutical", "pharma", "ltd", "limited", "pvt", "private",
                    "laboratories", "laboratory", "labs", "lab", "healthcare", "health", "care",
                    "lifesciences", "life", "sciences", "science", "india", "inc", "co", "corp",
                    "corporation", "remedies", "biotech", "biologicals", "formulations", "drugs",
                    "industries", "international", "products", "medicare", "therapeutics", "global"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (text or "").lower())


def parse_strengths(text: str) -> set:
    """'Dolo 650 Tablet' -> {'650'}; '30mg/5ml' -> {'30', '5'}; drops trailing '.0'."""
    out = set()
    for num, _unit in STRENGTH_RE.findall((text or "").lower()):
        out.add(num[:-2] if num.endswith(".0") else num)
    return out


def parse_form(text: str) -> Optional[str]:
    for tok in reversed(tokens(text)):
        if tok in FORMS:
            return FORMS[tok]
    return None


def brand_of(name: str) -> str:
    """First token with a letter that is not a dosage form word: 'Tab Dolo 650' -> 'dolo'."""
    for tok in tokens(name):
        if any(ch.isalpha() for ch in tok) and tok not in FORMS and len(tok) >= 2:
            return tok
    return ""


def query_brand(name: str, index: Optional["MedicineIndex"] = None) -> str:
    """The brand word of a WRITTEN name. Like ``brand_of``, but a short first word that is an OCR misreading of a
    dosage form ("Td", "Tob", "7ab", "Ta6", "Tub") is skipped when a real word follows: a known lookalike always,
    any other first word of at most 3 letters when the next word (4+ letters) matches a dataset brand at least as
    well. "Pan 40", "Mox CV", "Avil" keep their first word (nothing of 4+ letters follows, or it is the brand)."""
    words = [t for t in tokens(name) if any(ch.isalpha() for ch in t) and t not in FORMS and len(t) >= 2]
    if len(words) < 2:
        return words[0] if words else ""
    first, second = words[0], words[1]
    letters = lambda w: len(re.sub(r"[^a-z]", "", w))
    if letters(second) < 3:
        return first
    if first in FORM_PREFIX_LOOKALIKES:
        return second
    if index is not None and letters(first) <= 3 and letters(second) >= 4:
        best = lambda w: (process.extractOne(w, index.brands, scorer=fuzz.ratio) or (None, 0.0))[1]
        if best(second) >= best(first):
            return second
    return first


JOINED_SUFFIXES = MODIFIERS | {"cv", "pt", "sp", "mf", "hd", "dsr", "dt", "md", "gm", "tr"}


def split_joined_variant(name: str, index: "MedicineIndex") -> str:
    """'PanD 40mg' -> 'Pan D 40mg', 'ZerodolPT' -> 'Zerodol PT': a variant suffix written without a space. Only when
    the joined word is not a brand itself, the front part (3+ letters) is a brand, and that brand really has a
    product with that variant word. Otherwise the name is returned unchanged."""
    out = []
    for word in (name or "").split():
        low = word.lower()
        if low.isalpha() and len(low) >= 4 and low not in index.by_brand:
            for cut in range(len(low) - 1, 2, -1):
                head, tail = low[:cut], low[cut:]
                if (tail in JOINED_SUFFIXES and head in index.by_brand
                        and any(tail in index.rows[i]["variant"] for i in index.by_brand[head])):
                    word = f"{word[:cut]} {word[cut:]}"
                    break
        out.append(word)
    return " ".join(out)


def non_medicine(name: str) -> Optional[str]:
    """The non-medicine word in a written name ("Colgate Plax" -> "colgate"), else None."""
    low = (name or "").lower()
    for w in re.findall(r"[a-z][a-z-]+", low):
        if w in NON_MEDICINES:
            return w
    return "oral-b" if re.search(r"\boral\s*-?\s*b\b", low) else None


def variant_words(name: str) -> set:
    return {t for t in tokens(name) if t.isalpha() and t not in FORMS} - {brand_of(name)}


def company_core(name: str) -> str:
    words = [t for t in tokens(name) if t not in COMPANY_SUFFIXES and not t.isdigit()]
    return " ".join(words)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _composition(row: Dict[str, str]) -> Optional[str]:
    salt = _clean(row.get("salt_composition", ""))
    if salt:
        return salt
    parts = [_clean(row.get(k, "")) for k in ("short_composition1", "short_composition2")]
    parts = [p for p in parts if p]
    return " + ".join(parts) or None


def _interactions(raw: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(raw) if raw and raw.strip() else {}
    except (ValueError, TypeError):
        return []
    drugs, effects = data.get("drug") or [], data.get("effect") or []
    out = []
    for i, drug in enumerate(drugs):
        drug = _clean(str(drug))
        if drug:
            out.append({"drug": drug, "effect": _clean(str(effects[i])) if i < len(effects) else ""})
    return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
@dataclass
class MedicineIndex:
    rows: List[Dict[str, Any]]                    # active (not discontinued) medicines
    by_brand: Dict[str, List[int]]                # brand token -> row indices
    brands: List[str]                             # unique brands (rapidfuzz choices)
    companies: Dict[str, str]                     # company core -> manufacturer display name
    known: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # curated fallback
    known_source: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


def _csv_signature(path: Path) -> Tuple[int, int]:
    st = path.stat()
    return (st.st_size, int(st.st_mtime))


def build_index(csv_path: Path = CSV_PATH) -> MedicineIndex:
    started = time.perf_counter()
    csv.field_size_limit(sys.maxsize)
    rows: List[Dict[str, Any]] = []
    by_brand: Dict[str, List[int]] = {}
    companies: Dict[str, str] = {}
    seen = set()
    total = discontinued = duplicates = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for raw in csv.DictReader(f):
            total += 1
            manufacturer = _clean(raw.get("manufacturer_name", ""))
            core = company_core(manufacturer)
            if core:
                companies.setdefault(core, manufacturer)
            if str(raw.get("Is_discontinued", "")).strip().upper() == "TRUE":
                discontinued += 1
                continue
            name = _clean(raw.get("name", ""))
            key = name.lower()
            if not name or key in seen:
                duplicates += 1
                continue
            seen.add(key)
            brand = brand_of(name)
            if not brand:
                continue
            side = [s.strip() for s in (raw.get("side_effects") or "").split(",") if s.strip()]
            row = {
                "id": raw.get("id"),
                "name": name,
                "brand": brand,
                # strengths from the name AND the composition ("Crocin Advance" = 500 mg)
                "strengths": parse_strengths(name) | parse_strengths(_composition(raw) or ""),
                "form": parse_form(name) or parse_form(raw.get("pack_size_label", "")),
                "variant": variant_words(name),
                "manufacturer": manufacturer or None,
                "type": _clean(raw.get("type", "")) or None,
                "pack_size": _clean(raw.get("pack_size_label", "")) or None,
                "composition": _composition(raw),
                "description": _clean(raw.get("medicine_desc", "")) or None,
                "side_effects": side or None,
                "interactions": _interactions(raw.get("drug_interactions", "")) or None,
            }
            by_brand.setdefault(brand, []).append(len(rows))
            rows.append(row)
    stats = {"csv_rows": total, "discontinued_dropped": discontinued, "duplicates_dropped": duplicates,
             "active_rows": len(rows), "brands": len(by_brand), "companies": len(companies),
             "build_seconds": round(time.perf_counter() - started, 1)}
    log.info("Medicine index built: %s", stats)
    return MedicineIndex(rows=rows, by_brand=by_brand, brands=list(by_brand), companies=companies, stats=stats)


def _attach_known(index: MedicineIndex, path: Path = KNOWN_BRANDS_PATH) -> None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("Known-brand list unavailable (%s)", exc)
        return
    index.known = {b["name"].lower(): b for b in doc.get("brands", []) if b.get("name")}
    index.known_source = doc.get("_source", "")
    for company in doc.get("companies", []):
        index.companies.setdefault(company_core(company) or company.lower(), company)


def load_index(csv_path: Path = CSV_PATH, cache_path: Path = CACHE_PATH) -> MedicineIndex:
    """Load the cached index, rebuilding it if the CSV changed (size/mtime) or the format moved on."""
    sig = _csv_signature(csv_path)
    index: Optional[MedicineIndex] = None
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)  # our own cache file, written below
            if payload.get("version") == INDEX_VERSION and tuple(payload.get("csv")) == sig:
                index = payload["index"]
        except Exception as exc:  # corrupt/old cache: rebuild
            log.warning("Ignoring medicine index cache (%s)", exc)
    if index is None:
        index = build_index(csv_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"version": INDEX_VERSION, "csv": list(sig), "index": index}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    _attach_known(index)
    return index


_INDEX: Optional[MedicineIndex] = None
_INDEX_ERROR: Optional[str] = None
_LOCK = threading.Lock()


def get_index() -> Optional[MedicineIndex]:
    """Process-wide index, loaded once (thread-safe). None if the dataset is missing."""
    global _INDEX, _INDEX_ERROR
    if _INDEX is not None or _INDEX_ERROR is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is None and _INDEX_ERROR is None:
            started = time.perf_counter()
            try:
                _INDEX = load_index()
                log.info("Medicine index ready in %.1fs (%d medicines)",
                         time.perf_counter() - started, len(_INDEX.rows))
            except FileNotFoundError:
                _INDEX_ERROR = f"Medicine dataset not found: {CSV_PATH}"
                log.warning(_INDEX_ERROR)
            except Exception as exc:
                _INDEX_ERROR = f"Medicine index failed to load: {exc}"
                log.exception(_INDEX_ERROR)
    return _INDEX


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _db_info(row: Dict[str, Any]) -> Dict[str, Any]:
    """Dataset fields for the UI; empty fields are omitted entirely."""
    info = {"source": "From medicine database"}
    for key in ("composition", "type", "manufacturer", "pack_size", "description",
                "side_effects", "interactions"):
        if row.get(key):
            info[key] = row[key]
    return info


_COMP_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|gm|ml|iu|%)", re.I)


def composition_key(row: Dict[str, Any]) -> Optional[Tuple]:
    """'Paracetamol (650mg)' Tablet -> ((('paracetamol', ('650mg',)),), 'tab'). Two products with the same key are
    the same medicine: same ingredients, same strengths, same kind of form. None when the composition is unknown."""
    from .uses import _norm_one
    comp = row.get("composition") or ""
    parts = []
    for p in re.split(r"\s*\+\s*", comp):
        ing = _norm_one(p)
        if not ing:
            continue
        strengths = tuple(sorted(f"{float(n):g}{u.lower().replace('gm', 'g').replace('µg', 'mcg')}"
                                 for n, u in _COMP_PART.findall(p)))
        parts.append((ing, strengths))
    if not parts:
        return None
    form = row.get("form")
    form = "oral" if form in ("tab", "cap", None) else FORM_FAMILY.get(form, form)
    return tuple(sorted(parts)), form


def _candidate(row: Dict[str, Any], brand_score: float, variant_score: float) -> Dict[str, Any]:
    return {"name": row["name"], "score": round(brand_score, 1), "variant_score": round(variant_score, 1),
            "manufacturer": row.get("manufacturer"), "composition": row.get("composition"),
            "pack_size": row.get("pack_size")}


def _best_variant(index: MedicineIndex, brand: str, q_strengths: set, q_form: Optional[str],
                  q_words: set) -> Tuple[Dict[str, Any], float, bool]:
    """Pick the brand's variant with the highest strength/form bonus. Returns (row, bonus, strength_ok)."""
    best, best_bonus, best_ok = None, float("-inf"), True
    for i in index.by_brand[brand]:
        row = index.rows[i]
        bonus, ok = 0.0, True
        if q_strengths and row["strengths"]:
            if q_strengths & row["strengths"]:
                bonus += STRENGTH_BONUS
            else:
                bonus += STRENGTH_PENALTY
                ok = False
        if q_form and row["form"] == q_form:
            bonus += FORM_BONUS
        if q_words & row["variant"]:
            bonus += TOKEN_BONUS * len(q_words & row["variant"])
        # prefer the plain product (fewer extra words) on ties
        tie = -0.01 * len(row["variant"])
        if bonus + tie > best_bonus:
            best, best_bonus, best_ok = row, bonus + tie, ok
    return best, round(best_bonus, 2), best_ok


# ---------------------------------------------------------------------------
# Generic names: "Hydroxychloroquine 400mg", "Inj Liposomal Amphotericin B", "Remdesivir"
# ---------------------------------------------------------------------------
GENERIC_MIN = 92.0          # fuzzy score of the written words against an ingredient name
GENERIC_MIN_LETTERS = 6     # shorter words are too often an unrelated brand or an English word
GENERIC_NOT_OVER = 95.0     # a brand this close to what was written wins over a generic reading
_GENERIC: Dict[int, Dict[str, List[int]]] = {}


def _generic_table(index: MedicineIndex) -> Dict[str, List[int]]:
    """single-ingredient name -> rows of products with exactly that ingredient (built once, ~1 s)."""
    table = _GENERIC.get(id(index))
    if table is None:
        from .uses import ingredients_of
        table = {}
        for i, row in enumerate(index.rows):
            parts = ingredients_of(row.get("composition") or "")
            if len(parts) == 1 and not any(ch.isdigit() for ch in parts[0]):
                table.setdefault(parts[0], []).append(i)
        _GENERIC.clear()
        _GENERIC[id(index)] = table
    return table


def match_generic(name: str, dosage: Optional[str] = None,
                  index: Optional[MedicineIndex] = None) -> Optional[Dict[str, Any]]:
    """A medicine written by its ingredient (generic) name. Every run of 1-3 written words is compared with the
    dataset's single-ingredient names ("liposomal amphotericin b" -> "amphotericin b" / "liposomal amphotericin b");
    None when nothing is close enough. Products with that ingredient (written strength first) are candidates."""
    from .uses import SYNONYMS
    index = index or get_index()
    if index is None:
        return None
    table = _generic_table(index)
    words = [t for t in tokens(name) if t not in FORMS and not any(ch.isdigit() for ch in t)]
    best: Tuple[float, str] = (0.0, "")
    best_size = 0
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            phrase = SYNONYMS.get(phrase, phrase)
            if len(phrase.replace(" ", "")) < GENERIC_MIN_LETTERS:
                continue
            for form in {phrase, phrase.replace(" ", "")}:   # "HYDROXY CHLORO QUINE" -> "hydroxychloroquine"
                hit = process.extractOne(form, list(table), scorer=fuzz.ratio, score_cutoff=GENERIC_MIN)
                if hit and (hit[1], len(hit[0])) > (best[0], len(best[1])):
                    best, best_size = (float(hit[1]), hit[0]), size
    if not best[1]:
        return None
    score, ingredient = best
    rows = [index.rows[i] for i in table[ingredient]]
    q_strengths = parse_strengths(f"{name} {dosage or ''}")
    rows.sort(key=lambda r: (not (q_strengths & r["strengths"]), len(r["name"])))
    title = ingredient.title() if ingredient.islower() else ingredient
    return {"match_status": "matched", "match_score": round(score, 1),
            "suggested_name": title if score < 100 else None,
            "candidates": [_candidate(r, score, score) for r in rows[:3]],
            "db_info": {"source": "Generic (ingredient) name in the medicine database", "composition": title,
                        "variant_note": f"Written as the generic name; {len(rows)} products in the database "
                                        f"contain only {title}"},
            "known_brand": None, "generic": True, "generic_words": best_size,
            "strength_variants": len({frozenset(r["strengths"]) for r in rows if r["strengths"]}), "rival_score": None,
            "notes": [f"'{name}' is written as the generic name {title}; any brand of it with the same "
                      f"strength is the same medicine"]}


def match_medicine(name: str, dosage: Optional[str] = None,
                   index: Optional[MedicineIndex] = None) -> Dict[str, Any]:
    """Match one extracted medicine. Returns status, score, suggestion, candidates, db_info, notes."""
    index = index or get_index()
    result: Dict[str, Any] = {"match_status": "unmatched", "match_score": None, "suggested_name": None,
                              "candidates": [], "db_info": None, "known_brand": None, "notes": []}
    if index is None:
        result["notes"].append("Medicine database unavailable; name not checked")
        return result

    split = split_joined_variant(name or "", index)
    if split != (name or ""):
        result["notes"].append(f"Read '{name}' as '{split}' (brand + variant written together)")
        name = split
    q_text = f"{name or ''} {dosage or ''}"
    q_brand = query_brand(name or "", index)
    if not q_brand:
        return result
    thing = non_medicine(name or "")
    if thing:
        result["notes"].append(f"'{name}' looks like a non-medicine product ({thing}), not a medicine in the database")
        result["non_medicine"] = True
        return result
    q_strengths = parse_strengths(q_text)
    q_form = parse_form(q_text)
    q_words = (variant_words(q_text) - {q_brand}) & MODIFIERS | (variant_words(name or "") - {q_brand})
    # words written AFTER the brand ("Zerodol PT" -> {"pt"}); a skipped "Td" prefix is not one of them
    name_tokens = tokens(name or "")
    after_brand = name_tokens[name_tokens.index(q_brand) + 1:] if q_brand in name_tokens else []
    q_words = {w for w in q_words if w in after_brand or w in MODIFIERS}

    # 1. Company rule, checked FIRST: "Cipla" is a company even though a product
    # called "Cipla Kids Billargic ..." exists. Only exact company names without
    # digits that are not curated medicine names ("Pan 40" is a medicine).
    q_full = " ".join(t for t in tokens(name) if t not in FORMS)
    q_core = company_core(q_full)
    if (q_core and not any(ch.isdigit() for ch in q_full) and q_core in index.companies
            and q_core not in index.known and q_full not in index.known):
        result["notes"].append(
            f"'{name}' is a pharmaceutical company ({index.companies[q_core]}), not a medicine")
        return result

    # 2. Brand similarity across all dataset brands.
    hits = process.extract(q_brand, index.brands, scorer=fuzz.ratio, limit=8, score_cutoff=50)
    brand_scores = [(b, float(s)) for b, s, _ in hits]

    # 2b. Written as a generic (ingredient) name, and no brand is as close: "Remdesivir" is not "Predivir".
    # A generic name the OCR split into words ("HYDROXY CHLORO QUINE") wins even when its first word is a brand.
    top = brand_scores[0][1] if brand_scores else 0.0
    generic = match_generic(name, dosage, index) if len(tokens(name)) > 1 or top < GENERIC_NOT_OVER else None
    if generic and (generic["match_score"] >= top or
                    (generic.pop("generic_words", 1) > 1 and generic["match_score"] >= GENERIC_NOT_OVER)):
        generic.pop("generic_words", None)
        return generic

    # 4. Curated fallback when the dataset has no convincing brand.
    best_brand_score = brand_scores[0][1] if brand_scores else 0.0
    known_key = None
    for key in (" ".join(tokens(name)), q_brand):
        if key in index.known:
            known_key = key
            break
    if known_key is None and index.known:
        k = process.extractOne(q_brand, list(index.known), scorer=fuzz.ratio, score_cutoff=KNOWN_BRAND_MIN)
        known_key = k[0] if k else None
    def known_brand_result(key: str) -> Dict[str, Any]:
        entry = index.known[key]
        result.update(match_status="known_brand",
                      match_score=round(float(fuzz.ratio(q_brand, key.split()[0])), 1),
                      known_brand={"name": entry["name"], "composition": entry.get("composition"),
                                   "source": "Known brand list (no database details)"},
                      db_info=None)
        result["suggested_name"] = entry["name"].title() if brand_of(entry["name"]) != q_brand else None
        result["candidates"] = [_candidate(_best_variant(index, b, q_strengths, q_form, q_words)[0], s, s)
                                for b, s in brand_scores[:3]]
        result["notes"] = [f"'{name}' is a known brand but not in the medicine database; no details available"]
        return result

    if known_key and best_brand_score < 95:
        return known_brand_result(known_key)
    exact_known = " ".join(t for t in tokens(name) if t not in FORMS)
    exact_known = exact_known if exact_known in index.known else None

    if not brand_scores or best_brand_score < VERIFY_MIN:
        result["candidates"] = [_candidate(_best_variant(index, b, q_strengths, q_form, q_words)[0], s, s)
                                for b, s in brand_scores[:3]]
        result["match_score"] = round(best_brand_score, 1) if brand_scores else None
        result["notes"].append(f"'{name}' was not found in the medicine database")
        return result

    # 3. Variant within each brand; top-3 = distinct brands first, then more variants of the winner.
    scored = []
    for b, s in brand_scores:
        row, bonus, ok = _best_variant(index, b, q_strengths, q_form, q_words)
        scored.append((b, s, row, bonus, ok))
    scored.sort(key=lambda x: (x[1], x[3]), reverse=True)
    top_brand, top_score, top_row, top_bonus, strength_ok = scored[0]
    # kept even if the curated known-brand list takes over below: "Dolo 999mg" is no Dolo in the dataset
    result["written_strength_in_brand"] = strength_ok if q_strengths else None
    # A different brand scoring within BRAND_MARGIN is a real conflict only when it is a different medicine.
    # Brands with the same composition (ingredients + strengths) and the same kind of form are the same
    # medicine under another name: the best name match is kept and they are listed in a note instead.
    top_key = composition_key(top_row)
    same_comp, conflicting = [], []
    for b, s, row, *_ in scored[1:]:
        if b == top_brand:
            continue
        if top_key and composition_key(row) == top_key:
            same_comp.append((b, s, row))
        elif top_score - s < BRAND_MARGIN:
            conflicting.append(s)
    close_same = [(b, s, row) for b, s, row in same_comp if top_score - s < BRAND_MARGIN]
    other = max(conflicting) if conflicting else \
        next((s for b, s, *_ in scored[1:] if b != top_brand and not any(b == x[0] for x in close_same)), 0.0)

    candidates = [_candidate(row, s, min(100.0, s + bonus)) for b, s, row, bonus, _ in scored[:3]]
    if len(candidates) < 3:
        extra = [index.rows[i] for i in index.by_brand[top_brand] if index.rows[i] is not top_row]
        for row in extra[: 3 - len(candidates)]:
            candidates.append(_candidate(row, top_score, top_score))
    result["candidates"] = candidates
    result["match_score"] = round(top_score, 1)

    # Corroboration for a brand that is not spelt exactly: the written strength is in the product, or a written
    # variant word ("PT", "Plus") is in its name. Spelling similarity alone is not enough ("Colgate" ~ "Colate").
    corroborated = bool(q_strengths & top_row["strengths"]) or bool(q_words & top_row["variant"])
    # Written words after the brand that the product doesn't have ("Colgate Plax", "Oral B brush").
    unexplained = [w for w in after_brand if w.isalpha() and len(w) >= 4 and w not in FORMS and w not in MODIFIERS
                   and w not in top_row["variant"] and w not in IGNORED_WORDS]
    # A written non-tablet form the brand doesn't make ("Stolin gum paint" vs Strolin tablets). Tablet vs capsule
    # is not checked: prescriptions write "Tab" loosely.
    brand_forms = {index.rows[i]["form"] for i in index.by_brand[top_brand]}
    form_conflict = bool(q_form and q_form in DISTINCT_FORMS
                         and FORM_FAMILY.get(q_form, q_form) not in {FORM_FAMILY.get(f, f) for f in brand_forms})
    if form_conflict:
        result["notes"].append(f"'{name}' is written as a {q_form}, but {top_brand.title()} has no {q_form} in the "
                               f"database; verify")
    if top_score >= MATCHED_MIN and top_score - other >= BRAND_MARGIN and strength_ok and \
            (top_score >= 100 or corroborated) and not unexplained and not form_conflict:
        status = "matched"
    else:
        if top_score >= MATCHED_MIN and top_score < 100 and not corroborated:
            result["notes"].append(f"'{name}' is spelt like {top_row['name']}, but nothing else written confirms "
                                   f"it; check it on the prescription")
        if unexplained:
            result["notes"].append(f"'{name}': '{' '.join(unexplained)}' is not part of {top_row['name']}; verify")
        status = "verify"
        if top_score >= MATCHED_MIN and top_score - other < BRAND_MARGIN:
            result["notes"].append(f"'{name}': several brands match about equally; choose the right one")
        if not strength_ok:
            result["notes"].append(f"'{name}': no {top_brand.title()} product with the written strength; verify")
    if status == "matched" and close_same:
        names = ", ".join(dict.fromkeys(row["name"] for _, _, row in close_same))
        result["notes"].append(f"'{name}': picked {top_row['name']} (closest name). {names} "
                               f"{'has' if len(close_same) == 1 else 'have'} the same composition "
                               f"({top_row.get('composition')}), so either is the same medicine")
        result["same_composition_alternatives"] = [row["name"] for _, _, row in close_same]
    if status != "matched" and exact_known:
        # e.g. "Omega 3": curated supplement; the dataset's "Omega" is a different drug
        return known_brand_result(exact_known)
    result["match_status"] = status
    result["rival_score"] = round(other, 1) if other else None
    info = _db_info(top_row)

    # No strength written but the brand has several strengths: don't invent one.
    brand_strengths = {frozenset(index.rows[i]["strengths"]) for i in index.by_brand[top_brand]}
    # strengths among the products of the variant that was written: "Sizodon Plus" is one product, plain
    # "Sizodon" comes in 1 / 2 / 3 / 4 mg (falls back to the whole brand when no product has that variant)
    extra = lambda words: {w for w in words if w not in IGNORED_WORDS and w not in {"mouth", "eye", "ear", "nasal"}}
    want = extra(q_words)
    same_variant = [index.rows[i] for i in index.by_brand[top_brand] if extra(index.rows[i]["variant"]) == want]
    result["strength_variants"] = len({frozenset(r["strengths"]) for r in (same_variant or
                                       [index.rows[i] for i in index.by_brand[top_brand]]) if r["strengths"]})
    strength_unknown = not q_strengths and len(brand_strengths) > 1
    if strength_unknown and info.get("composition"):
        info["composition"] = re.sub(r"\s*\([^)]*\)", "", info["composition"]).strip()
        info["variant_note"] = (f"Strength not written; {len(index.by_brand[top_brand])} "
                                f"{top_brand.title()} products in the database")
    result["db_info"] = info

    if top_brand != q_brand:
        result["suggested_name"] = top_row["name"]
        result["notes"].append(f"Read as '{name}'; closest database medicine is '{top_row['name']}'."
                               + ("" if status == "matched" else " Verify."))
    elif status == "matched" and not strength_unknown and top_row["name"].lower() != (name or "").lower():
        result["suggested_name"] = top_row["name"]  # same brand, full product name
    return result


READER_WEIGHT = {"matched": 1.0, "verify": 0.8, "known_brand": 0.7, "unmatched": 0.5}
GROUNDING_CAP = 0.30


def combined_confidence(reader_confidence: float, status: str, match_score: Optional[float],
                        grounded: bool) -> float:
    """reader × (match_score/100) × status weight, all 0-1; ungrounded names stay capped at 0.30."""
    reader = max(0.0, min(1.0, float(reader_confidence or 0.0)))
    if status in ("matched", "verify", "known_brand") and match_score is not None:
        value = reader * (float(match_score) / 100.0) * READER_WEIGHT[status]
    else:
        value = reader * READER_WEIGHT["unmatched"]
    if not grounded:
        value = min(value, GROUNDING_CAP)
    return round(max(0.0, min(1.0, value)), 3)


STATUS_RANK = {"matched": 3, "known_brand": 2, "verify": 1, "unmatched": 0}

# ---- confidence level shown next to the status (never a raw percentage) -----------------------------------
# Strength, four cases (2026-11-18):
#   written, in the product                        -> good
#   written, not in the product                    -> bad (Low)
#   not written, the product has one strength / none listed -> neutral "not applicable" (never lowers it)
#   not written, the brand has several strengths   -> ambiguous: 2-3 strengths fair (Medium), 4+ bad (Low)
# Other similar names: a different brand scoring within BRAND_MARGIN of the match -> bad; none -> good.
# HIGH    matched or known brand, every factor good or neutral, and the name similarity is >= 95; when the strength
#         says nothing (not applicable / unknown) the name must match exactly (100), since the strength can't confirm it
# LOW     any factor bad: not in the list, name not in the text read, variant or strength mismatch, many strengths
#         and none written, a close different brand, name score < 80, or OCR confidence < 0.60
# MEDIUM  everything in between
# 95 / 80 sit inside the matcher's own bands (matched >= 90, verify 75-89); 0.85 is the "check" marker's OCR
# threshold and 0.60 the point below which OCR lines are mostly misread (see project_review/ADR-005).
HIGH_SCORE, LOW_SCORE = 95.0, 80.0
HIGH_OCR, LOW_OCR = 0.85, 0.60
MANY_STRENGTHS = 4


def strength_agreement(item: Dict[str, Any], product: Optional[str], composition: Optional[str]) -> Optional[bool]:
    """True: the written strength is in the chosen product; False: it isn't; None: no strength written / no product."""
    sm = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu)\b", str(item.get("strength") or ""), re.I)
    if not sm or not (product or composition):
        return None
    if not STRENGTH_UNIT.search(f"{product or ''} {composition or ''}"):
        return None          # nothing to compare with (known-brand list entries have no strengths; "D3" is not one)
    rx = re.compile(r"(?<![\d.])" + re.escape(sm.group(1)) + r"\s*" + sm.group(2) + r"\b", re.I)
    # product names often carry the strength without a unit: "Augmentin 625 Duo", "Dolo 650", "PAN 40"
    bare = re.compile(r"(?<![\d.])" + re.escape(sm.group(1)) + r"(?![\d.])(?!\s*(?:mg|mcg|g|ml|iu)\b)", re.I)
    return bool(rx.search(product or "") or rx.search(composition or "") or bare.search(product or ""))


def strength_factor(item: Dict[str, Any], agreement: Optional[bool]) -> Tuple[str, str]:
    """(value, effect) of the Strength factor, from the four cases above."""
    if agreement is True:
        return "matches the product", "good"
    if agreement is False:
        return "not in the product", "bad"
    if item.get("strength"):
        return "written; no strengths in the data to compare", "neutral"
    n = item.get("strength_variants")
    if n is None:                       # known-brand list: no strengths in the data
        return "not written (no strengths in the data)", "neutral"
    if n <= 1:
        return "not applicable (one strength only)" if n == 1 else "not applicable (no strengths listed)", "neutral"
    return f"not written; {n} strengths exist", "bad" if n >= MANY_STRENGTHS else "fair"


def confidence_level(item: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """("high" | "medium" | "low", factors). Factors are shown in Details: [{"factor", "value", "effect"}]."""
    status, score = item.get("match_status"), item.get("match_score")
    ocr, grounded = item.get("ocr_confidence"), item.get("in_ocr_text", True)
    product = (item.get("candidates") or [{}])[0].get("name") if status != "known_brand" else (item.get("known_brand") or {}).get("name")
    comp = (item.get("db_info") or {}).get("composition") or (item.get("known_brand") or {}).get("composition")
    agreement = strength_agreement(item, product, comp) if status in ("matched", "known_brand", "verify") else None
    if item.get("written_strength_in_brand") is False:
        agreement = False    # the dataset has this brand, but none of its products has the written strength
    elif agreement is None and item.get("written_strength_in_brand") is True:
        agreement = True     # the strength is part of the written name ("Augmentin 625 Duo", "Dolo 650")
    strength_value, strength_effect = strength_factor(item, agreement)
    variant_ok = not item.get("variant_mismatch")
    rival = item.get("rival_score")
    rival_close = rival is not None and score is not None and score - rival < BRAND_MARGIN
    status_word = {"matched": "matched", "known_brand": "known brand", "verify": "close matches", "unmatched": "not in the list"}
    factors = [
        {"factor": "Medicine list", "value": status_word.get(status, status),
         "effect": "good" if status in ("matched", "known_brand") else "bad" if status == "unmatched" else "fair"},
        {"factor": "Name similarity", "value": f"{score:.0f} of 100" if score is not None else "none",
         "effect": "good" if (score or 0) >= HIGH_SCORE else "bad" if (score or 0) < LOW_SCORE else "fair"},
        {"factor": "Other similar names", "value": "none close" if not rival_close else f"another brand scores {rival:.0f}",
         "effect": "bad" if rival_close else "good"},
        {"factor": "Strength", "value": strength_value, "effect": strength_effect},
        {"factor": "Variant (Plus, SR, DS ...)", "value": "matches" if variant_ok else "differs",
         "effect": "good" if variant_ok else "bad"},
        {"factor": "Handwriting read", "value": "unknown" if ocr is None else f"{'clear' if ocr >= HIGH_OCR else 'unclear' if ocr < LOW_OCR else 'fair'}",
         "effect": "neutral" if ocr is None else "good" if ocr >= HIGH_OCR else "bad" if ocr < LOW_OCR else "fair"},
        {"factor": "Name in the text read", "value": "yes" if grounded else "no", "effect": "good" if grounded else "bad"},
    ]
    effects = [f["effect"] for f in factors]
    # with nothing to confirm it from the strength, only an exact name match is High
    exact_needed = strength_effect == "neutral" and (score or 0) < 100
    if "bad" in effects:
        level = "low"
    elif status in ("matched", "known_brand") and all(e in ("good", "neutral") for e in effects) and not exact_needed:
        level = "high"
    else:
        level = "medium"
    return level, factors


def add_uses(item: Dict[str, Any], m: Dict[str, Any]) -> None:
    """'What it's for' from the ingredient lookup (database text only, never AI). Verify rows get it only per
    candidate: the line is shown after the person picks the right medicine."""
    from .uses import SOURCE_LABEL, lookup, what_for
    for c in m.get("candidates") or []:
        info = lookup(c.get("composition"))
        c["what_for"] = what_for(info)
        c["what_for_source"] = SOURCE_LABEL.get((info or {}).get("source"))
    item["what_for"], item["what_for_source"], item["other_uses"], item["common_side_effects"], item["drug_class"] = \
        None, None, [], [], None
    if m["match_status"] not in ("matched", "known_brand"):
        return
    comp = (m.get("db_info") or {}).get("composition") or (m.get("known_brand") or {}).get("composition")
    info = lookup(comp)
    line = what_for(info)
    if line:
        item["what_for"] = line
        item["what_for_source"] = SOURCE_LABEL.get(info.get("source"))
        item["other_uses"] = [u for u in info.get("uses", []) if u.lower() not in line.lower()]
        item["common_side_effects"] = info.get("side_effects", [])
        item["drug_class"] = info.get("class")


STRENGTH_UNIT = re.compile(r"\d\s*(?:mg|mcg|µg|g|gm|ml|iu)\b", re.I)
# A brand's variants are different products: "X Plus" adds an ingredient, "X SR" releases slowly, ...
VARIANT_SUFFIXES = {"plus", "forte", "dsr", "ds", "ls", "sr", "cr", "mr", "xr", "cv"}


def variant_suffixes(name: str) -> set:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower())} & VARIANT_SUFFIXES


def chosen_product(m: Dict[str, Any]) -> Optional[str]:
    if m.get("match_status") == "known_brand":
        return (m.get("known_brand") or {}).get("name")
    return (m.get("candidates") or [{}])[0].get("name") or m.get("suggested_name")


def check_variant(m: Dict[str, Any], written: str) -> Dict[str, Any]:
    """A matched product must have the same variant suffixes as what was written: "Sizodon Plus" is never
    "Matched" to plain "Sizodon", and plain "Pan" never to "Pan D"... (only the suffixes in VARIANT_SUFFIXES).
    A mismatch becomes Verify, with the candidates left for the person to choose from."""
    if m.get("match_status") not in ("matched", "known_brand"):
        return m
    product = chosen_product(m) or ""
    want, have = variant_suffixes(written), variant_suffixes(product)
    if want == have:
        return m
    m = dict(m)
    m["variant_mismatch"] = True
    m["match_status"] = "verify"
    show = lambda s: " ".join(sorted(x.upper() if len(x) <= 3 else x.title() for x in s)) or "no suffix"
    m["notes"] = [*m.get("notes", []), f"“{written}” ({show(want)}) and “{product}” ({show(have)}) are different "
                  f"products. Choose the right one, or ask your pharmacist."]
    return m


def prefer_written_strength(m: Dict[str, Any], strength: Optional[str]) -> Dict[str, Any]:
    """'Xocalm 5mg': candidates with the written strength go first ("Zocalm 5mg" before "Ocalm 0.25mg").
    The whole number must match: '5mg' never matches '0.5mg' or '25mg'."""
    sm = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu)\b", str(strength or ""), re.I)
    if not sm or not m.get("candidates"):
        return m
    rx = re.compile(r"(?<![\d.])" + re.escape(sm.group(1)) + r"\s*" + sm.group(2) + r"\b", re.I)
    fits = [c for c in m["candidates"] if rx.search(c["name"]) or rx.search(c.get("composition") or "")]
    if not fits:
        return m
    m = dict(m)
    m["candidates"] = fits + [c for c in m["candidates"] if c not in fits]
    if m["match_status"] == "verify" and not rx.search(m.get("suggested_name") or ""):
        m["suggested_name"] = fits[0]["name"]
        m["notes"] = [*m["notes"], f"Closest match with the written strength {sm.group(1)}{sm.group(2)}: '{fits[0]['name']}'"]
    return m
_ALT_SPLIT = re.compile(r"\s*(?:/|\(\s*or\s*\)|\bor\b)\s*", re.I)


_BRACKETED_NAME = re.compile(r"\(\s*([A-Za-z][A-Za-z\s-]{2,})\s*\)")


def split_names(name: str) -> List[str]:
    """'Dothepin / Prothiaden', 'Clonazepam (OR) Lonazep' or 'Ativan (Lorazepam) 2mg' -> each name, the written
    brand first; a single name stays as it is. A bracket holding a number ("Augmentin (625)") is not a name."""
    bracketed = [m.group(1).strip() for m in _BRACKETED_NAME.finditer(name or "") if m.group(1).strip().lower() != "or"]
    rest = _BRACKETED_NAME.sub(lambda m: " " if m.group(1).strip().lower() != "or" else m.group(0), name or "")
    parts = [p.strip(" .,-") for p in _ALT_SPLIT.split(re.sub(r"\s{2,}", " ", rest)) if p and p.strip(" .,-")]
    return parts + [b for b in bracketed if b not in parts]


def match_all(medicines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Annotate extracted medicines in place-safe copies; returns (medicines, review notes)."""
    started = time.perf_counter()
    index = get_index()
    out, notes = [], []
    for med in medicines:
        item = dict(med)
        name = str(item.get("name") or "").strip()
        # "Dothepin / Prothiaden 75mg" is ONE medicine written with alternative brand names: match every
        # name and keep the best result (matched > known brand > verify > not found, then score).
        parts = split_names(name)
        alternatives = [a for a in (item.get("alternatives") or []) if a and a not in parts]
        names = parts + alternatives
        strength = item.get("strength") if STRENGTH_UNIT.search(str(item.get("strength") or "")) else None
        # the written form ("gum paint", "syrup") narrows the variant like the strength does
        form = item.get("form")
        strength = " ".join(x for x in (strength, form) if x) or None
        matches = [(n, match_medicine(n, strength, index)) for n in names] if names else [(name, match_medicine(name, strength, index))]
        best_name, m = max(matches, key=lambda nm: (STATUS_RANK.get(nm[1]["match_status"], 0), nm[1]["match_score"] or 0))
        item["name"] = names[0] if names else name
        item["alternatives"] = names[1:]
        item["alternative_matches"] = [{"name": n, "match_status": mm["match_status"], "suggested_name": mm["suggested_name"]}
                                       for n, mm in matches]
        if best_name != item["name"]:
            m = dict(m)
            m["notes"] = [*m["notes"], f"Matched through the alternative name '{best_name}'"]
        m = prefer_written_strength(m, item.get("strength"))
        m = check_variant(m, split_joined_variant(" ".join(names) if names else name, index) if index else name)
        add_uses(item, m)
        name = " / ".join(names) if len(names) > 1 else (names[0] if names else name)
        if item.get("strength") and item["strength"] not in name:
            name = f"{name} {item['strength']}"
        item["read_as"] = name
        # the reader's own confidence, before the grounding cap was applied
        item["reader_confidence"] = float(item.get("reader_confidence", item.get("confidence")) or 0.0)
        item["strength_variants"] = m.get("strength_variants")
        item["written_strength_in_brand"] = m.get("written_strength_in_brand")
        item["rival_score"] = m.get("rival_score")
        for key in ("match_status", "match_score", "suggested_name", "candidates", "db_info", "known_brand"):
            item[key] = m[key]
        item["non_medicine"] = bool(m.get("non_medicine"))
        item["confidence"] = combined_confidence(item["reader_confidence"], m["match_status"],
                                                 m["match_score"], bool(item.get("in_ocr_text", True)))
        item["variant_mismatch"] = bool(m.get("variant_mismatch"))
        item["confidence_level"], item["confidence_factors"] = confidence_level(item)
        notes.extend(m["notes"])
        out.append(item)
    log.info("Matched %d medicine(s) in %.0f ms (%s)", len(out), (time.perf_counter() - started) * 1000,
             ", ".join(f"{i['read_as']}={i['match_status']}" for i in out))
    return out, notes
