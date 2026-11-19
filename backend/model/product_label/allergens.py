"""Rule-based allergen detection (deterministic; no LLM).

Rules: data/product_label/allergens.json (keywords incl. Indian terms, exclusion
phrases, may-contain and negation markers, Open Food Facts tag mapping).

Inputs are the ingredients text and the printed "contains" and "may contain"
statements, as read from the photo.
Output: two lists, ``contains`` and ``may_contain``, each item naming the
allergen and the exact word that triggered it, plus explanatory notes.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "product_label"
RULES_PATH = DATA_DIR / "allergens.json"


@lru_cache(maxsize=4)
def load_rules(path: str = str(RULES_PATH)) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _phrase_regex(phrase: str) -> re.Pattern:
    """Whole-word, case-insensitive; spaces/hyphens inside the phrase are flexible."""
    parts = [re.escape(p) for p in re.split(r"[\s-]+", phrase.strip()) if p]
    return re.compile(r"(?<![a-z0-9])" + r"[\s-]+".join(parts) + r"(?![a-z0-9])", re.IGNORECASE)


def _spans(pattern: re.Pattern, text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text)]


def _split_may_contain(text: str, markers: Iterable[str]) -> Tuple[str, str]:
    """Split an ingredients text into (contains part, may-contain part).

    The may-contain part starts at the first marker ("may contain", "traces of",
    ...) and runs to the end of that sentence; the rest stays in contains.
    """
    lower = text.lower()
    best = None
    for marker in markers:
        m = _phrase_regex(marker).search(lower)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end())
    if best is None:
        return text, ""
    start = best[0]
    end_match = re.search(r"[.;\n]", text[best[1]:])
    end = best[1] + end_match.start() if end_match else len(text)
    return text[:start] + " " + text[end:], text[start:end]


def _is_negated(text: str, start: int, end: int, negations: Iterable[str]) -> bool:
    before = text[max(0, start - 20):start].lower().rstrip(" (-")  # "non-dairy" -> "non"
    after = text[end:end + 6].lower()
    if re.match(r"^\s*-?\s*free\b", after):
        return True
    for neg in negations:
        if re.search(r"(?<![a-z])" + re.escape(neg) + r"$", before):
            return True
    return False


def _find(text: str, rule: Dict[str, Any], negations: List[str]) -> Optional[Tuple[str, Optional[str]]]:
    """First non-excluded, non-negated trigger for one allergen in ``text``."""
    if not text:
        return None
    excluded: List[Tuple[int, int]] = []
    for phrase in rule.get("exclude_phrases", []):
        excluded += _spans(_phrase_regex(phrase), text)

    def ok(span: Tuple[int, int]) -> bool:
        s, e = span
        if any(es <= s and e <= ee for es, ee in excluded):
            return False
        return not _is_negated(text, s, e, negations)

    hits: List[Tuple[int, str, str]] = []  # (position, matched text, keyword)
    # longest keywords first so "skimmed milk powder" wins over "milk"
    for kw in sorted(rule.get("keywords", []), key=len, reverse=True):
        for span in _spans(_phrase_regex(kw), text):
            if ok(span):
                hits.append((span[0], text[span[0]:span[1]], kw))
                break
    for pat in rule.get("patterns", []):
        for m in re.finditer(pat, text, re.IGNORECASE):
            if ok(m.span()):
                hits.append((m.start(), text[m.start():m.end()], rule.get("pattern_keyword", "lecithin")))
                break
    if not hits:
        return None
    pos, matched, kw = min(hits, key=lambda h: (h[0], -len(h[1])))
    return matched.strip(), rule.get("notes", {}).get(kw)


def detect_allergens(
    ingredients_text: Optional[str] = None,
    contains_statement: Optional[str] = None,
    may_contain_statement: Optional[str] = None,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {"contains": [...], "may_contain": [...], "notes": [...]}.

    Each item: {"allergen", "label", "trigger", "source"} where source is one of
    ingredients | contains_statement | may_contain_statement.
    """
    rules = rules or load_rules()
    negations = rules.get("negation_markers", [])
    markers = rules.get("may_contain_markers", [])
    ingredients_main, ingredients_may = _split_may_contain(ingredients_text or "", markers)

    contains_sources = [("ingredients", ingredients_main), ("contains_statement", contains_statement or "")]
    may_sources = [("may_contain_statement", may_contain_statement or ""), ("ingredients", ingredients_may)]

    contains: Dict[str, Dict[str, Any]] = {}
    may: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    for key, rule in rules["allergens"].items():
        for source, text in contains_sources:
            found = _find(text, rule, negations)
            if found:
                contains[key] = {"allergen": key, "label": rule["label"], "trigger": found[0], "source": source}
                if found[1]:
                    notes.append(f"{rule['label']}: {found[1]}")
                break
        if key in contains:
            continue
        for source, text in may_sources:
            found = _find(text, rule, negations)
            if found:
                may[key] = {"allergen": key, "label": rule["label"], "trigger": found[0], "source": source}
                if found[1]:
                    notes.append(f"{rule['label']}: {found[1]}")
                break

    # Lecithin / INS 322 with no stated source: flag, don't decide.
    all_text = " ".join(t for t in (ingredients_text, contains_statement) if t)
    if "soy" not in contains and re.search(r"(?<![a-z0-9])(lecithin|(ins|e)\s*-?\s*322)(?![a-z0-9])", all_text, re.I):
        notes.append("Lecithin (INS 322) is listed without its source; it may come from soy. Check the label.")

    return {"contains": list(contains.values()), "may_contain": list(may.values()),
            "notes": list(dict.fromkeys(notes))}
