"""Plain-language explanation and generic tips, from the parsed label data only.

Guarantees, enforced in code (not just in the prompt):
- The LLM only ever sees the parsed facts, wrapped in delimited data blocks after
  sanitising (the transcript can contain anything printed on a package).
- Every number in an LLM answer must appear in those facts; otherwise the text is
  discarded (explanation -> deterministic template).
- Tips are generated deterministically from the Nutri-Score breakdown; there are no
  product recommendations.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from model.common.llm import LLM, AIBusy, data_block, default_llm, numbers_supported

log = logging.getLogger("product_label.explain")

UNAVAILABLE_NOTE = "detailed explanation unavailable right now"
FORBIDDEN = re.compile(r"\b(recommend|instead try|buy|switch to|better brand|cure|treat|prevent|detox)\w*", re.I)

EXPLAIN_SYSTEM = """You explain a packaged food's label to a shopper in plain language.
Use ONLY the facts in the FACTS block. Do not add any number, nutrient value, health claim,
product name or brand that is not in FACTS. Do not recommend products. Do not give medical advice.
If product_name_typed_by_shopper is set, call the product by that name (the shopper typed it; it was not read
from the label). Write 3 to 5 short sentences: what the product is, notable allergens, the Nutri-Score letter and
the main reasons from the breakdown, and any data-quality warnings. Plain text, no lists.
ALLERGENS: follow allergen_status exactly. If it is "unknown", say the allergens are unknown because the
ingredients list couldn't be read; never say there are no allergens. Say "no common allergens detected"
only when allergen_status is "none_detected"."""

def facts_for_llm(result: Dict[str, Any]) -> Dict[str, Any]:
    """The only data the LLM may use: parsed fields only (spec).
    The Nutri-Score breakdown is compacted to component/value/points (token budget).
    """
    ns = result.get("nutriscore") or {}
    facts = {
        "data_source": "the printed label (photo)",
        "product_name_typed_by_shopper": result.get("product_name"),
        "ingredients": result.get("ingredients"),
        "nutrition_per_100": result.get("nutrition_per_100"),
        "nutrition_basis": result.get("nutrition_basis"),
        "allergens": {k: [{"allergen": a["label"], "because_of": a["trigger"]} for a in v]
                      for k, v in (result.get("allergens") or {}).items() if k in ("contains", "may_contain")},
        "allergen_status": _allergen_status_fact(result),
        "ingredients_read": (result.get("allergens") or {}).get("ingredients_read", bool(result.get("ingredients_text"))),
        "additives": [{"code": a["code"], "name": a.get("name"), "classes": a.get("classes")}
                      for a in result.get("additives") or []],
        "nutriscore": {"grade": ns.get("grade"), "points": ns.get("points"), "category": ns.get("category"),
                       "breakdown": [[b.get("component"), b.get("value"), b.get("points")]
                                     for b in ns.get("breakdown") or []]},
        "data_quality_notes": result.get("sanity_notes") or [],
    }
    return facts


ALLERGEN_STATUS_FACTS = {
    "unknown": "unknown: the ingredients list could not be read, so allergens are unknown (NOT 'none')",
    "none_detected": "none_detected: the ingredients were read and no common allergens were found",
    "found": "found: see allergens",
}
ALLERGENS_UNKNOWN_ANSWER = "Allergens unknown: the ingredients list couldn't be read. Check the pack."
# "no allergens", "allergen-free", "does not contain any allergens" ...
NO_ALLERGEN_CLAIM = re.compile(
    r"\b(?:no|zero|without|free\s+(?:of|from))\b[^.?!]{0,40}\ballergen"
    r"|\ballergen[\s-]*free\b"
    r"|\ballergens?\b[^.?!]{0,30}\b(?:none|not\s+(?:present|found|listed|detected))\b", re.I)
# any claim that something is absent ("doesn't contain nuts", "gluten-free", "no milk")
ABSENCE_CLAIM = re.compile(
    r"\b(?:does\s*n[o']t|do\s*n[o']t|does\s+not|is\s+not\s+made\s+with)\s+(?:\w+\s+){0,2}(?:contain|include|have)"
    r"|\bfree\s+(?:of|from)\b|\b[a-z]+[\s-]free\b"
    r"|\bno\s+(?:\w+\s+){0,2}(?:allergens?|nuts?|milk|dairy|lactose|gluten|wheat|soy|soya|eggs?|peanuts?|sesame|fish|"
    r"shellfish|crustaceans?|mustard|sulphites?|sulfites?)\b", re.I)


def _allergen_status(result: Dict[str, Any]) -> str:
    al = result.get("allergens") or {}
    if al.get("status"):
        return al["status"]
    if al.get("contains") or al.get("may_contain"):
        return "found"
    return "none_detected" if (result.get("ingredients_text") or "").strip() else "unknown"


def _allergen_status_fact(result: Dict[str, Any]) -> str:
    return ALLERGEN_STATUS_FACTS[_allergen_status(result)]


def allergen_claim_ok(text: str, result: Dict[str, Any]) -> bool:
    """False when the text claims allergens are absent but the data can't support it."""
    status = _allergen_status(result)
    ingredients_read = (result.get("allergens") or {}).get("ingredients_read", status != "unknown")
    if status != "none_detected" and NO_ALLERGEN_CLAIM.search(text or ""):
        return False
    if not ingredients_read and ABSENCE_CLAIM.search(text or ""):
        return False
    return True


def tips_from_breakdown(nutriscore: Dict[str, Any], per100: Dict[str, Any], has_sweeteners: bool) -> List[str]:
    """Generic, deterministic tips. No product recommendations."""
    tips: List[str] = []
    unit = "100 mL" if (nutriscore or {}).get("category") == "beverage" else "100 g"
    parts = {b["component"]: b for b in (nutriscore or {}).get("breakdown") or []}
    rules = [
        ("sugars", 4, "sugars_g", "High in sugar ({v} g per {u}): compare sugar per {u} with similar products."),
        ("saturated_fat", 4, "saturated_fat_g", "High in saturated fat ({v} g per {u}): compare saturated fat per {u} with similar products."),
        ("saturates_to_lipids", 4, "saturated_fat_g", "A large share of the fat is saturated ({v} g per {u})."),
        ("salt", 5, "salt_g", "High in salt ({v} g per {u}): compare salt per {u} with similar products."),
        ("energy", 6, "energy_kj", "Energy-dense ({v} kJ per {u}): check the portion size."),
    ]
    if per100.get("energy_kcal") is not None:  # show the printed unit, not a conversion
        rules[-1] = ("energy", 6, "energy_kcal", "Energy-dense ({v} kcal per {u}): check the portion size.")
    for comp, min_points, key, text in rules:
        b = parts.get(comp)
        if b and b.get("points", 0) >= min_points:
            value = per100.get(key, b.get("value"))
            tips.append(text.format(v=f"{value:g}" if isinstance(value, (int, float)) else value, u=unit))
    fibre = parts.get("fibre")
    if fibre and fibre.get("value") is not None and fibre.get("points") == 0 and nutriscore.get("category") != "beverage":
        tips.append(f"Low in fibre: products with more fibre per {unit} score better.")
    if has_sweeteners and nutriscore.get("category") == "beverage":
        tips.append("Contains non-nutritive sweeteners, which lower the Nutri-Score of drinks.")
    if nutriscore.get("grade") is None:
        tips.append("Nutri-Score could not be calculated; check the printed nutrition table.")
    return tips


def template_explanation(result: Dict[str, Any]) -> str:
    ns = result.get("nutriscore") or {}
    parts = ["Summary of the label in your photo."]
    contains = [a["label"] for a in (result.get("allergens") or {}).get("contains", [])]
    may = [a["label"] for a in (result.get("allergens") or {}).get("may_contain", [])]
    if contains:
        parts.append("Contains: " + ", ".join(contains) + ".")
    if may:
        parts.append("May contain: " + ", ".join(may) + ".")
    status = _allergen_status(result)
    if status == "unknown":
        parts.append(ALLERGENS_UNKNOWN_ANSWER)
    elif status == "found" and not (result.get("allergens") or {}).get("ingredients_read", True):
        parts.append("The ingredients list couldn't be read, so other allergens may be present.")
    if ns.get("grade"):
        top = sorted((b for b in ns.get("breakdown") or [] if b["kind"] == "negative" and b["points"] > 0),
                     key=lambda b: -b["points"])[:2]
        why = " and ".join(b["component"].replace("_", " ") for b in top)
        parts.append(f"Health score (Nutri-Score) {ns['grade']}" + (f", mainly because of {why}." if why else "."))
    else:
        parts.append("The Nutri-Score could not be calculated from the available data.")
    if result.get("sanity_notes"):
        parts.append("Some values may be misread; check the printed label.")
    return " ".join(parts)


class Explainer:
    """Explanation through the app's AI client (OpenRouter free, Groq backup). ``LLM([])`` = no AI:
    the deterministic template is used."""

    def __init__(self, llm: Optional[LLM] = None) -> None:
        self.llm = llm if llm is not None else default_llm()

    def explain(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """{"text", "source": llm|template, "note"}."""
        facts = json.dumps(facts_for_llm(result), ensure_ascii=False, default=str)
        if not self.llm.available:
            return {"text": template_explanation(result), "source": "template", "note": "no AI provider configured"}
        try:
            text = self.llm.complete(EXPLAIN_SYSTEM, data_block("FACTS", facts, 9000), 400)["text"]
        except AIBusy:
            return {"text": template_explanation(result), "source": "template", "note": UNAVAILABLE_NOTE}
        if numbers_supported(text, facts) and not FORBIDDEN.search(text) and allergen_claim_ok(text, result):
            return {"text": text, "source": "llm", "note": None}
        return {"text": template_explanation(result), "source": "template",
                "note": "AI explanation rejected (unsupported number, recommendation or allergen claim)"}
