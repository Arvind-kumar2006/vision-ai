"""What a medicine is for, looked up by INGREDIENT (salt), so every brand with that ingredient gets it.

Data: data/medicines/uses_by_ingredient.json (built by scripts/build_medicine_uses.py); fallback for
ingredients it doesn't cover: the Indian Medicine Dataset's own descriptions.

The text always comes from that data. No AI writes or rewrites medicine descriptions; the line is only
shortened (whole phrases, about 12 words at most).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

PATH = Path(__file__).resolve().parents[2] / "data" / "medicines" / "uses_by_ingredient.json"
NO_DESCRIPTION = "No description available. Ask your pharmacist what this medicine is for."
SOURCE_LABEL = {"250k": "250k Medicines Usage, Side Effects and Substitutes (Shudhanshu Singh, CC BY-SA 4.0)",
                "indian_medicine_dataset": "Indian Medicine Dataset"}
MAX_WORDS = 12

# Same ingredient, different names (INN vs older/US names). Keys and values are lowercase.
SYNONYMS = {
    "dothiepin": "dosulepin", "acetaminophen": "paracetamol", "albuterol": "salbutamol",
    "levalbuterol": "levosalbutamol", "glyceryl trinitrate": "nitroglycerin", "frusemide": "furosemide",
    "lignocaine": "lidocaine", "adrenaline": "epinephrine", "noradrenaline": "norepinephrine",
    "amoxycillin": "amoxicillin", "cephalexin": "cefalexin", "rifampin": "rifampicin",
    "vitamin c": "ascorbic acid", "vitamin b12": "methylcobalamin",
}


def _norm_one(name: str) -> str:
    n = re.sub(r"\(.*?\)", " ", name or "")              # "(75mg)"
    n = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|%|w/w|w/v)\b", " ", n, flags=re.I)
    n = " ".join(n.lower().replace("-", " ").split()).strip(" ,.;")
    return SYNONYMS.get(n, n)


def ingredients_of(composition: str) -> List[str]:
    """'Domperidone (30mg) + Rabeprazole (20mg)' -> ['domperidone', 'rabeprazole'] (normalised, sorted)."""
    parts = re.split(r"\s*\+\s*|\s*,\s*(?=[A-Za-z])", composition or "")
    return sorted({p for p in (_norm_one(x) for x in parts) if p})


def ingredient_key(ingredients: List[str]) -> str:
    return " + ".join(sorted(ingredients))


@lru_cache(maxsize=1)
def _table() -> Dict[str, Any]:
    if not PATH.exists():
        return {}
    return json.loads(PATH.read_text()).get("ingredients", {})


def lookup(composition: Optional[str]) -> Optional[Dict[str, Any]]:
    """{"uses", "side_effects", "class", "source"} for a composition; combinations fall back to combining
    each ingredient's uses. None when nothing is known."""
    ing = ingredients_of(composition or "")
    if not ing:
        return None
    table = _table()
    hit = table.get(ingredient_key(ing))
    if hit:
        return hit
    parts = [table[i] for i in ing if i in table]
    if not parts:
        return None
    uses: List[str] = []
    for p in parts:
        for u in p.get("uses", [])[:2]:
            if u not in uses:
                uses.append(u)
    effects = [s for p in parts for s in p.get("side_effects", [])[:3]]
    return {"uses": uses, "side_effects": list(dict.fromkeys(effects))[:5],
            "source": "250k" if any(p.get("source") == "250k" for p in parts) else parts[0].get("source"),
            "class": parts[0].get("class")}


def what_for(info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Short line from the database uses: 'anxiety disorder and depression'. Whole phrases only, <= 12 words."""
    if not info or not info.get("uses"):
        return None
    chosen: List[str] = []
    uses: List[str] = []
    for u in info["uses"]:   # drop a phrase contained in another ("heart attack" inside "heart attack and stroke")
        low = u.lower()
        if any(low in o.lower() for o in info["uses"] if o.lower() != low):
            continue
        if low not in [x.lower() for x in uses]:
            uses.append(u)
    for u in uses:
        if len(" ".join(chosen + [u]).split()) > MAX_WORDS:
            break
        chosen.append(u)
    if not chosen:
        chosen = [" ".join(uses[0].split()[:MAX_WORDS])]
    phrases = [c if c[:2].isupper() else c[0].lower() + c[1:] for c in chosen]   # keep acronyms like "GERD"
    text = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return text[0].upper() + text[1:]
