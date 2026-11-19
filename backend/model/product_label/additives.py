"""INS / E-number lookup in an ingredients list (deterministic; category only, no health claims).

Data: data/product_label/additives.json, extracted from the Open Food Facts additives
taxonomy (pinned commit recorded inside the file).

Recognised forms: "INS 322", "INS-322", "E322", "E 150d", "Emulsifier (322, 471)",
"Colour (150d)", "Acidity regulator (INS 330, 296)", and sub-items like "E420(ii)".
Intense sweeteners written by name ("sucralose", "aspartame", "steviol glycosides")
are also found, because the Nutri-Score drink rule needs them.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "product_label" / "additives.json"

CODE = r"\d{3,4}[a-z]?(?:\s*\(\s*[ivx]+\s*\))?"
PREFIXED = re.compile(r"(?<![a-z0-9])(?:ins|e)\s*-?\s*(" + CODE + r")(?![0-9])", re.I)
# a parenthesised list right after a class name: "Emulsifier (322, 471)", "Colour (150d)"
CLASS_LIST = re.compile(
    r"(?:emulsifiers?|stabili[sz]ers?|thickeners?|preservatives?|antioxidants?|colou?rs?|"
    r"acidity\s+regulators?|acids?|raising\s+agents?|sweeteners?|flavou?r\s+enhancers?|"
    r"humectants?|anti-?caking\s+agents?|firming\s+agents?|gelling\s+agents?|glazing\s+agents?|"
    r"sequestrants?|leavening\s+agents?|improvers?|class\s+ii\s+preservatives?)"
    r"\s*[:(]\s*\(?((?:(?:ins|e)\s*-?\s*)?" + CODE + r"(?:\s*[,&/]\s*(?:(?:ins|e)\s*-?\s*)?" + CODE + r")*)\)?",
    re.I)


# Distinctive name stems of the non-nutritive sweeteners listed in additives.json
# (taxonomy names: "Acesulfame k", "Cyclamic acid and its Na and Ca salts", ...).
# First words like "Salt" (E962) or "Rebaudioside" variants are deliberately not used.
NNS_NAME_STEMS = {
    "acesulfame": "E950", "aspartame": "E951", "cyclamic": "E952", "cyclamate": "E952",
    "saccharin": "E954", "sucralose": "E955", "alitame": "E956", "thaumatin": "E957",
    "glycyrrhizin": "E958", "neohesperidine": "E959", "steviol": "E960", "stevia": "E960",
    "neotame": "E961", "advantame": "E969",
}


@lru_cache(maxsize=2)
def load_additives(path: str = str(DATA_PATH)) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_code(raw: str) -> str:
    raw = re.sub(r"\s+", "", raw.lower())
    m = re.match(r"(\d{3,4}[a-z]?)(?:\(([ivx]+)\))?$", raw)
    if not m:
        return "E" + raw
    return "E" + m.group(1) + (f"({m.group(2)})" if m.group(2) else "")


def _lookup(code: str, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if code in table:
        return table[code]
    base = re.sub(r"\(.*\)$", "", code)  # E420(ii) -> E420
    return table.get(base)


def find_additives(ingredients_text: Optional[str], data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return {"additives": [...], "has_non_nutritive_sweetener": bool, "non_nutritive_sweeteners": [...]}."""
    data = data or load_additives()
    table = data["additives"]
    text = ingredients_text or ""
    found: Dict[str, Dict[str, Any]] = {}

    def add(code: str, matched: str) -> None:
        entry = _lookup(code, table)
        if entry is None:
            found.setdefault(code, {"code": code, "ins": code[1:], "name": None, "classes": [],
                                    "matched_text": matched.strip(), "known": False})
            return
        found.setdefault(entry["code"], {
            "code": entry["code"], "ins": entry["ins"], "name": entry["name"],
            "classes": entry.get("classes", []), "matched_text": matched.strip(), "known": True,
            "non_nutritive_sweetener": bool(entry.get("non_nutritive_sweetener", False)),
        })

    for m in PREFIXED.finditer(text):
        add(_norm_code(m.group(1)), m.group(0))
    for m in CLASS_LIST.finditer(text):
        for raw in re.split(r"\s*[,&/]\s*", m.group(1)):
            raw = re.sub(r"^(?:ins|e)\s*-?\s*", "", raw.strip(), flags=re.I)
            if re.fullmatch(CODE, raw, re.I):
                add(_norm_code(raw), m.group(0))

    # intense sweeteners written by name (needed for the Nutri-Score drink rule)
    for stem, code in NNS_NAME_STEMS.items():
        if code in table and re.search(r"(?<![a-z])" + stem + r"[a-z]*", text, re.I):
            m = re.search(r"(?<![a-z])" + stem + r"[a-z]*(?:\s+(?:potassium|k|sodium|calcium))?", text, re.I)
            add(code, m.group(0))

    items = sorted(found.values(), key=lambda x: x["code"])
    nns = [i["code"] for i in items if i.get("non_nutritive_sweetener")]
    return {"additives": items, "has_non_nutritive_sweetener": bool(nns), "non_nutritive_sweeteners": nns}
