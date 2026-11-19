"""Deterministic nutrition-panel parsing from a label transcript (regex first, no LLM).

1. Find the column header(s) to learn the basis order: per 100 g / per 100 mL /
   per serving (and ignore %RDA / %GDA columns).
2. For every nutrient line, take the numbers (skipping percentages) in column order.
3. Convert units (mg -> g where the nutrient is in g, kJ <-> kcal kept separately)
   and produce per-100 values. Per-serving values are converted with the printed
   serving size; without a serving size, per-100 values are reported unavailable.
4. ``fill_from_llm`` adds values the regex missed from an LLM's nutrition_rows, but
   ONLY when that number literally appears in the transcript. Dormant: the compact
   reader (project_review/ADR-002) no longer asks for nutrition_rows.

Every value carries a source: "label_regex" or "label_llm".
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Row names per nutrient, English first, then the languages seen on real labels (Open Food Facts
# photos): French, German, Spanish, Italian, Dutch, Portuguese, Polish, Czech. Order matters:
# more specific names first ("saturated fat" before "fat", "added sugars" before "sugars").
NAMES: List[Tuple[str, List[str], str]] = [
    ("mufa", [r"mono[\s-]*unsaturated", r"mono[\s-]*insatur\w*", r"einfach\s*unges\w*", r"enkelvoudig\s*onverzadig\w*"], "g"),
    ("pufa", [r"poly[\s-]*unsaturated", r"poli[\s-]*insatur\w*", r"poly[\s-]*insatur\w*", r"mehrfach\s*unges\w*",
              r"meervoudig\s*onverzadig\w*"], "g"),
    ("saturated_fat", [r"saturated\s*(?:fat|fatty\s*acids?)", r"saturates", r"sat\.?\s*fat",
                        r"(?:ag|acides?\s*gras)\s*satur\w*", r"(?:[aá]cidos\s*grasos\s*|acidi\s*grassi\s*)?saturad?\w*",
                       r"acidi\s*grassi\s*satur\w*", r"gesättigte\s*fetts\w*", r"gesättigte", r"verzadigde\s*vetzuren", r"verzadigd\w*",
                       r"nasi[cč]en[ae]?\s*(?:masne\s*)?(?:kiseline|masti|fat)", r"zasi[cć]en[ae]?\s*(?:masne\s*)?(?:kiseline|masti|masti)",
                       r"kwasy\s*tłuszczowe\s*nasycone", r"nasycon\w*", r"nasycené\s*mastné\s*kyseliny"], "g"),
    ("trans_fat", [r"trans\s*(?:fat|fatty\s*acids?)", r"acides?\s*gras\s*trans", r"transfett\w*"], "g"),
    ("fat", [r"total\s*fat", r"fat\s*\(total\)", r"fats?", r"lipids?", r"lipides", r"matières\s*grasses", r"matieres\s*grasses",
              r"graisses", r"grasas(?:\s*totales)?", r"grassi", r"fett", r"fette", r"vetten", r"vet", r"gorduras(?:\s*totais)?", r"lípidos",
              r"tłuszcz", r"tuky", r"m\s*asti", r"masno[cć]e?", r"masti"], "g"),
    ("added_sugars", [r"added\s*sugars?", r"sucres\s*ajoutés", r"azúcares\s*añadidos"], "g"),
    ("sugars", [r"total\s*sugars?", r"sugars?", r"sucres?", r"zucker", r"azúcares", r"azucares", r"zuccheri", r"suikers",
                 r"açúcares", r"acucares", r"cukry", r"cukr\w*", r"sladkor\w*", r"[šs]e[cć]er\w*"], "g"),
    ("fibre", [r"(?:dietary|total)?\s*fib(?:re|er)s?(?:\s*alimentaires?)?", r"fibres?\s*alimentari", r"ballaststoffe",
               r"nahrungsfasern", r"fibra(?:\s*alimentaria)?", r"fibre", r"vezels", r"voedingsvezel\w*", r"błonnik", r"vláknina"], "g"),
    ("carbohydrate", [r"(?:total\s*)?carbohydrates?", r"carbs?", r"glucides", r"hydrates?\s*de\s*carbone", r"kohlenhydrate",
                       r"hidratos\s*de\s*carbono", r"carboidrati", r"koolhydraten", r"węglowodany", r"sacharidy",
                       r"hidratos", r"ogljikovi\s*hidrati", r"ugljeni\s*hidrati", r"ugljikohidrati"], "g"),
    ("protein", [r"proteins?", r"protéines", r"proteines", r"protéïnes", r"eiwei(?:ss|ß)", r"eiwitten", r"proteínas", r"proteinas",
                  r"proteine", r"białko", r"bílkoviny", r"beljakovin\w*", r"bjelan[cč]evin\w*", r"bjelan\w*"], "g"),
    ("cholesterol", [r"cholesterol", r"cholestérol"], "mg"),
    ("sodium", [r"sodium", r"natrium", r"sodio", r"sód"], "mg"),
    ("salt", [r"salt", r"sel", r"salz", r"sal", r"sale", r"zout", r"sól", r"sůl", r"sol", r"so[lľ]"], "g"),
    ("energy", [r"energy", r"énergie", r"energie", r"energía", r"energia", r"valeur\s*énergétique", r"valeur\s*energetique",
                r"valor\s*energético", r"valor\s*energetico", r"valore\s*energetico", r"brennwert", r"energetická\s*hodnota",
                r"wartość\s*energetyczna", r"calories", r"calorific\s*value", r"kcal"], "kcal"),
]
# Words that start a row but are not nutrients we score: used only to split inline text so their
# numbers never leak into a neighbouring nutrient ("Sel : 0,05 g - Calcium : 120 mg").
STOP_NAMES = [r"calcium", r"iron", r"fer", r"eisen", r"potassium", r"kalium", r"magnesium", r"magnésium", r"zinc",
              r"phosph\w*", r"vitamin\w*", r"vitamine\w*", r"vnr", r"ajr", r"polyols?", r"amidon", r"starch", r"stärke",
              r"lactose", r"omega[\s-]*\d"]
CONNECTORS = (r"of\s+which|including|incl\.?|dont|davon|de\s+las\s+cuales|de\s+los\s+cuales|de\s+las\s+que|di\s+cui|"
              r"waarvan|w\s+tym|z\s+toho|dos\s+quais|das\s+quais")


def _alt(names: List[str]) -> str:
    return "(?:" + "|".join(names) + ")"


# For row classification the name must START the row (see _nutrient_for).
NUTRIENTS: List[Tuple[str, str, str]] = [
    (key, r"(?<!\w)" + _alt(names) + r"(?!\w)", unit) for key, names, unit in NAMES]
CONNECTOR_RE = re.compile(r"^(?:" + CONNECTORS + r")\b[\s:,\-–]*", re.I)
# Inline segmentation anchors: every nutrient/stop name (except unit words like "kcal").
ANCHOR_RE = re.compile(
    r"(?<!\w)(?:(?:" + CONNECTORS + r")[\s:,\-–]*)?"
    + _alt([n for _, names, _ in NAMES for n in names if n not in (r"kcal",)] + STOP_NAMES) + r"(?!\w)",
    re.I)
NUMBER = r"(<\s*)?(\d+(?:[.,]\d+)?)"
UNIT = r"\s*(kcal|kj|mg|mcg|µg|g|ml)?"
VALUE_RE = re.compile(NUMBER + UNIT + r"(?!\s*%)", re.IGNORECASE)

PER = r"(?:per|pour|par|por|pro|je|para|na|w)"
SERVING_WORDS = r"(?:serve|serving|portion|porción|porcion|porzione|portie|porcja|porce|pack|piece|biscuit|bar|sachet|cup)"
BASIS_RE = [
    ("per_100g", re.compile(PER + r"\s*100\s*g(?!\w)|/\s*100\s*g(?!\w)|(?<![\d,.])100\s*g\s*(?:provides|product)?(?!\w)", re.I)),
    ("per_100ml", re.compile(PER + r"\s*100\s*ml(?!\w)|/\s*100\s*ml(?!\w)|(?<![\d,.])100\s*ml(?!\w)", re.I)),
    # column headers only: "Per serve (30 g)", "per serving", "pro Portion"; NOT "Serving size: 30 g"
    ("per_serving", re.compile(PER + r"\s*" + SERVING_WORDS + r"(?!\w)|\bper\s*\d+\s*(?:g|ml)\s*serv", re.I)),
]
RDA_RE = re.compile(r"%\s*(?:rda|gda|dv|ri)|rda\s*%|%\s*daily", re.I)
SERVING_RE = re.compile(
    r"(?:serving|serve)\s*size\s*[:\-]?\s*(?:approx\.?\s*)?(\d+(?:[.,]\d+)?)\s*(g|ml)\b"
    # "Serving size 1/4 cup shelled (36g)", "Serving size: 2 pieces 30 g" (also OCR-misspelt "Seving size"): a household
    # measure first, then the weight. Only a number directly followed by g / ml counts, so "1/4 cup" never does.
    r"|\bs\w{0,3}v\w{0,4}\s*size\b[^\n]{0,40}?(?<![\d/.,])(\d+(?:[.,]\d+)?)\s*(g|ml)\b"
    r"|per\s*(?:serve|serving)\s*\(?\s*(\d+(?:[.,]\d+)?)\s*(g|ml)\s*\)?"
    r"|(\d+(?:[.,]\d+)?)\s*(g|ml)\s*(?:per\s*)?serv(?:e|ing)\b",
    re.I)

PER100_KEYS = {
    "energy_kcal": "kcal", "energy_kj": "kJ", "protein_g": "g", "fat_g": "g", "saturated_fat_g": "g",
    "trans_fat_g": "g", "carbohydrate_g": "g", "sugars_g": "g", "added_sugars_g": "g", "fibre_g": "g",
    "sodium_mg": "mg", "salt_g": "g", "cholesterol_mg": "mg",
}


def _num(text: str) -> float:
    """'1,234' -> 1234 (thousands separator), '7,5' -> 7.5 (decimal comma), '12.5' -> 12.5."""
    text = text.strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
        return float(text.replace(",", ""))
    return float(text.replace(",", "."))


BASIS_STRIP = re.compile(
    r"(?:" + PER + r"|/)\s*100\s*(?:g|ml)(?!\w)\s*:?|"
    + PER + r"\s*" + SERVING_WORDS + r"\s*\(?\s*\d+(?:[.,]\d+)?\s*(?:g|ml)\s*\)?\s*:?",
    re.I)


def _clean(line: str) -> str:
    """Remove basis phrases ('Per 100 ml:', 'per serve (25 g)') so their numbers aren't read as values."""
    return BASIS_STRIP.sub(" ", line).strip()


def _value_start(line: str) -> int:
    m = re.search(r"<?\s*\d", line)
    return m.start() if m else len(line)


def find_serving_size(transcript: str) -> Optional[Dict[str, Any]]:
    m = SERVING_RE.search(transcript or "")
    if not m:
        return None
    groups = [g for g in m.groups() if g]
    value, unit = _num(groups[0]), groups[1].lower()
    return {"value": value, "unit": unit, "text": m.group(0).strip()}


def detect_bases(lines: List[str]) -> List[str]:
    """Column order (left to right) from the header line naming the MOST bases.

    Ties go to the earliest line. Serving-size statements are not headers.
    """
    best: List[Tuple[int, str]] = []
    for line in lines:
        found = []
        for basis, rx in BASIS_RE:
            m = rx.search(line)
            if m:
                found.append((m.start(), basis))
        if len(found) > len(best):
            best = found
    return [b for _, b in sorted(best)]


ENERGY_UNIT_RE = re.compile(r"\d\s*(?:kcal|kj)(?!\w)", re.I)
# reference-intake lines ("8400 kJ / 2000 kcal", "RI", "AR", "GDA") are never product values
REFERENCE_RE = re.compile(r"(?<!\w)(?:ri|ar|gda|rda|ajr|vnr|reference|référence|referenz|apport|intake|2000\s*kcal|8400\s*kj)(?!\w)", re.I)
SEGMENT_SPLIT = re.compile(r",\s+(?=[A-Za-z])|;|\|")


def _segments(line: str) -> List[str]:
    """Split inline nutrition text into one segment per nutrient.

    'Energy 43 kcal, Sugars 10.6 g, Salt 0.02 g' and
    'Energie : 739 kJ - Matières grasses : 11 g dont acides gras saturés : 10 g - Sel : 0,05 g'.
    Splits at every nutrient (or stop) name; only used when at least two segments carry
    numbers, so table rows like 'Energy (kcal) | 480 | 144', multilingual names like
    'Fett/lipides/grassi 7,2 g', and '1,234' stay whole.
    """
    starts = [m.start() for m in ANCHOR_RE.finditer(line)]
    pieces = [line[a:b] for a, b in zip(starts, starts[1:] + [len(line)])]
    if starts and ENERGY_UNIT_RE.search(line[:starts[0]]) and not REFERENCE_RE.search(line[:starts[0]]):
        # "Val nut(100g) : 299 Kcal, 1253 KJ, Lipides 12,6g ...": unnamed kcal/kJ values are energy
        pieces.insert(0, "Energy " + _clean(line[:starts[0]]))
    with_numbers = [p.strip(" -–•*:|/,") for p in pieces if re.search(r"\d", p)]
    named = [p for p in with_numbers if _nutrient_for(p) or ANCHOR_RE.match(p)]
    if len(named) >= 2:
        return named
    parts = [p.strip() for p in SEGMENT_SPLIT.split(line) if p.strip()]
    named = [p for p in parts if _nutrient_for(p) and re.search(r"\d", p)]
    return named if len(named) >= 2 else [line]


INGREDIENT_LINE = re.compile(r"^\s*(?:ingredients?|ingrédients|zutaten|ingredientes|ingredienti|ingrediënten|składniki|"
                             r"složení|contains|may\s+contain|peut\s+contenir|kann\s+spuren|allergen)", re.I)


def _nutrient_for(line: str) -> Optional[Tuple[str, str, str]]:
    """The nutrient a table row describes, or None.

    The nutrient name must START the row and the row label must be short, so an
    ingredients sentence like "...Sugar, Palm Oil, Emulsifier (322), Salt." is never
    read as "sugar 322" (a real bug caught by the tests).
    """
    line = _clean(line)
    if INGREDIENT_LINE.match(line):
        return None
    name_part = line[:_value_start(line)].lower().strip(" •*-–:|\t")
    name_part = CONNECTOR_RE.sub("", name_part).strip(" •*-–:|\t")
    # Multilingual row labels: "fett/lipides/grassi", "matières grasses / Fett / tuky / tłuszcz".
    # Try the first name, then the others: a garbled first name ("[illegible] grasses", a word cut
    # off at the pack's edge) must not drop the whole row when another language is readable.
    names = re.split(r"\s*/\s*", name_part)
    if len(names[0].split()) > 6 or names[0].count(",") > 1:
        return None
    for name in names:
        name = CONNECTOR_RE.sub("", name.strip(" •*-–:|\t")).strip(" •*-–:|\t")
        if not name or len(name.split()) > 6 or name.count(",") > 1:
            continue
        for key, rx, unit in NUTRIENTS:
            m = re.search(rx, name, re.I)
            if m and m.start() <= 2:
                return key, rx, unit
    return None


ENERGY_ALONE = re.compile(r"^\W*(?:calories|energy|energie|kcal)\W*$", re.I)
BARE_NUMBER_END = re.compile(r"(?:^|\s)(\d+(?:[.,]\d+)?)\s*$")


def join_split_energy(lines: List[str]) -> List[str]:
    """Large-print energy rows ("Calories 100" on US labels) are often read as two lines: "Amount per serving 100" +
    "Calories", or "Calories" + "100". A lone energy word takes the bare number that ends the line before (or is the
    whole line after), only when that line names no other nutrient."""
    out = list(lines)
    for i, line in enumerate(out):
        if not ENERGY_ALONE.match(line):
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(out):
                m = BARE_NUMBER_END.search(out[j])
                if m and not _nutrient_for(out[j]) and (j == i - 1 or out[j].strip() == m.group(1)):
                    word = "Calories" if "cal" in line.lower() else line.strip(" :")
                    out[i] = f"{word} {m.group(1)}" + (" kcal" if "cal" in line.lower() else "")
                    out[j] = out[j][:m.start()].strip() or ""
                    break
    return [l for l in out if l]


def parse_nutrition(transcript: str) -> Dict[str, Any]:
    """Parse the transcript into rows [{nutrient, value, unit, basis, less_than, line}]."""
    lines = join_split_energy([l.strip() for l in (transcript or "").splitlines() if l.strip()])
    bases = detect_bases(lines)
    assumed = False
    if not bases:
        # No column header was read. EU (Reg. 1169/2011 art. 32) requires the per-100 g/mL
        # declaration, given first; we take the first column as per 100 g and flag every value.
        bases, assumed = ["per_100g"], True
    rows: List[Dict[str, Any]] = []
    for line in (_clean(seg) for l in lines for seg in _segments(l)):
        hit = _nutrient_for(line)
        if not hit and ENERGY_UNIT_RE.search(line) and not ANCHOR_RE.search(line) \
                and not REFERENCE_RE.search(line) and not INGREDIENT_LINE.match(line):
            line = "Energy " + line  # a line of kJ/kcal values without a name is the energy row
            hit = _nutrient_for(line)
        if not hit:
            continue
        key, _, default_unit = hit
        header_unit = re.search(r"\((kcal|kj|mg|mcg|µg|g|ml)\)", line, re.I)
        values = []
        for m in VALUE_RE.finditer(line[_value_start(line):]):
            less, number, unit = m.group(1), m.group(2), (m.group(3) or "").lower()
            unit = unit or (header_unit.group(1).lower() if header_unit else default_unit)
            values.append({"value": _num(number), "unit": unit, "less_than": bool(less)})
        if not values:
            continue
        if key == "energy":
            # "Energy 2010 kJ / 480 kcal | 603 kJ / 144 kcal": pair kJ and kcal per column
            kcal = [v for v in values if v["unit"] == "kcal"]
            kj = [v for v in values if v["unit"] == "kj"]
            for unit_name, series in (("kcal", kcal), ("kJ", kj)):
                for i, v in enumerate(series):
                    rows.append({"nutrient": f"energy_{unit_name.lower()}", "value": v["value"], "unit": unit_name,
                                 "basis": bases[i] if i < len(bases) else "unknown",
                                 "less_than": v["less_than"], "line": line})
            continue
        for i, v in enumerate(values):
            rows.append({"nutrient": key, "value": v["value"], "unit": v["unit"],
                         "basis": bases[i] if i < len(bases) else "unknown",
                         "less_than": v["less_than"], "line": line})
    if assumed:
        for r in rows:
            r["assumed_basis"] = True
    return {"bases": bases, "rows": rows, "serving_size": find_serving_size(transcript), "assumed_basis": assumed}


def _to_key_and_value(nutrient: str, value: float, unit: str) -> Optional[Tuple[str, float]]:
    unit = unit.lower()
    if nutrient in ("energy_kcal", "energy_kj"):
        return nutrient, value
    if nutrient in ("sodium", "cholesterol"):
        return f"{nutrient}_mg", value * 1000 if unit == "g" else value / 1000 if unit in ("mcg", "µg") else value
    factor = {"g": 1.0, "mg": 0.001, "mcg": 1e-6, "µg": 1e-6}.get(unit, 1.0)
    if nutrient in ("mufa", "pufa"):
        return None
    return f"{nutrient}_g", round(value * factor, 4)


def to_per100(parsed: Dict[str, Any], source: str = "label_regex") -> Dict[str, Any]:
    """Per-100 values with sources. Prefers printed per-100 columns; else converts per serving."""
    serving = parsed.get("serving_size")
    per100: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    flags: Dict[str, List[str]] = {}
    notes: List[str] = []
    basis_unit = "100 mL" if "per_100ml" in parsed.get("bases", []) else "100 g"

    direct = [r for r in parsed["rows"] if r["basis"] in ("per_100g", "per_100ml")]
    serving_rows = [r for r in parsed["rows"] if r["basis"] == "per_serving"]
    for r in direct:
        kv = _to_key_and_value(r["nutrient"], r["value"], r["unit"])
        if kv and kv[0] not in per100:
            per100[kv[0]], sources[kv[0]] = kv[1], source
            if r["less_than"]:
                flags.setdefault(kv[0], []).append("printed as '<' value; upper bound used")
            if r.get("assumed_basis"):
                flags.setdefault(kv[0], []).append("column header not read; first column assumed to be per 100 g")
    used_serving = False
    if serving_rows:
        missing_rows = [r for r in serving_rows if (_to_key_and_value(r["nutrient"], 1, r["unit"]) or ("",))[0] not in per100]
        if missing_rows:
            if serving and serving["value"] > 0:
                factor = 100.0 / serving["value"]
                for r in missing_rows:
                    kv = _to_key_and_value(r["nutrient"], r["value"], r["unit"])
                    if kv and kv[0] not in per100:
                        per100[kv[0]] = round(kv[1] * factor, 3)
                        sources[kv[0]] = source
                        flags.setdefault(kv[0], []).append(
                            f"converted from per serving ({r['value']} {r['unit']} per {serving['value']:g} {serving['unit']})")
                        used_serving = True
                if serving["unit"] == "ml":
                    basis_unit = "100 mL"
            elif not direct:
                notes.append("Only per-serving values are printed and the serving size is missing: "
                             "per-100 g values unavailable.")
    if used_serving:
        notes.append(f"Some values were converted from per serving using the serving size "
                     f"({serving['value']:g} {serving['unit']}).")
    if parsed.get("assumed_basis") and per100:
        notes.append("The nutrition table's column header (per 100 g / per serving) was not read; the first column "
                     "is assumed to be per 100 g. Check against the label.")
    if not parsed["rows"]:
        notes.append("No nutrition values found in the transcript.")
    return {"per100": per100, "basis": f"per {basis_unit}", "field_sources": sources,
            "flags": flags, "notes": notes, "available": bool(per100)}


def per_serving_values(parsed: Dict[str, Any]) -> Dict[str, float]:
    """The values printed per serving (shown when per-100 g can't be worked out, e.g. "Serving size: 1 piece")."""
    out: Dict[str, float] = {}
    for r in parsed.get("rows") or []:
        if r["basis"] != "per_serving":
            continue
        kv = _to_key_and_value(r["nutrient"], r["value"], r["unit"])
        if kv and kv[0] not in out:
            out[kv[0]] = kv[1]
    return out


SERVING_TEXT = re.compile(r"\bs\w{0,3}v\w{0,4}\s*size\s*[:\-]?\s*([^\n]{1,40})", re.I)


def serving_text(transcript: str) -> Optional[str]:
    """What the label says a serving is ("1 piece"), for the message when it has no weight."""
    m = SERVING_TEXT.search(transcript or "")
    if not m:
        return None
    words = re.split(r"[,;]", m.group(1))[0].split()[:4]
    while len(words) > 2 and words[-1][:1].isupper():   # a capitalised word after "1 piece" is the next column
        words.pop()
    return " ".join(words).strip(" :,.") or None
