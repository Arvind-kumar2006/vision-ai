"""Nutri-Score, Updated Algorithm (Santé publique France, Exhibit 1-B; foods 2022, beverages 2023).

Every threshold comes from data/product_label/nutriscore_thresholds.json, which
records the source document, version and page numbers. Nothing is hard-coded here.

Inputs are per 100 g (foods) or per 100 mL (beverages). Category rules (which
products are cheese / red meat / fats-nuts-seeds / beverages / waters) are defined
in the official Q&A document, which the app does not have; ``category`` must be
supplied by the caller (default "general") and the result says which table was used.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "product_label" / "nutriscore_thresholds.json"

CATEGORIES = {"general", "cheese", "red_meat", "fats_nuts_seeds", "beverage", "water"}
KCAL_TO_KJ = 4.184
SODIUM_TO_SALT = 2.5


@lru_cache(maxsize=2)
def load_tables(path: str = str(DATA_PATH)) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points_exceeded(value: float, thresholds: List[float]) -> int:
    """Number of thresholds strictly exceeded (tables read 'n points: > threshold')."""
    return sum(1 for t in thresholds if value > t)


def points_at_least(value: float, thresholds: List[float]) -> int:
    """Number of thresholds reached (Table 7 saturates/lipids uses '<' upper bounds)."""
    return sum(1 for t in thresholds if value >= t)


def fvl_points(percent: float, spec: Dict[str, Any]) -> int:
    pts = 0
    for threshold, p in zip(spec["thresholds"], spec["points"]):
        if percent > threshold:
            pts = p
    return pts


def grade_for(score: int, grades: List[Dict[str, Any]]) -> str:
    for g in grades:
        if g["max"] is None or score <= g["max"]:
            return g["grade"]
    return grades[-1]["grade"]


def normalise_inputs(n: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Fill energy_kj / salt_g from kcal / sodium when only those are printed."""
    out = dict(n)
    conversions = []
    if out.get("energy_kj") is None and out.get("energy_kcal") is not None:
        out["energy_kj"] = round(out["energy_kcal"] * KCAL_TO_KJ, 1)
        conversions.append(f"energy {out['energy_kcal']} kcal x 4.184 = {out['energy_kj']} kJ")
    if out.get("salt_g") is None and out.get("sodium_mg") is not None:
        out["salt_g"] = round(out["sodium_mg"] / 1000 * SODIUM_TO_SALT, 3)
        conversions.append(f"salt = sodium {out['sodium_mg']} mg x 2.5 = {out['salt_g']} g")
    out["_conversions"] = conversions
    return out


def compute(per100: Dict[str, Optional[float]], category: str = "general",
            non_nutritive_sweeteners: bool = False, fvl_percent: Optional[float] = None) -> Dict[str, Any]:
    """Compute the Nutri-Score.

    per100 keys (per 100 g or 100 mL): energy_kj or energy_kcal, sugars_g,
    saturated_fat_g, salt_g or sodium_mg, protein_g, fibre_g, and for fats/nuts/seeds
    also fat_g (total lipids). Returns grade (or None), score, N, P, a breakdown of
    every component with its value and points, and notes (assumptions, missing data).
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    tables = load_tables()
    notes: List[str] = []
    if category == "water":
        return {"grade": tables["beverages"]["waters_grade"], "score": None, "n_points": 0, "p_points": 0,
                "category": category, "table": tables["beverages"]["table"], "breakdown": [],
                "notes": ["Waters are always graded A."]}

    key = {"beverage": "beverages", "fats_nuts_seeds": "fats_nuts_seeds"}.get(category, "general")
    t = tables[key]
    v = normalise_inputs(per100)
    notes += v.pop("_conversions")
    unit = "100 mL" if key == "beverages" else "100 g"

    # --- required negative inputs -------------------------------------------
    required = ["sugars_g", "saturated_fat_g", "salt_g"]
    if key == "fats_nuts_seeds":
        required.append("fat_g")  # Table 7 uses energy FROM SATURATES (saturates x 37), not total energy
    else:
        required.insert(0, "energy_kj")
    missing = [k for k in required if v.get(k) is None]
    if missing:
        return {"grade": None, "score": None, "n_points": None, "p_points": None, "category": category,
                "table": t["table"], "breakdown": [],
                "notes": notes + [f"Not scored: missing {', '.join(missing)} per {unit}."],
                "missing": missing}

    breakdown: List[Dict[str, Any]] = []

    def add(component: str, kind: str, value: Any, pts: int, unit_: str) -> int:
        breakdown.append({"component": component, "kind": kind, "value": value, "unit": unit_, "points": pts})
        return pts

    neg = t["negative"]
    n = 0
    if key == "fats_nuts_seeds":
        e_sat = round(v["saturated_fat_g"] * 37, 1)
        n += add("energy_from_saturates", "negative", e_sat, points_exceeded(e_sat, neg["energy_from_saturates_kj"]), "kJ")
        n += add("sugars", "negative", v["sugars_g"], points_exceeded(v["sugars_g"], neg["sugars_g"]), "g")
        ratio = round(100 * v["saturated_fat_g"] / v["fat_g"], 1) if v["fat_g"] else 0.0
        n += add("saturates_to_lipids", "negative", ratio, points_at_least(ratio, neg["saturates_to_lipids_percent"]), "%")
        n += add("salt", "negative", v["salt_g"], points_exceeded(v["salt_g"], neg["salt_g"]), "g")
    else:
        n += add("energy", "negative", v["energy_kj"], points_exceeded(v["energy_kj"], neg["energy_kj"]), "kJ")
        n += add("sugars", "negative", v["sugars_g"], points_exceeded(v["sugars_g"], neg["sugars_g"]), "g")
        n += add("saturated_fat", "negative", v["saturated_fat_g"],
                 points_exceeded(v["saturated_fat_g"], neg["saturated_fat_g"]), "g")
        n += add("salt", "negative", v["salt_g"], points_exceeded(v["salt_g"], neg["salt_g"]), "g")
        if key == "beverages":
            n += add("non_nutritive_sweeteners", "negative", bool(non_nutritive_sweeteners),
                     neg["non_nutritive_sweeteners_points"] if non_nutritive_sweeteners else 0, "")

    # --- positive inputs (missing -> 0 with a note) ----------------------------
    pos = t["positive"]
    protein = v.get("protein_g")
    fibre = v.get("fibre_g")
    if protein is None:
        notes.append("Protein not available: counted as 0 (score may be slightly pessimistic).")
    if fibre is None:
        notes.append("Fibre not available: counted as 0 (score may be slightly pessimistic).")
    if fvl_percent is None:
        notes.append("Fruit/vegetable/legume % is not on the label: counted as 0.")
    protein_pts = points_exceeded(protein or 0.0, pos["protein_g"])
    if category == "red_meat":
        capped = min(protein_pts, tables["general"]["red_meat_protein_max_points"])
        if capped != protein_pts:
            notes.append("Red meat: protein points capped at 2.")
        protein_pts = capped
    fibre_pts = points_exceeded(fibre or 0.0, pos["fibre_g"])
    fvl_pts = fvl_points(fvl_percent or 0.0, pos["fvl_percent"])

    rule = t["protein_rule"]
    limit = rule["exclude_protein_when_n_at_least"]
    protein_counted = not (limit is not None and n >= limit and category not in rule["except"])
    add("protein", "positive", protein, protein_pts if protein_counted else 0, "g")
    add("fibre", "positive", fibre, fibre_pts, "g")
    add("fruit_veg_legumes", "positive", fvl_percent, fvl_pts, "%")
    if not protein_counted:
        breakdown[-3]["note"] = f"not counted (N = {n} >= {limit})"
        notes.append(f"Protein not counted because the negative points total {n} (>= {limit}).")

    p = (protein_pts if protein_counted else 0) + fibre_pts + fvl_pts
    score = n - p
    grades = t["grades"]
    return {"grade": grade_for(score, grades), "score": score, "n_points": n, "p_points": p,
            "category": category, "table": t["table"], "basis": f"per {unit}",
            "breakdown": breakdown, "notes": notes}
