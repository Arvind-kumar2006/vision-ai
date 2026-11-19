"""Nutrition sanity checks, applied separately to label values and Open Food Facts values.

Checks (per 100 g / 100 mL):
  1. energy ≈ 4·protein + 4·carbohydrate + 9·fat (kcal), within ±20 %
     (only when energy, protein, carbohydrate and fat are all present and the
     computed energy is above 40 kcal, so near-zero products like water are skipped)
  2. when both are given, kJ ≈ kcal × 4.184 (±10 %)
  3. no nutrient above 100 g per 100 g (sodium above 100 000 mg)
  4. sugars, added sugars and fibre not above total carbohydrate;
     saturated + trans fat not above total fat; added sugars not above sugars

Any failure produces the user-facing note "values may be misread or incorrect in the
database" plus a precise detail per failed check. Polyols, alcohol and organic acids
also carry energy, so a small energy mismatch can be genuine; ±20 % absorbs most of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

NOTE = "values may be misread or incorrect in the database"
GRAM_KEYS = ["protein_g", "fat_g", "saturated_fat_g", "trans_fat_g", "carbohydrate_g", "sugars_g",
             "added_sugars_g", "fibre_g", "salt_g"]
KCAL_PER_KJ = 1 / 4.184
TOLERANCE = 0.20
MIN_KCAL_FOR_ENERGY_CHECK = 40.0


def _kcal(per100: Dict[str, float]) -> Optional[float]:
    if per100.get("energy_kcal") is not None:
        return float(per100["energy_kcal"])
    if per100.get("energy_kj") is not None:
        return float(per100["energy_kj"]) * KCAL_PER_KJ
    return None


def check(per100: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Return {"ok", "source", "failures": [{"check", "detail"}], "note"}."""
    v = {k: float(x) for k, x in (per100 or {}).items() if isinstance(x, (int, float))}
    failures: List[Dict[str, str]] = []

    kcal = _kcal(v)
    if kcal is not None and all(v.get(k) is not None for k in ("protein_g", "carbohydrate_g", "fat_g")):
        computed = 4 * v["protein_g"] + 4 * v["carbohydrate_g"] + 9 * v["fat_g"]
        if computed > MIN_KCAL_FOR_ENERGY_CHECK or kcal > MIN_KCAL_FOR_ENERGY_CHECK:
            diff = abs(kcal - computed) / max(computed, 1.0)
            if diff > TOLERANCE:
                failures.append({"check": "energy_balance",
                                 "detail": f"energy {kcal:.0f} kcal vs 4·protein + 4·carbs + 9·fat = {computed:.0f} kcal "
                                           f"({diff:.0%} apart, limit 20 %)"})

    if v.get("energy_kj") is not None and v.get("energy_kcal") is not None and v["energy_kcal"] > 0:
        expected_kj = v["energy_kcal"] * 4.184
        if abs(v["energy_kj"] - expected_kj) / max(expected_kj, 1.0) > 0.10:
            failures.append({"check": "energy_units",
                             "detail": f"{v['energy_kj']:g} kJ does not match {v['energy_kcal']:g} kcal "
                                       f"(= {expected_kj:.0f} kJ)"})

    for key in GRAM_KEYS:
        if v.get(key) is not None and v[key] > 100:
            failures.append({"check": "over_100g", "detail": f"{key.replace('_g', '')} {v[key]:g} g per 100 g is impossible"})
    if v.get("sodium_mg") is not None and v["sodium_mg"] > 100000:
        failures.append({"check": "over_100g", "detail": f"sodium {v['sodium_mg']:g} mg per 100 g is impossible"})

    carbs = v.get("carbohydrate_g")
    if carbs is not None:
        for key in ("sugars_g", "added_sugars_g", "fibre_g"):
            if v.get(key) is not None and v[key] > carbs + 0.05:
                failures.append({"check": f"{key.replace('_g', '')}_above_carbohydrate",
                                 "detail": f"{key.replace('_g', '').replace('_', ' ')} {v[key]:g} g is above total "
                                           f"carbohydrate {carbs:g} g"})
    fat = v.get("fat_g")
    if fat is not None:
        sub = (v.get("saturated_fat_g") or 0) + (v.get("trans_fat_g") or 0)
        if sub > fat + 0.05:
            failures.append({"check": "fat_parts_above_fat",
                             "detail": f"saturated + trans fat {sub:g} g is above total fat {fat:g} g"})
    if v.get("added_sugars_g") is not None and v.get("sugars_g") is not None and v["added_sugars_g"] > v["sugars_g"] + 0.05:
        failures.append({"check": "added_sugars_above_sugars",
                         "detail": f"added sugars {v['added_sugars_g']:g} g is above total sugars {v['sugars_g']:g} g"})

    return {"ok": not failures, "source": source, "failures": failures, "note": NOTE if failures else None}
