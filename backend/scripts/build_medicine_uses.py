"""Build data/medicines/uses_by_ingredient.json: what each ingredient (salt) is used for.

Input data
- Medicines-usage CSV (use0-use4, side effects and classes per medicine NAME; no composition column).
  Raw CSV (89 MB) lives in data/medicines/250k_raw/ and is git-ignored.
- Indian Medicine Dataset (data/medicines/updated_indian_medicine_data.csv) gives each name's composition,
  so uses are collected per ingredient (and per ingredient combination). Its medicine_desc is the fallback.

The output is derived from a CC BY-SA 4.0 dataset and is therefore licensed CC BY-SA 4.0 as well.

    venv/bin/python scripts/build_medicine_uses.py
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "medicines" / "250k_raw" / "medicine_dataset.csv"
OURS = ROOT / "data" / "medicines" / "updated_indian_medicine_data.csv"
OUT = ROOT / "data" / "medicines" / "uses_by_ingredient.json"

sys.path.insert(0, str(ROOT))
from model.prescription.uses import ingredient_key, ingredients_of  # noqa: E402

csv.field_size_limit(10**7)


def clean_use(text: str) -> str | None:
    """'Treatment of Insomnia' -> 'Insomnia'; scraped junk ('$name is thought to work ... Depression') -> 'Depression'."""
    t = " ".join((text or "").split())
    if not t or t.upper() == "NA":
        return None
    if "$name" in t or len(t.split()) > 8:
        tail = t.rsplit(".", 1)[-1].strip()       # the real use is left at the end of the scraped paragraph
        t = tail if tail and len(tail.split()) <= 6 else ""
    t = re.sub(r"^(treatment|prevention) of\s+", lambda m: "" if m.group(1).lower() == "treatment" else "prevention of ", t, flags=re.I)
    t = t.strip(" .,")
    if not t or len(t.split()) > 6:
        return None
    return t[0].upper() + t[1:]


def first_sentence(text: str) -> str | None:
    t = " ".join((text or "").split())
    m = re.search(r"\bis (?:a medicine )?used (?:to|for|in) ([^.]+)\.", t)
    if m:
        words = m.group(1).split()
        return " ".join(words[:12]).rstrip(",;") if words else None
    return None


def main() -> None:
    if not RAW.exists():
        sys.exit(f"Missing {RAW}. Download the 250k dataset first (see this file's docstring).")
    composition = {}
    fallback = collections.defaultdict(collections.Counter)
    with OURS.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            ing = ingredients_of(" + ".join(x for x in (r["short_composition1"], r["short_composition2"]) if x and x.strip()))
            if not ing:
                continue
            composition[r["name"].lower().strip()] = ing
            desc = first_sentence(r.get("medicine_desc", ""))
            if desc:
                fallback[ingredient_key(ing)][desc] += 1

    uses = collections.defaultdict(collections.Counter)
    effects = collections.defaultdict(collections.Counter)
    classes = collections.defaultdict(collections.Counter)
    matched = 0
    with RAW.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            ing = composition.get(r["name"].lower().strip())
            if not ing:
                continue
            matched += 1
            key = ingredient_key(ing)
            for i in range(5):
                u = clean_use(r.get(f"use{i}", ""))
                if u:
                    uses[key][u] += 1
            for i in range(42):
                s = (r.get(f"sideEffect{i}") or "").strip()
                if s:
                    effects[key][s] += 1
            c = (r.get("Therapeutic Class") or "").strip()
            if c and c.upper() != "NA":
                classes[key][c.title()] += 1

    out = {}
    for key in set(uses) | set(fallback):
        entry = {}
        if uses.get(key):
            entry["uses"] = [u for u, _ in uses[key].most_common(4)]
            entry["source"] = "250k"
        elif fallback.get(key):
            entry["uses"] = [fallback[key].most_common(1)[0][0]]
            entry["source"] = "indian_medicine_dataset"
        if effects.get(key):
            entry["side_effects"] = [s for s, _ in effects[key].most_common(5)]
        if classes.get(key):
            entry["class"] = classes[key].most_common(1)[0][0]
        if entry.get("uses"):
            out[key] = entry
    doc = {
        "_licence": "CC BY-SA 4.0",
        "_source": ('Derived from "250k Medicines Usage, Side Effects and Substitutes" by Shudhanshu Singh (Kaggle, '
                    "CC BY-SA 4.0), joined to the Indian Medicine Dataset by medicine name; fallback: that dataset's "
                    "medicine descriptions."),
        "_built_from": {"250k_rows_matched_to_our_names": matched, "our_medicines": len(composition)},
        "ingredients": dict(sorted(out.items())),
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    singles = sum(1 for k in out if " + " not in k)
    print(f"matched {matched} of the 250k rows to our names; {len(out)} ingredient keys ({singles} single ingredients); "
          f"{OUT.stat().st_size / 1e6:.2f} MB -> {OUT}")


if __name__ == "__main__":
    main()
