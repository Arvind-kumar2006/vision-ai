"""Score the medicine-name matcher (model/prescription/medicine_db.py) on scripts/matcher_cases.json.

Per case, by what the result card would show:
    correct   Matched / Known brand, and the chosen product contains the expected ingredient
    verify    Verify, and one of the 3 candidates shown contains it (the person can pick it)
    wrong     Matched / Known brand with a product that does NOT contain it, or any confident match for a
              non-medicine / brand missing from the database: the dangerous outcome
    missed    everything else (not found, or the right product isn't among the candidates)

    venv/bin/python scripts/eval_matcher.py [--heldout] [--show wrong,missed,verify] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

CASES = Path(__file__).with_name("matcher_cases.json")
HELDOUT = Path(__file__).with_name("matcher_cases_heldout.json")


def _comp(m: dict) -> str:
    return ((m.get("db_info") or {}).get("composition") or (m.get("known_brand") or {}).get("composition")
            or ((m.get("candidates") or [{}])[0].get("composition")) or "").lower()


def outcome(case: dict, m: dict) -> str:
    want, status = case["ingredient"], m["match_status"]
    confident = status in ("matched", "known_brand")
    if want is None:
        return "wrong" if confident else "correct"
    if confident:
        return "correct" if want in _comp(m) else "wrong"
    if status == "verify" and any(want in (c.get("composition") or "").lower() for c in m.get("candidates", [])[:3]):
        return "verify"
    return "missed"


def run(cases):
    from model.prescription.medicine_db import match_all
    rows = []
    for c in cases:
        med = {"name": c["written"], "strength": c["strength"], "confidence": 0.9, "in_ocr_text": True}
        m = match_all([med])[0][0]
        rows.append({**c, "status": m["match_status"], "product": (m.get("candidates") or [{}])[0].get("name")
                     if m["match_status"] != "known_brand" else (m.get("known_brand") or {}).get("name"),
                     "composition": _comp(m), "outcome": outcome(c, m)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default="wrong,missed")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--heldout", action="store_true", help="score the held-out set (never used for tuning)")
    a = ap.parse_args()
    import logging
    logging.disable(logging.INFO)
    rows = run(json.loads((HELDOUT if a.heldout else CASES).read_text())["cases"])
    total = Counter(r["outcome"] for r in rows)
    by_kind = defaultdict(Counter)
    for r in rows:
        by_kind[r["kind"]][r["outcome"]] += 1
    show = set(a.show.split(","))
    for r in rows:
        if r["outcome"] in show:
            print(f"  {r['outcome']:7} [{r['kind']}] {r['written']!r} {r['strength'] or ''} -> {r['status']}: "
                  f"{r['product']} ({r['composition'][:60]}) expected {r['ingredient']}")
    n = len(rows)
    print(f"\n{n} cases: correct {total['correct']}, verify (right one offered) {total['verify']}, "
          f"WRONG {total['wrong']}, missed {total['missed']}")
    for k, c in by_kind.items():
        print(f"  {k:28} " + ", ".join(f"{o} {c[o]}" for o in ("correct", "verify", "wrong", "missed")))
    if a.json:
        a.json.write_text(json.dumps({"summary": dict(total), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
