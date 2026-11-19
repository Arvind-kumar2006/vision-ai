"""Evaluate prescription medicine IDENTIFICATION (the feature's goal since 2026-11-17) on labelled prescriptions.

Ground truth (git-ignored, contains patient data): data/eval/prescriptions/ground_truth.json
    {"p5.jpg": {"image": "/path/to/p5.jpg",            # optional; default: next to ground_truth.json
                "split": "tuned",                         # tuned (rules were developed on it) | unseen (default)
                "filled": true,                           # false = ground truth not written yet: skipped
                "medicines": [{"aliases": ["dothepin", "prothiaden", "dosulepin"], "strength": "75mg"}, ...],
                "already_taking": ["ecosprin av"]}}
    aliases: the brand as written, other brands of it, and the ingredient (generic) name. Other keys (dose, when,
    duration) are ignored.

Per prescription:
    found        a listed medicine whose name matches an alias (one-to-one, fuzzy >= 0.8)
    identified   found AND Matched / Known brand to a database product whose name or ingredients contain an alias
    to confirm   found AND Verify with a candidate containing an alias (the person can pick it)
    WRONG        Matched / Known brand to a product that contains no alias: the dangerous outcome
    junk rows    listed medicines that match nothing in the truth
    strength     written strength read correctly

    venv/bin/python scripts/eval_prescription.py [--only p5.jpg] [--json report.json] [--ai-cache DIR] [--no-ai]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EVAL_DIR = BACKEND_ROOT / "data" / "eval" / "prescriptions"


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _words(m: Dict[str, Any]) -> set:
    names = [m.get("name"), m.get("read_as"), m.get("suggested_name"), *(m.get("alternatives") or []),
             *[c.get("name") for c in (m.get("candidates") or [])[:1]]]
    words = {w.lower().strip(".,()") for n in names if n for w in re.split(r"[\s/]+", n) if len(w) >= 4}
    return words | {re.sub(r"\s+", " ", n.lower()).strip() for n in names if n}   # whole names, for multi-word aliases


def _product_text(m: Dict[str, Any]) -> str:
    """Name + ingredients of the product the result card confirms (empty when nothing is confirmed)."""
    if m.get("match_status") == "known_brand":
        kb = m.get("known_brand") or {}
        return f"{kb.get('name') or ''} {kb.get('composition') or ''}".lower()
    if m.get("match_status") == "matched":
        c = (m.get("candidates") or [{}])[0]
        return f"{c.get('name') or ''} {(m.get('db_info') or {}).get('composition') or c.get('composition') or ''}".lower()
    return ""


def _has_alias(text: str, aliases) -> bool:
    return any(a.lower() in text for a in aliases)


def score(result: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, Any]:
    meds = result.get("medicines") or []
    want = truth["medicines"]
    left = list(range(len(want)))
    detail = []
    counts = {"found": 0, "identified": 0, "to_confirm": 0, "wrong": 0, "strength_ok": 0, "strength_total": 0}
    for m in meds:
        product = _product_text(m)
        words = _words(m)
        hit = next((i for i in left if any(_sim(w, a) >= 0.8 for w in words for a in want[i]["aliases"])), None)
        if product and (hit is None or not _has_alias(product, want[hit]["aliases"])):
            # confirmed as a product that is not this medicine (or that matches nothing on the page)
            counts["wrong"] += 1
        if hit is None:
            detail.append({"read_as": m.get("read_as") or m.get("name"), "match": None, "status": m.get("match_status"),
                           "product": product or None})
            continue
        left.remove(hit)
        t = want[hit]
        counts["found"] += 1
        outcome = "missed"
        if product and _has_alias(product, t["aliases"]):
            outcome = "identified"
        elif product:
            outcome = "WRONG"
        elif m.get("match_status") == "verify" and any(
                _has_alias(f"{c.get('name') or ''} {c.get('composition') or ''}".lower(), t["aliases"])
                for c in (m.get("candidates") or [])[:3]):
            outcome = "to_confirm"
        counts["identified"] += outcome == "identified"
        counts["to_confirm"] += outcome == "to_confirm"
        strength_ok = None
        if t.get("strength"):
            counts["strength_total"] += 1
            strength_ok = re.sub(r"\s", "", (m.get("strength") or "").lower()) == t["strength"]
            counts["strength_ok"] += strength_ok
        detail.append({"read_as": m.get("read_as") or m.get("name"), "match": t["aliases"][0], "status": m.get("match_status"),
                       "outcome": outcome, "product": product or None, "strength": m.get("strength"),
                       "strength_ok": strength_ok, "what_for": m.get("what_for")})
    taking = [a.lower() for a in result.get("already_taking") or []]
    taking_ok = all(any(_sim(x, y) >= 0.75 or y.split()[0] in x for x in taking) for y in truth.get("already_taking", []))
    n = len(want)
    return {"medicines": n, "found": f"{counts['found']}/{n}", "identified": f"{counts['identified']}/{n}",
            "to_confirm": counts["to_confirm"], "wrong": counts["wrong"],
            "junk_rows": sum(1 for d in detail if d["match"] is None),
            "strength_correct": f"{counts['strength_ok']}/{counts['strength_total']}", "already_taking_ok": taking_ok,
            "_counts": counts, "per_medicine": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", type=Path, default=EVAL_DIR / "ground_truth.json")
    ap.add_argument("--only", help="evaluate only this entry")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--ai-cache", type=Path, help="keep AI answers here so repeated runs make no new AI requests")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()
    if not args.truth.exists():
        print(f"No ground truth at {args.truth}. See this file's docstring.")
        return 1
    from PIL import Image
    from model.common.llm import LLM
    from model.prescription.pipeline import PrescriptionPipeline
    from scripts.eval_common import EvalLLM
    llm = EvalLLM(args.ai_cache, LLM([]) if args.no_ai else None)
    pipeline = PrescriptionPipeline(llm=llm)
    entries = json.loads(args.truth.read_text())
    report = {}
    skipped = []
    for key, truth in entries.items():
        if args.only and key != args.only:
            continue
        if truth.get("filled") is False or not truth.get("medicines"):
            skipped.append(key)   # ground truth not written yet
            continue
        image = Path(truth.get("image") or args.truth.parent / key).expanduser()
        before = llm.requests + llm.cached
        t = time.perf_counter()
        r = pipeline.run(Image.open(image)).model_dump()
        ms = int((time.perf_counter() - t) * 1000)
        report[key] = {"split": truth.get("split", "unseen"), "status": r["status"], "latency_ms": ms, "timings": r["timings"],
                       "ai_calls": llm.requests + llm.cached - before, "already_taking": r["already_taking"],
                       "score": score(r, truth)}
        s = report[key]["score"]
        print(f"[{report[key]['split']}] {key}: identified {s['identified']}, to confirm {s['to_confirm']}, WRONG {s['wrong']}, "
              f"found {s['found']}, junk {s['junk_rows']}, strength {s['strength_correct']}, already taking "
              f"{'ok' if s['already_taking_ok'] else 'missed'}, {ms} ms, status {r['status']}")
        for d in s["per_medicine"]:
            print("   ", {k: v for k, v in d.items() if v not in (None, [], "")})
    if skipped:
        print("skipped (ground truth not filled yet):", ", ".join(skipped))
    tot = lambda k: sum(v["score"]["_counts"][k] for v in report.values())
    n_meds = sum(v["score"]["medicines"] for v in report.values())
    summary = {"prescriptions": len(report), "medicines": n_meds, "identified": tot("identified"),
               "to_confirm": tot("to_confirm"), "wrong": tot("wrong"), "found": tot("found"), "skipped": skipped, "ai_requests": llm.requests, "ai_cached_answers": llm.cached,
               "by_split": {sp: sorted(k for k, v in report.items() if v["split"] == sp) for sp in ("tuned", "unseen")},
               "latency_ms_mean": round(statistics.mean(v["latency_ms"] for v in report.values())) if report else None}
    print(summary)
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "per_prescription": report}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
