"""Evaluate the food label pipeline on labelled photos (ADR-005: RapidOCR -> parser -> AI only for missing values).

Data (git-ignored; format in data/eval/product_labels/README.md):
    backend/data/eval/product_labels/*.jpg|jpeg|png + ground_truth.json
    {"file.jpg": {"allergens": ["milk", "wheat_gluten"], "sugar_g_per_100": 12.5,
                  "energy_kcal_per_100": 480, "barcode": "8901063010222"}}

Run from backend/:
    venv/bin/python scripts/eval_product_label.py --all [--json report.json] [--ai-cache DIR] [--no-ai]

Reports per photo and overall:
    allergen precision / recall (CONTAINS list only)
    sugar and energy per 100 within ±5 % of the ground truth
    mean / p95 latency, AI requests per photo

NOTE: only OCR text (never the photo) goes to the AI, and only for values the parser missed. --ai-cache keeps
AI answers in a local folder so repeated runs make no new requests.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EVAL_DIR = BACKEND_ROOT / "data" / "eval" / "product_labels"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TOLERANCE = 0.05
ALIASES = {"wheat": "wheat_gluten", "gluten": "wheat_gluten", "nuts": "tree_nuts", "tree nuts": "tree_nuts",
           "soya": "soy", "soybean": "soy", "groundnut": "peanut", "peanuts": "peanut", "eggs": "egg",
           "shellfish": "crustaceans", "sulphite": "sulphites", "sulfites": "sulphites"}


def norm_allergen(name: str) -> str:
    n = str(name).strip().lower()
    return ALIASES.get(n, n.replace(" ", "_"))


def within(pred: Optional[float], truth: Optional[float]) -> Optional[bool]:
    if truth is None:
        return None
    if pred is None:
        return False
    if truth == 0:
        return abs(pred) <= 0.5
    return abs(pred - truth) / abs(truth) <= TOLERANCE


def kcal_of(n: Dict[str, Any]) -> Optional[float]:
    if n.get("energy_kcal") is not None:
        return float(n["energy_kcal"])
    if n.get("energy_kj") is not None:
        return round(float(n["energy_kj"]) / 4.184, 1)
    return None


SCORED = ["energy_kcal", "sugars_g", "fat_g", "saturated_fat_g", "carbohydrate_g", "protein_g", "salt_g"]


def load_entries(eval_dir: Path, verified_only: bool = True):
    """-> (entries {id: {"images": {kind: path}, "allergens", "nutrients_per_100", "barcode", ...}}, skipped)

    Format 2 (scripts/build_label_eval_set.py): {"version": 2, "entries": {...}}; only verified=true count.
    Format 1 (legacy): {"file.jpg": {"allergens", "sugar_g_per_100", "energy_kcal_per_100", "barcode"}}.
    """
    data = json.loads((eval_dir / "ground_truth.json").read_text())
    if data.get("version") == 2:
        entries, skipped = {}, []
        for key, e in data["entries"].items():
            if verified_only and not e.get("verified"):
                skipped.append(key)
            else:
                entries[key] = e
        return entries, skipped
    entries = {}
    for name, gt in data.items():
        entries[name] = {"images": {"label": name}, "allergens": gt.get("allergens", []), "barcode": gt.get("barcode"),
                         "nutrients_per_100": {k: v for k, v in (("sugars_g", gt.get("sugar_g_per_100")),
                                                                 ("energy_kcal", gt.get("energy_kcal_per_100")))
                                               if v is not None}, "source": "legacy"}
    return entries, []


def truth_value(n: Dict[str, Any], key: str) -> Optional[float]:
    if n.get(key) is not None:
        return float(n[key])
    if key == "energy_kcal" and n.get("energy_kj") is not None:
        return round(float(n["energy_kj"]) / 4.184, 1)
    if key == "salt_g" and n.get("sodium_mg") is not None:
        return round(float(n["sodium_mg"]) * 2.5 / 1000, 3)
    return None


def pred_value(n: Dict[str, Any], key: str) -> Optional[float]:
    if key == "energy_kcal":
        return kcal_of(n)
    return truth_value(n, key)


def evaluate(run: Callable[[Any], Dict[str, Any]], eval_dir: Path, verified_only: bool = True,
             limit: Optional[int] = None, only: Optional[str] = None) -> Dict[str, Any]:
    from PIL import Image, ImageOps

    entries, unverified = load_entries(eval_dir, verified_only)
    if only:
        entries = {k: v for k, v in entries.items() if k.startswith(only)}
    if limit is not None:
        entries = dict(list(entries.items())[:max(0, limit)])
    rows: List[Dict[str, Any]] = []
    tp = fp = fn = 0
    counts = {"contains": [0, 0, 0], "may_contain": [0, 0, 0]}  # tp, fp, fn per list
    latencies: List[float] = []
    missing_files = []
    for key, gt in entries.items():
        paths = {k: eval_dir / p for k, p in (gt.get("images") or {}).items()}
        if not paths or not all(p.exists() for p in paths.values()):
            missing_files.append(key)
            continue
        results = {}
        ms = 0.0
        for kind, path in paths.items():  # one interpret call per photo (nutrition / ingredients / label)
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            started = time.perf_counter()
            results[kind] = run(img)
            ms += (time.perf_counter() - started) * 1000
        latencies.append(ms)
        nut_res = results.get("nutrition") or results.get("label") or {}
        ing_res = results.get("ingredients") or results.get("label") or {}

        # Ground truth allergens: a list (= CONTAINS) or {"contains": [...]|null, "may_contain": [...]|null};
        # null = not filled in -> not scored.
        gt_al = gt.get("allergens")
        truth_lists = ({"contains": gt_al} if isinstance(gt_al, list)
                       else {k: v for k, v in (gt_al or {}).items() if k in counts and isinstance(v, list)})
        allergen_detail = {}
        for kind, truth in truth_lists.items():
            p_set = {a["allergen"] for a in (ing_res.get("allergens") or {}).get(kind, [])}
            w_set = {norm_allergen(a) for a in truth}
            k_tp, k_fp, k_fn = len(p_set & w_set), len(p_set - w_set), len(w_set - p_set)
            for i, v in enumerate((k_tp, k_fp, k_fn)):
                counts[kind][i] += v
            allergen_detail[kind] = {"pred": sorted(p_set), "truth": sorted(w_set), "tp": k_tp, "fp": k_fp, "fn": k_fn}
        allergens_scored = "contains" in allergen_detail
        pred = set(allergen_detail.get("contains", {}).get("pred", []))
        want = set(allergen_detail.get("contains", {}).get("truth", []))
        a_tp = a_fp = a_fn = None
        if allergens_scored:
            a_tp, a_fp, a_fn = (allergen_detail["contains"][k] for k in ("tp", "fp", "fn"))
            tp, fp, fn = tp + a_tp, fp + a_fp, fn + a_fn
        no_wait_ms = sum((r.get("timings") or {}).get("time_to_results_ms", 0) for r in results.values())
        guessed = sorted({k for k, fl in (nut_res.get("nutrition_flags") or {}).items()
                          if any("assumed" in f for f in fl)})

        n_pred = nut_res.get("nutrition_per_100") or {}
        n_true = gt.get("nutrients_per_100") or {}
        fields = {}
        for k in SCORED:
            t = truth_value(n_true, k)
            if t is not None:
                p = pred_value(n_pred, k)
                fields[k] = {"pred": p, "truth": t, "ok": within(p, t)}

        rows.append({
            "image": key, "source_gt": gt.get("source"), "status": nut_res.get("status"),
            "error": next((n for r in results.values() for n in r.get("notes") or []
                           if "fail" in n.lower() or "busy" in n.lower() or "unreadable" in n.lower()), None),
            "allergens_pred": sorted(pred) if allergens_scored else None,
            "allergens_truth": sorted(want) if allergens_scored else None,
            "allergen_tp": a_tp, "allergen_fp": a_fp, "allergen_fn": a_fn, "allergens": allergen_detail,
            "latency_no_wait_ms": no_wait_ms, "first_column_guess": guessed,
            "fields": fields,
            "sugar_pred": fields.get("sugars_g", {}).get("pred"), "sugar_truth": fields.get("sugars_g", {}).get("truth"),
            "sugar_ok": fields.get("sugars_g", {}).get("ok"),
            "energy_pred": fields.get("energy_kcal", {}).get("pred"),
            "energy_truth": fields.get("energy_kcal", {}).get("truth"),
            "energy_ok": fields.get("energy_kcal", {}).get("ok"),
            "nutriscore": (nut_res.get("nutriscore") or {}).get("grade"),
            "sanity_flags": len(nut_res.get("sanity_notes") or []),
            "ai_values": sorted(k for r in results.values() for k, s in (r.get("field_sources") or {}).items() if s == "ai_from_text"),
            "latency_ms": round(ms), "photos": len(paths),
            "ai_values_rejected": sum(len(r.get("rejected_ai_values") or []) for r in results.values()),
            "notes": [n for r in results.values() for n in r.get("notes") or []][:6],
            "transcript_head": [line for r in results.values() for line in (r.get("transcript") or [])][:40],
        })

    def rate(key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"correct": sum(1 for v in vals if v), "of": len(vals),
                "rate": round(sum(1 for v in vals if v) / len(vals), 3) if vals else None}

    per_field = {}
    for k in SCORED:
        vals = [r["fields"][k]["ok"] for r in rows if k in r["fields"]]
        if vals:
            per_field[k] = {"correct": sum(vals), "of": len(vals), "rate": round(sum(vals) / len(vals), 3)}

    lat = sorted(latencies)
    return {
        "images": len(rows), "unverified_skipped": unverified, "unlabelled_images": [], "missing_files": missing_files,
        "allergens_by_list": {k: {"precision": round(c[0] / (c[0] + c[1]), 3) if c[0] + c[1] else None,
                                  "recall": round(c[0] / (c[0] + c[2]), 3) if c[0] + c[2] else None,
                                  "tp": c[0], "fp": c[1], "fn": c[2]} for k, c in counts.items()},
        "latency_no_wait_ms_mean": round(statistics.mean([r["latency_no_wait_ms"] for r in rows])) if rows else None,
        "allergen_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "allergen_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "sugar_within_5pct": rate("sugar_ok"), "energy_within_5pct": rate("energy_ok"),
        "fields_within_5pct": per_field,
        "latency_ms_mean": round(statistics.mean(lat)) if lat else None,
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]) if lat else None,
        "per_image": rows,
    }


DEFAULT_LIMIT = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"only the first N entries (default {DEFAULT_LIMIT})")
    ap.add_argument("--all", action="store_true", help="evaluate every entry (no --limit)")
    ap.add_argument("--only", help="evaluate only entries whose id starts with this, e.g. my_photos/")
    ap.add_argument("--include-unverified", action="store_true", help="also unverified entries (NOT for reporting)")
    ap.add_argument("--ai-cache", type=Path, help="keep AI answers here so repeated runs make no new AI requests")
    ap.add_argument("--no-ai", action="store_true", help="parser only, no AI at all")
    args = ap.parse_args()
    if not (args.dir / "ground_truth.json").exists():
        print(f"No ground_truth.json in {args.dir}. See data/eval/product_labels/README.md.")
        return 1
    from model.common.llm import LLM
    from model.product_label.pipeline import ProductLabelPipeline
    from scripts.eval_common import EvalLLM
    llm = EvalLLM(args.ai_cache, LLM([]) if args.no_ai else None)
    pipeline = ProductLabelPipeline(llm=llm)
    limit = None if args.all else args.limit
    started = time.perf_counter()
    report = evaluate(lambda img: pipeline.run(img), args.dir, verified_only=not args.include_unverified,
                      limit=limit, only=args.only)
    report["wall_s"] = round(time.perf_counter() - started, 1)
    report["ai_requests"] = llm.requests
    report["ai_cached_answers"] = llm.cached
    report["ai_calls_per_photo"] = round((llm.requests + llm.cached) / max(1, report["images"]), 2)

    def fmt(v, digits=1):
        return "—" if v is None else f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    def mark(ok):
        return "—" if ok is None else ("✓" if ok else "✗")

    short = {"energy_kcal": "kcal", "sugars_g": "sugar", "fat_g": "fat", "saturated_fat_g": "satfat",
             "carbohydrate_g": "carbs", "protein_g": "prot", "salt_g": "salt"}
    for r in report["per_image"]:
        fl = "  ".join(f"{short[k]} {fmt(v['pred'])}/{fmt(v['truth'])}{mark(v['ok'])}" for k, v in r["fields"].items())
        print(f"{r['image'][:34]:34} {str(r['status']):10} {fl}  {r['latency_ms']} ms"
              + (f"  AI: {', '.join(r['ai_values'])}" if r["ai_values"] else ""))
    pf = report["fields_within_5pct"]
    c, n = sum(v["correct"] for v in pf.values()), sum(v["of"] for v in pf.values())
    report["values_within_5pct"], report["accuracy"] = f"{c}/{n}", round(c / n, 4) if n else None
    print("\nper field within ±5 %: " + ", ".join(f"{short[k]} {v['correct']}/{v['of']}" for k, v in pf.items()))
    print(f"all values: {c}/{n} ({c / max(1, n):.1%}) | latency mean {report['latency_ms_mean']} ms, "
          f"p95 {report['latency_ms_p95']} ms | AI requests {llm.requests} (+{llm.cached} cached) for "
          f"{report['images']} photos | wall {report['wall_s']} s")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("full report:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
