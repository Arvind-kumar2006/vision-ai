"""Compare RapidOCR model sets and preprocessing steps on our eval photos. No AI is used.

Labels: the 16 verified photos (data/eval/product_labels), OCR -> regex parser -> 7 values per 100 g within
±5 % of the ground truth (the same score as scripts/eval_product_label.py, without the AI gap filling).
Prescription: p5.jpg (or --rx PATH), share of the key strings found in the OCR text (medicine names,
strengths, timing, course, registration number).

    venv/bin/python scripts/compare_ocr.py --json report.json [--models v6-small,v5-server] [--rx ~/Downloads/p5.jpg]
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
sys.path.insert(0, str(BACKEND_ROOT))

from PIL import Image  # noqa: E402

from model.common.ocr import MODEL_SETS, OCREngine, PROFILE_LABEL, PROFILE_PRESCRIPTION  # noqa: E402
from model.common.preprocess import preprocess  # noqa: E402
from model.product_label.parser import parse_nutrition, to_per100  # noqa: E402
from scripts.eval_product_label import EVAL_DIR, SCORED, load_entries, pred_value, truth_value, within  # noqa: E402

STEP_SETS = {"exif": ("exif",), "exif+resize": ("exif", "resize"), "+perspective": ("exif", "resize", "perspective"),
             "+clahe": ("exif", "resize", "clahe"), "+perspective+clahe": ("exif", "resize", "perspective", "clahe")}
# What a reader must get from p5.jpg (printed + handwritten); fuzzy match per item.
P5_KEYS = ["dothepin", "prothiaden", "dosulepin", "clonazepam", "lonazep", "clonotril", "zocalm", "5mg", "75mg",
           "2mg", "sos", "8months", "ecosprin", "4967/2013", "रात"]


def found(key: str, text: str) -> bool:
    t = text.lower().replace(" ", "")
    if key in t:
        return True
    n = len(key)
    return any(SequenceMatcher(None, key, t[i:i + n]).ratio() >= 0.8 for i in range(0, max(1, len(t) - n + 1)))


def run(models, steps_name, entries, rx):
    eng = OCREngine(models, hindi=True)
    ok = total = 0
    times = []
    for key, gt in entries.items():
        path = EVAL_DIR / list(gt["images"].values())[0]
        t = time.perf_counter()
        bgr, _ = preprocess(Image.open(path), STEP_SETS[steps_name])
        r = eng.read(bgr, hindi=False)
        times.append((time.perf_counter() - t) * 1000)
        per = to_per100(parse_nutrition("\n".join(r["lines"])))["per100"]
        for k in SCORED:
            truth = truth_value(gt.get("nutrients_per_100") or {}, k)
            if truth is not None:
                total += 1
                ok += int(bool(within(pred_value(per, k), truth)))
    out = {"models": models, "steps": steps_name, "label_values": f"{ok}/{total}", "label_accuracy": round(ok / total, 4),
           "label_ms_mean": round(statistics.mean(times)), "label_ms_max": round(max(times))}
    if rx:
        t = time.perf_counter()
        bgr, _ = preprocess(Image.open(rx), STEP_SETS[steps_name])
        r = eng.read(bgr)
        text = "\n".join(r["lines"])
        hits = [k for k in P5_KEYS if found(k, text)]
        out.update(rx_keys=f"{len(hits)}/{len(P5_KEYS)}", rx_missing=[k for k in P5_KEYS if k not in hits],
                   rx_ms=round((time.perf_counter() - t) * 1000))
    return out


def profile_report(rx: Path, models: str, steps_name: str) -> Dict[str, Any]:
    """A/B the two OCR profiles on one prescription. No AI, no medicine DB.

    A = label profile: RapidOCR per-line 0/180 classifier ON, default text_score (0.5).
    B = prescription profile: per-line classifier OFF, text_score 0 (keep every non-empty reading).
    """
    bgr, _ = preprocess(Image.open(rx), STEP_SETS[steps_name])
    eng = OCREngine(models, hindi=True)
    report: Dict[str, Any] = {"image": str(rx), "models": models, "steps": steps_name, "profiles": {}}
    for profile in (PROFILE_LABEL, PROFILE_PRESCRIPTION):
        t = time.perf_counter()
        r = eng.read(bgr, hindi=True, profile=profile, preprocessing=steps_name)
        boxes = r.get("boxes") or []
        candidates = sum(len(b.get("candidates") or []) for b in boxes)
        report["profiles"][profile] = {
            "settings": {
                "use_cls": profile == PROFILE_LABEL,
                "text_score": "default (0.5)" if profile == PROFILE_LABEL else 0.0,
            },
            "detected_boxes": len(boxes),
            "retained_candidates": candidates,
            "rows": len(r.get("rows") or []),
            "rotated": r.get("rotated"),
            "latency_ms": int((time.perf_counter() - t) * 1000),
            "text": r.get("lines") or [],
            "boxes": [
                {k: b.get(k) for k in ("id", "text", "confidence", "box", "engine", "orientation",
                                       "preprocessing", "selected", "candidates")}
                for b in boxes
            ],
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_SETS))
    ap.add_argument("--steps", default=",".join(STEP_SETS))
    ap.add_argument("--rx", type=Path, default=Path.home() / "Downloads" / "p5.jpg")
    ap.add_argument("--rx-steps", default="exif+resize", choices=tuple(STEP_SETS),
                    help="preprocessing for --rx-profiles (default: exif+resize)")
    ap.add_argument("--rx-profiles", action="store_true",
                    help="compare label (A) vs prescription (B) OCR on --rx; no label eval, no AI")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--limit", type=int, help="only the first N label photos (slow model sets)")
    args = ap.parse_args()

    if args.rx_profiles:
        if not args.rx.exists():
            print(f"Prescription image not found: {args.rx}")
            return 1
        models = args.models.split(",")[0]
        report = profile_report(args.rx, models, args.rx_steps)
        print(f"image: {report['image']}  models: {models}  steps: {args.rx_steps}")
        for profile, p in report["profiles"].items():
            print(f"{profile:12} boxes {p['detected_boxes']:>3} candidates {p['retained_candidates']:>3} "
                  f"rows {p['rows']:>3} rotated {p['rotated']} {p['latency_ms']:>5} ms")
            for b in p["boxes"]:
                if b.get("text"):
                    print(f"  {b['confidence']:.3f} {b['text'][:70]}")
        if args.json:
            args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False))
            print("full report:", args.json)
        return 0

    entries, _ = load_entries(EVAL_DIR)
    if args.limit:
        entries = dict(list(entries.items())[:args.limit])
    rx = args.rx if args.rx.exists() else None
    rows = []
    for models in args.models.split(","):
        for steps in args.steps.split(","):
            r = run(models, steps, entries, rx)
            rows.append(r)
            print(f"{models:10} {steps:20} labels {r['label_values']:>7} ({r['label_accuracy']:.1%}) "
                  f"{r['label_ms_mean']:>5} ms mean / {r['label_ms_max']:>5} max | p5 {r.get('rx_keys', '-')} "
                  f"{r.get('rx_ms', '-')} ms missing {r.get('rx_missing')}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
