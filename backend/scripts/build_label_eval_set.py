"""Build a REAL evaluation set for the Product Label Interpreter (no synthetic images).

Ground truth NEVER comes from this project's own model output. Sources:

  ndl/        openfoodfacts/nutrient-detection-layout (Hugging Face, CC BY-SA 3.0).
              Nutrition-table photos whose nutrient values were tagged by professional
              annotators (BIO token tags on Google Cloud Vision OCR tokens). Rows with
              meta.checked = true only; only *_100G tags are converted (per-serving values
              are not converted). Indian barcodes (890...) first, then the held-out test split.
              -> "source": "nutrient-detection-layout (human-annotated)", "verified": true
  off_india/  photos downloaded earlier from Open Food Facts (volunteer data, UNVERIFIED). The download code
              was removed with the Open Food Facts integration (ADR-005); existing entries stay.
  my_photos/  your own photos; empty ground truth for you to fill in ("verified": false).

Only entries with "verified": true are used by scripts/eval_product_label.py.

Run from backend/:
    venv/bin/python scripts/build_label_eval_set.py build [--ndl 20]
    venv/bin/python scripts/build_label_eval_set.py skeleton      # add my_photos/* entries
    venv/bin/python scripts/build_label_eval_set.py review        # review page on 127.0.0.1:8765

Network: Hugging Face datasets-server only (rows API; the 635 MB parquet files are NOT downloaded),
with pauses between requests. Images are stored at max 2000 px (JPEG q90) to save disk.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

EVAL_DIR = BACKEND_ROOT / "data" / "eval" / "product_labels"
GT_FILE = "ground_truth.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 2000

HF_DATASET = "openfoodfacts/nutrient-detection-layout"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_FILTER = "https://datasets-server.huggingface.co/filter"

SRC_NDL = "nutrient-detection-layout (human-annotated)"
SRC_OFF = "open_food_facts (volunteer, UNVERIFIED)"
SRC_OWN = "own photo"
LIC_NDL = "CC BY-SA 3.0 (dataset openfoodfacts/nutrient-detection-layout; images from Open Food Facts)"
LIC_OFF = "CC BY-SA 3.0 (Open Food Facts images); ODbL (Open Food Facts data)"

# nutrient-detection-layout tag -> our per-100 key and the unit our key is in
NDL_TAGS = {
    "ENERGY_KCAL_100G": ("energy_kcal", "kcal"), "ENERGY_KJ_100G": ("energy_kj", "kj"),
    "FAT_100G": ("fat_g", "g"), "SATURATED_FAT_100G": ("saturated_fat_g", "g"),
    "TRANS_FAT_100G": ("trans_fat_g", "g"), "CARBOHYDRATES_100G": ("carbohydrate_g", "g"),
    "SUGARS_100G": ("sugars_g", "g"), "ADDED_SUGARS_100G": ("added_sugars_g", "g"),
    "FIBER_100G": ("fibre_g", "g"), "PROTEINS_100G": ("protein_g", "g"),
    "SALT_100G": ("salt_g", "g"), "SODIUM_100G": ("sodium_mg", "mg"),
}
NUTRIENT_KEYS = ["energy_kcal", "energy_kj", "fat_g", "saturated_fat_g", "carbohydrate_g", "sugars_g",
                 "fibre_g", "protein_g", "salt_g", "sodium_mg"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def user_agent() -> str:
    from app.config import settings
    return settings.off_user_agent


def http_get(url: str, timeout: float = 60.0, ua: Optional[str] = None, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": ua or user_agent()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # temporary HF / OFF errors happen; back off and retry
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url.split('?')[0]} ({last})")


def save_image(data: bytes, path: Path) -> Tuple[int, int]:
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=90)
    return img.size


def load_gt(eval_dir: Path) -> Dict[str, Any]:
    p = eval_dir / GT_FILE
    if p.exists():
        data = json.loads(p.read_text())
        if data.get("version") == 2:
            return data
        raise SystemExit(f"{p} is in the old format; move it aside first (nothing was changed).")
    return {"version": 2, "note": "Only entries with verified=true are evaluated. Ground truth never comes "
                                  "from this project's model output.", "entries": {}}


def save_gt(eval_dir: Path, gt: Dict[str, Any]) -> None:
    p = eval_dir / GT_FILE
    if p.exists():
        shutil.copy2(p, p.with_suffix(".json.bak"))
    p.write_text(json.dumps(gt, indent=2, ensure_ascii=False))


def empty_entry(source: str, licence: str, images: Dict[str, str], verified: bool) -> Dict[str, Any]:
    return {"source": source, "licence": licence, "verified": verified, "images": images, "barcode": None,
            "product_name": None, "allergens": None, "nutrients_per_100": {}, "origin": {}, "notes": []}


# ---------------------------------------------------------------------------
# 1. nutrient-detection-layout (Hugging Face)
# ---------------------------------------------------------------------------
# A number must be whole (not part of a longer number) and its unit must end the word, so OCR
# noise like "10gl" is rejected instead of being read as 1.0 (never guess ground truth).
NUM = re.compile(r"(?<![\d.,])([<>≤≥]|less than)?\s*(\d+(?:[.,]\d+)?)(?![\d.,]\d)\s*(kcal|kj|mg|µg|mcg|g)?(?![a-z0-9])", re.I)


def parse_value(text: str, unit_expected: str) -> Tuple[Optional[float], Optional[str]]:
    """Annotated span text -> value in our unit, or (None, reason)."""
    nums = NUM.findall(text.replace(" ", " "))
    if len(nums) > 1 and unit_expected in ("kcal", "kj") and all(n[2] for n in nums):
        # only when EVERY number has its own unit ("2 345 kJ" must stay ambiguous)
        # "294kJ/70kcal" tagged as one span: keep the single number whose unit is the tagged one
        same = [n for n in nums if (n[2] or "").lower() == unit_expected]
        nums = same if len(same) == 1 else nums
    if len(nums) != 1:
        return None, f"{len(nums)} numbers in '{text.strip()}'"
    cmp_, num, unit = nums[0]
    if cmp_:
        return None, f"'{text.strip()}' is a bound, not a value"
    v = float(num.replace(",", "."))
    unit = (unit or "").lower()
    if unit_expected in ("kcal", "kj"):
        if unit and unit != unit_expected:
            return None, f"unit {unit} in '{text.strip()}', expected {unit_expected}"
        return v, None
    if unit_expected == "g":
        if unit in ("", "g"):
            return v, None
        if unit == "mg":
            return round(v / 1000, 4), None
        return None, f"unit {unit} in '{text.strip()}'"
    if unit_expected == "mg":  # sodium
        if unit == "mg":
            return v, None
        if unit == "g":
            return round(v * 1000, 3), None
        return None, f"no unit for sodium in '{text.strip()}'"
    return None, "unknown unit"


def ndl_spans(row: Dict[str, Any], names: List[str]) -> List[Tuple[str, str]]:
    """BIO tags -> [(TAG, text)]."""
    spans, cur_tag, cur = [], None, []
    for tok, tid in zip(row["tokens"], row["ner_tags"]):
        name = names[tid]
        if name.startswith("B-") or name == "O" or (name.startswith("I-") and name[2:] != cur_tag):
            if cur_tag:
                spans.append((cur_tag, "".join(cur)))
            cur_tag, cur = (name[2:], [tok]) if name != "O" else (None, [])
        else:
            cur.append(tok)
    if cur_tag:
        spans.append((cur_tag, "".join(cur)))
    return spans


def ndl_values(row: Dict[str, Any], names: List[str]) -> Tuple[Dict[str, float], List[str]]:
    values: Dict[str, float] = {}
    notes: List[str] = []
    seen: Dict[str, List[float]] = {}
    for tag, text in ndl_spans(row, names):
        if tag not in NDL_TAGS:
            continue
        key, unit = NDL_TAGS[tag]
        v, why = parse_value(text, unit)
        if v is None:
            notes.append(f"{key} not converted: {why}")
            continue
        seen.setdefault(key, []).append(v)
    for key, vals in seen.items():
        if len(set(vals)) == 1:
            values[key] = vals[0]
        else:
            notes.append(f"{key} not converted: annotated twice with different values {vals}")
    return values, notes


def hf_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = urllib.parse.urlencode({"dataset": HF_DATASET, "config": "default", **params})
    for attempt in range(6):  # the datasets-server sometimes answers "index is loading" / auth hiccups
        try:
            data = json.loads(http_get(f"{url}?{q}", ua="VisionAI-eval-builder/1.0"))
        except RuntimeError as exc:
            data = {"error": str(exc)}
        if "error" not in data:
            return data
        time.sleep(15)
    raise RuntimeError(f"Hugging Face datasets-server: {data.get('error')}")


def build_ndl(eval_dir: Path, gt: Dict[str, Any], n: int) -> List[str]:
    from model.product_label.sanity import check as sanity_check
    have = sum(1 for e in gt["entries"].values() if e["source"] == SRC_NDL)
    if have >= n:
        print(f"ndl: already {have} entries")
        return []
    names = None
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    # Indian barcodes first (both splits), then the held-out test split, then train.
    for split in ("train", "test"):
        try:
            d = hf_get(HF_FILTER, {"split": split, "where": "\"meta\".\"barcode\" LIKE '890%'", "length": 100})
            names = names or d["features"][0]["type"]["feature"]["names"]
            candidates += [(split, r) for r in d["rows"]]
        except RuntimeError as exc:
            print("ndl: Indian filter failed:", exc)
    print(f"ndl: {len(candidates)} rows with Indian barcodes")
    for split in ("test", "train"):
        offset = 0
        while len(candidates) < 4 * n:
            d = hf_get(HF_ROWS, {"split": split, "offset": offset, "length": 100})
            names = names or d["features"][0]["type"]["feature"]["names"]
            if not d["rows"]:
                break
            candidates += [(split, r) for r in d["rows"]]
            offset += 100
            time.sleep(1)
        if len(candidates) >= 4 * n:
            break

    added: List[str] = []
    for split, r in candidates:
        if have + len(added) >= n:
            break
        row, meta = r["row"], r["row"]["meta"]
        key = f"ndl/{meta['barcode']}_{meta['image_id']}.jpg"
        if key in gt["entries"] or not meta.get("checked") or meta.get("no_nutrition_table"):
            continue
        values, notes = ndl_values(row, names)
        if "sugars_g" not in values or not ({"energy_kcal", "energy_kj"} & set(values)):
            continue  # need at least sugar and energy per 100 g to be useful
        # Auto-verify only clean conversions that also pass our deterministic sanity check
        # (the annotations sit on OCR tokens, so OCR errors can hide in them).
        sane = sanity_check(values, "label")
        auto_ok = not notes and sane["ok"]
        if not sane["ok"]:
            notes.append("Sanity check failed: " + "; ".join(f["detail"] for f in sane["failures"])
                         + " — check against the photo.")
        size = save_image(http_get(row["image"]["src"], ua="VisionAI-eval-builder/1.0"), eval_dir / key)
        e = empty_entry(SRC_NDL, LIC_NDL, {"label": key}, verified=auto_ok)
        e.update(barcode=meta["barcode"], nutrients_per_100=values, notes=notes + [
            "Nutrition-table photo only: allergens are not annotated (not scored)."])
        e["origin"] = {"dataset": HF_DATASET, "split": split, "row_idx": r["row_idx"], "image_url": meta["image_url"],
                       "label_studio_id": meta.get("label_studio_id"), "checked": meta.get("checked"),
                       "stored_size": list(size)}
        e["verified_by"] = ("dataset annotators (meta.checked = true) + clean conversion + sanity check; "
                            "untick in the review page if you disagree") if auto_ok else None
        gt["entries"][key] = e
        added.append(key)
        print(f"ndl: + {key} {values}")
        time.sleep(1)
    return added


# ---------------------------------------------------------------------------
# 3. own photos
# ---------------------------------------------------------------------------
def skeleton(eval_dir: Path, gt: Dict[str, Any]) -> List[str]:
    own = eval_dir / "my_photos"
    own.mkdir(parents=True, exist_ok=True)
    added = []
    for p in sorted(own.iterdir()):
        key = f"my_photos/{p.name}"
        if p.suffix.lower() in IMAGE_EXTS and key not in gt["entries"]:
            e = empty_entry(SRC_OWN, "your own photo", {"label": key}, verified=False)
            e["nutrients_per_100"] = {k: None for k in NUTRIENT_KEYS}
            e["allergens"] = {"contains": None, "may_contain": None}
            e["notes"] = ["Fill in by hand from the printed label (per 100 g / 100 mL), then set verified to true. "
                          "allergens.contains / allergens.may_contain: lists of allergen names, [] if none, "
                          "null to skip. Leave a nutrient null if it is not printed."]
            gt["entries"][key] = e
            added.append(key)
    return added


# ---------------------------------------------------------------------------
# 4. credits
# ---------------------------------------------------------------------------
def write_credits(eval_dir: Path, gt: Dict[str, Any]) -> None:
    rows = []
    for key, e in gt["entries"].items():
        for kind, path in e["images"].items():
            o = e.get("origin") or {}
            url = o.get("image_url") or (o.get("image_urls") or {}).get(kind) or ""
            rows.append({"image": path, "entry": key, "kind": kind, "source": e["source"], "licence": e["licence"],
                         "origin_url": url, "product_url": o.get("product_url") or
                         (f"https://world.openfoodfacts.org/product/{e['barcode']}" if e.get("barcode") else "")})
    with open(eval_dir / "CREDITS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
        w.writeheader()
        w.writerows(rows)
    lines = ["# Credits — product label evaluation images", "",
             "- `ndl/`: images and annotations from the Hugging Face dataset "
             "[openfoodfacts/nutrient-detection-layout](https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout), "
             "licence CC BY-SA 3.0. Images originally from Open Food Facts contributors.",
             "- `off_india/`: photos from [Open Food Facts](https://world.openfoodfacts.org) contributors, CC BY-SA 3.0; "
             "nutrition values and allergen tags from the Open Food Facts database, ODbL 1.0.",
             "- `my_photos/`: the project owner's own photos.", "",
             "Images were resized to at most 2000 px. Per-image sources: `CREDITS.csv`.", ""]
    (eval_dir / "CREDITS.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 5. review page
# ---------------------------------------------------------------------------
def review_html(gt: Dict[str, Any]) -> str:
    cols = NUTRIENT_KEYS
    rows = []
    for key, e in gt["entries"].items():
        imgs = "".join(f'<a href="{html.escape(p)}" target="_blank"><img src="{html.escape(p)}" alt="{html.escape(k)} photo" '
                       f'loading="lazy"></a><div class="cap">{html.escape(k)}</div>' for k, p in e["images"].items())
        vals = e.get("nutrients_per_100") or {}
        inputs = "".join(
            f'<label>{c}<input data-k="{c}" value="{"" if vals.get(c) is None else vals[c]}" inputmode="decimal"></label>'
            for c in cols)
        al_raw = e.get("allergens")
        al_lists = al_raw if isinstance(al_raw, dict) else {"contains": al_raw}  # list = CONTAINS (OFF/legacy)

        def al_value(v):  # None -> "" (skip), [] -> "none", list -> "milk, soy"
            return "" if v is None else (", ".join(v) or "none")
        status = "verified" if e.get("verified") else "UNVERIFIED"
        rows.append(f"""<tr data-id="{html.escape(key)}" class="{status.lower()}">
<td class="imgs">{imgs}</td>
<td><div class="meta"><b>{html.escape(key)}</b><br>{html.escape(e.get('product_name') or '')}<br>
barcode {html.escape(e.get('barcode') or '—')}<br><span class="src">{html.escape(e['source'])}</span>
<br><span class="state">{status}</span></div>
<ul>{''.join(f'<li>{html.escape(n)}</li>' for n in e.get('notes') or [])}</ul></td>
<td class="vals">{inputs}<label>allergens contains (blank = skip, "none" = none)
<input data-k="allergens.contains" value="{html.escape(al_value(al_lists.get('contains')))}"></label>
<label>allergens may contain (blank = skip, "none" = none)
<input data-k="allergens.may_contain" value="{html.escape(al_value(al_lists.get('may_contain')))}"></label></td>
<td><label class="ver"><input type="checkbox" data-k="verified" {"checked" if e.get("verified") else ""}> verified</label></td>
</tr>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Label eval review</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --mut:#666; --line:#ddd; --warn:#fff6e0; --ok:#eaf8ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121317; --fg:#e8e8ea; --mut:#9a9aa3; --line:#2c2d33; --warn:#2b2414; --ok:#13261a; }} }}
body {{ font: 14px system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }} td {{ border-top: 1px solid var(--line); padding: 10px; vertical-align: top; }}
tr.unverified {{ background: var(--warn); }} tr.verified {{ background: var(--ok); }}
.imgs img {{ max-width: 420px; max-height: 520px; display: block; }} .cap {{ color: var(--mut); font-size: 12px; margin: 2px 0 8px; }}
.vals label {{ display: grid; grid-template-columns: 130px 1fr; gap: 6px; margin: 3px 0; font-size: 12.5px; }}
.vals input {{ width: 100%; font: inherit; padding: 3px 6px; background: transparent; color: inherit; border: 1px solid var(--line); }}
.src, .meta ul, ul {{ color: var(--mut); font-size: 12px; }} .ver {{ white-space: nowrap; font-weight: 600; }}
header {{ position: sticky; top: 0; background: var(--bg); padding: 8px 0; display: flex; gap: 12px; align-items: center; z-index: 1; }}
button {{ font: inherit; padding: 6px 14px; }}
</style></head><body>
<header><h1 style="font-size:18px;margin:0">Label eval review</h1>
<button id="save">Save ground truth</button><span id="msg" role="status"></span></header>
<p>Check each value against the photo (per 100 g / 100 mL as printed). Fix wrong values, clear values that are not printed, then tick <b>verified</b>.
Only verified entries are evaluated. Ground truth must come from the label, never from this app's output.</p>
<table><tbody>{''.join(rows)}</tbody></table>
<script>
function collect() {{
  const out = {{}};
  for (const tr of document.querySelectorAll('tr[data-id]')) {{
    const e = {{ nutrients_per_100: {{}} }};
    for (const inp of tr.querySelectorAll('input[data-k]')) {{
      const k = inp.dataset.k, v = inp.value.trim();
      if (k === 'verified') e.verified = inp.checked;
      else if (k.startsWith('allergens.')) {{
        e.allergens = e.allergens || {{}};
        e.allergens[k.slice(10)] = v === '' ? null : v.toLowerCase() === 'none' ? [] : v.split(',').map(s => s.trim()).filter(Boolean);
      }}
      else e.nutrients_per_100[k] = v === '' ? null : Number(v.replace(',', '.'));
    }}
    out[tr.dataset.id] = e;
  }}
  return out;
}}
document.getElementById('save').onclick = async () => {{
  const msg = document.getElementById('msg'), body = JSON.stringify(collect());
  for (const [id, e] of Object.entries(JSON.parse(body))) {{
    if (Object.values(e.nutrients_per_100).some(v => v !== null && Number.isNaN(v))) {{ msg.textContent = 'Not a number in ' + id; return; }}
  }}
  try {{
    const r = await fetch('/save', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body }});
    msg.textContent = r.ok ? 'Saved ' + new Date().toLocaleTimeString() : 'Save failed: ' + r.status;
    if (r.ok) setTimeout(() => location.reload(), 600);
  }} catch {{
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([body], {{ type: 'application/json' }}));
    a.download = 'review_edits.json'; a.click();
    msg.textContent = 'No review server: downloaded review_edits.json (apply with: build_label_eval_set.py apply review_edits.json)';
  }}
}};
</script></body></html>"""


def apply_edits(eval_dir: Path, edits: Dict[str, Any]) -> int:
    gt = load_gt(eval_dir)
    changed = 0
    for key, ed in edits.items():
        e = gt["entries"].get(key)
        if e is None:
            continue
        vals = {k: v for k, v in (ed.get("nutrients_per_100") or {}).items() if k in NUTRIENT_KEYS}
        new_vals = {k: float(v) for k, v in vals.items() if v is not None}
        if e["source"] == SRC_OWN:  # keep the empty skeleton keys visible
            new_vals = {k: (float(vals[k]) if vals.get(k) is not None else None) for k in NUTRIENT_KEYS}
        before = (e.get("nutrients_per_100"), e.get("allergens"), e.get("verified"))
        allergens = ed.get("allergens")
        if isinstance(allergens, dict) and not isinstance(e.get("allergens"), dict):
            # OFF / legacy entries keep a plain CONTAINS list unless "may contain" was filled in
            allergens = allergens.get("contains") if allergens.get("may_contain") is None else allergens
        e["nutrients_per_100"], e["allergens"], e["verified"] = new_vals, allergens, bool(ed.get("verified"))
        if before != (e["nutrients_per_100"], e["allergens"], e["verified"]):
            changed += 1
            e["reviewed_at"] = time.strftime("%Y-%m-%d %H:%M")
            if e["source"] == SRC_OFF and e["verified"]:
                e["verified_by"] = "project owner (review page)"
    save_gt(eval_dir, gt)
    (eval_dir / "review.html").write_text(review_html(gt))
    return changed


def serve_review(eval_dir: Path, port: int) -> None:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(eval_dir), **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.path = "/review.html"
            return super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                changed = apply_edits(eval_dir, json.loads(self.rfile.read(n)))
            except Exception as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"changed": changed}).encode())

    gt = load_gt(eval_dir)
    (eval_dir / "review.html").write_text(review_html(gt))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # localhost only
    print(f"Review page: http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    srv.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=EVAL_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ndl", type=int, default=20)
    sub.add_parser("skeleton")
    r = sub.add_parser("review")
    r.add_argument("--port", type=int, default=8765)
    a = sub.add_parser("apply")
    a.add_argument("edits", type=Path)
    args = ap.parse_args()
    eval_dir = args.dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "review":
        serve_review(eval_dir, args.port)
        return 0
    if args.cmd == "apply":
        print("changed entries:", apply_edits(eval_dir, json.loads(args.edits.read_text())))
        return 0
    gt = load_gt(eval_dir)
    if args.cmd == "build":
        added = build_ndl(eval_dir, gt, args.ndl)
    else:
        added = []
    added += skeleton(eval_dir, gt)
    save_gt(eval_dir, gt)
    write_credits(eval_dir, gt)
    (eval_dir / "review.html").write_text(review_html(gt))
    counts: Dict[str, int] = {}
    for e in gt["entries"].values():
        counts[f"{e['source']} ({'verified' if e['verified'] else 'unverified'})"] = \
            counts.get(f"{e['source']} ({'verified' if e['verified'] else 'unverified'})", 0) + 1
    print(f"added {len(added)} entries; totals:", json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
