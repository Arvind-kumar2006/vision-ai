"""Download the RapidOCR (PaddleOCR) ONNX models once into backend/weights/ocr/.

The app never downloads anything at request time: model/common/ocr.py only loads files from this folder.
Each file is checked against the SHA-256 listed in RapidOCR's own model catalogue (default_models.yaml).

    venv/bin/python scripts/setup_ocr_models.py            # all models below (~250 MB)
    venv/bin/python scripts/setup_ocr_models.py --check    # only verify what is there

Models (Apache-2.0):
    detection    PP-OCRv6 small/medium (multilingual), PP-OCRv5 mobile/server
    recognition  PP-OCRv6 small/medium (multilingual), PP-OCRv5 English mobile, PP-OCRv5 server (zh+en),
                 PP-OCRv5 Devanagari mobile (Hindi; no server size exists)
    orientation  PP-OCRv4 text-line classifier (0/180 degrees)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUT = BACKEND_ROOT / "weights" / "ocr"

# (catalogue version, task, catalogue key)
MODELS = [
    ("PP-OCRv6", "det", "multi_PP-OCRv6_det_small"),
    ("PP-OCRv6", "det", "multi_PP-OCRv6_det_medium"),
    ("PP-OCRv5", "det", "ch_PP-OCRv5_det_mobile"),
    ("PP-OCRv5", "det", "ch_PP-OCRv5_det_server"),
    ("PP-OCRv6", "rec", "multi_PP-OCRv6_rec_small"),
    ("PP-OCRv6", "rec", "multi_PP-OCRv6_rec_medium"),
    ("PP-OCRv5", "rec", "en_PP-OCRv5_rec_mobile"),
    ("PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server"),
    ("PP-OCRv5", "rec", "devanagari_PP-OCRv5_rec_mobile"),
    ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile"),
]


def catalogue():
    import rapidocr
    return yaml.safe_load((Path(rapidocr.__file__).parent / "default_models.yaml").read_text())["onnxruntime"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    args = ap.parse_args()
    cat = catalogue()
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for version, task, key in MODELS:
        entry = cat[version][task][key]
        url, want = entry["model_dir"], entry["SHA256"]
        dest = OUT / url.rsplit("/", 1)[-1]
        if dest.exists() and sha256(dest) == want:
            print(f"ok        {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        if args.check:
            print(f"MISSING   {dest.name}")
            bad += 1
            continue
        print(f"download  {dest.name} ...", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != want:
            tmp.unlink()
            print(f"FAILED    {dest.name}: checksum mismatch")
            bad += 1
            continue
        tmp.rename(dest)
        print(f"ok        {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    total = sum(p.stat().st_size for p in OUT.glob("*.onnx"))
    print(f"{OUT}: {total / 1e6:.0f} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
