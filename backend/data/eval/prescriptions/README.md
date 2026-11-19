# Prescription evaluation set

Put labelled prescription photos here (`.jpg`, `.jpeg` or `.png`) plus a `ground_truth.json`:

```json
{
  "p5.jpg": {
    "image": "/absolute/path/to/p5.jpg",
    "split": "tuned",
    "filled": true,
    "medicines": [
      {"aliases": ["dothepin", "prothiaden", "dosulepin"], "strength": "75mg"},
      {"aliases": ["pantop", "pantoprazole"], "strength": "40mg"}
    ],
    "already_taking": ["ecosprin av"]
  }
}
```

- `image` (optional) — defaults to the photo next to `ground_truth.json`.
- `split` — `tuned` (rules were developed on it) or `unseen` (default, held out).
- `filled` — `false` means the ground truth isn't written yet; the row is skipped.
- `aliases` — the brand as written, other brands of it, and the ingredient (generic) name.
  Other keys such as dose, timing or duration are ignored: the feature **identifies medicines only**.

List every medicine **as written on the prescription** (brand + strength is enough). Then run, from
`backend/`:

```bash
venv/bin/python scripts/eval_prescription.py                    # all prescriptions
venv/bin/python scripts/eval_prescription.py --only p5.jpg      # one photo
venv/bin/python scripts/eval_prescription.py --json report.json
venv/bin/python scripts/eval_prescription.py --no-ai            # OCR + dataset only
venv/bin/python scripts/eval_prescription.py --ai-cache DIR     # reuse AI answers between runs
```

Per prescription the script reports:

| Metric | Meaning |
|---|---|
| found | a listed medicine whose name matches an alias (one-to-one, fuzzy ≥ 0.8) |
| identified | found **and** Matched / Known brand to a product whose name or ingredients contain an alias |
| to confirm | found **and** Verify, with an alias among the candidates (the person can pick it) |
| **WRONG** | Matched / Known brand to a product containing **no** alias — the dangerous outcome |
| junk rows | listed medicines that match nothing in the truth |
| strength | written strength read correctly |

It also reports precision/recall/F1, match-status counts and latency. `--ai-cache` is for repeat
evaluation runs only; the app itself never caches AI answers.

The pipeline reads the photo **on this machine** (OpenCV + RapidOCR), strips patient/doctor lines, and
sends only the medicine rows as text to the configured AI provider (OpenRouter free model, Groq backup).
No image and no patient details are sent anywhere. See `scripts/eval_prescription.py` and ADR-005 in
`project_review/` for the full flow.

Images here contain patient data: the photos and `ground_truth.json` are git-ignored.
