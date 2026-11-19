# 🖼️ Vision AI Backend: captioning, prescriptions, food labels, plants

FastAPI backend with four features:

- **Image captioning:** three pretrained models, chosen per request (`?model=`), each returning one attention map per word, which shows where the model looked:

  | `model` | Architecture | Trained on | Attention grid |
  |---|---|---|---|
  | `blip` (**configured default**) | ViT-B/16 + BERT-style text decoder | COCO (fine-tuned) | 24×24 |
  | `sat` | ResNet-101 + soft attention + LSTM ("Show, Attend and Tell", the PRD architecture) | COCO 2014 | 14×14 |
  | `catr` | ResNet-101 + 6-layer transformer | COCO 2017 | follows image aspect |

  On our test photos BLIP was the most accurate (for example "a woman in a suit leaning on a railing", where SAT and CATR both said "talking on a cell phone"). `configs.yaml` therefore sets `blip` as the default; SAT and CATR remain available with `?model=sat` and `?model=catr`. `deploy/deploy_ec2.sh` pre-caches BLIP so no visitor waits for the download. All three still say "a dog running with a ball in its mouth" for a dog with the ball at its paws.
- **Prescription reader:** photo → OpenCV preprocessing → RapidOCR (PaddleOCR models, English + Hindi) on this machine → patient/doctor lines removed → a free AI text model structures the medicine lines → checked against the Indian Medicine Dataset, with "What it's for" by ingredient.
- **Food label checker:** photo → the same preprocessing and OCR → regex parser (AI only for values the parser missed) → allergen rules and Nutri-Score 2023, plus a guarded explanation.
- **Plant identification:** 1–5 photos → PlantNet API → candidates plus a small local medicinal guide; medicinal uses are written by the text AI from the **name only** (never the photo), in English, Hindi or Kannada.

The prescription and label pipelines are described in `../project_review/ADR-005-simplified-pipelines.md` (a local, git-ignored design document).

The React app lives in `../frontend/`.

---

## 📁 Structure

```
backend/
├── run.py                        # starts the API + built web app on one port (uvicorn)
├── configs.yaml                  # captioning inference settings (default model, weights, beam size)
├── requirements.txt              # FastAPI, torch, transformers, openai, rapidfuzz, OpenCV (2-step install)
├── requirements-ocr.txt          # RapidOCR + onnxruntime, pinned; installed with --no-deps
├── Dockerfile                    # Linux image; OCR models downloaded at build time
├── .dockerignore                 # keeps venv, .env, weights, eval data out of the image
├── .env.example                  # up-to-date template: OPENROUTER_*, GROQ_*, PLANTNET_API_KEY
├── app/                          # FastAPI service layer
│   ├── main.py                   # app, CORS, routers; 4 services created once at startup
│   ├── config.py                 # env-overridable settings (.env)
│   ├── dependencies.py           # image upload validation (type, size, EXIF)
│   ├── schemas/                  # API response models: __init__ (health, caption), prescription.py,
│   │                             # product_label.py, plant.py
│   ├── routes/                   # health.py, caption.py, prescription.py, product_label.py, plant.py
│   └── services/                 # caption_service.py (+ captioners/), prescription_service.py,
│                                 # product_label_service.py, plant_service.py
├── model/
│   ├── captioning/               # SAT model + converter, CATR architecture; BLIP is loaded
│   │                             # via HuggingFace transformers in app/services/captioners/blip.py
│   ├── common/                   # preprocess.py, ocr.py (RapidOCR), llm.py (OpenRouter + Groq), redact.py
│   ├── prescription/             # pipeline, extract, layout, medicine_db, uses, schemas
│   ├── product_label/            # pipeline, parser, allergens, nutriscore, additives, sanity, explain, schemas
│   └── plant/                    # identify.py (PlantNet + local guide), uses.py (AI-written, cached)
├── weights/                      # model files (git-ignored): captioning weights + word map; ocr/ from
│                                 # setup_ocr_models.py; catr_tokenizer/
├── scripts/                      # setup_ocr_models.py, compare_ocr.py, eval_*.py, build_*.py,
│                                 # matcher_cases*.json
├── tests/                        # unittest; no network, OCR and AI are fakes
├── data/medicines/               # Indian Medicine Dataset CSV (MIT), known_brands.json,
│                                 # uses_by_ingredient.json (CC BY-SA 4.0), 250k_raw/, Images/
├── data/product_label/           # allergens.json, additives.json, nutriscore_thresholds.json
├── data/cache/                   # medicine_index.pkl (built from the CSV) + ai_eval_rx/ eval caches (git-ignored)
├── data/samples/                 # prescription_test.jpeg
└── data/eval/                    # eval photos and ground truth (git-ignored), eval_before.json / eval_after.json
```

### Caption request flow

```
POST /api/caption?model=sat|catr|blip (image)   # sat = configs.yaml default
  → dependencies.read_image          validate type/size, fix EXIF rotation, RGB
  → CaptionService.caption           picks the engine (default: configs.yaml inference.model)
      blip  BlipProcessor 384×384 → generate (beam 3, no repeated word pairs)
            → cross-attention of each word over 24×24 patches, minus the caption's average
              (removes ViT "attention sink" patches so each word shows its own region)
      sat   256×256, ImageNet norm → ResNet-101 → (1, 14, 14, 2048) → beam 5 with the gated
            soft-attention LSTM → 14×14 map per word
      catr  longest side 299 → ResNet-101 + transformer encoder once → beam 3 over the decoder
            → cross-attention (mean of heads and layers) on an H×W grid per word
  → { model, caption, words, alphas, grid_shape, grid_size, generation_time_ms, inference_ms }
```

`alphas` always contains **one map per word**, scaled 0–1, but the shape depends on the engine:
**14×14** for SAT, **24×24** for BLIP and **H×W** (image aspect ratio) for CATR. The exact shape is
returned as `grid_shape`; the older square-only `grid_size` field is kept for older clients.
Beam size defaults come from `configs.yaml`: SAT 5, CATR 3, BLIP 3 (overridable with `?beam_size=`).

---

## 🛠 Installation

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-ocr.txt   # RapidOCR + onnxruntime, pinned; uses our one OpenCV
python scripts/setup_ocr_models.py                         # OCR models -> weights/ocr/ (once, ~364 MB, SHA-256 checked)
cp .env.example .env                                       # then add OPENROUTER_API_KEY (and GROQ_API_KEY as backup)
```

The app never downloads OCR models while running; without them the prescription and label pages report
"not available" and point to `setup_ocr_models.py`. Docker: `docker build -t vision-ai-backend .` does all of this
(headless OpenCV on Linux).

Put the captioning weights in `weights/`. They are git-ignored. To recreate them:

1. Download `BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar` and
   `WORDMAP_coco_5_cap_per_img_5_min_word_freq.json` from the project's model assets.
2. Copy both into `weights/`.
3. Run:
   ```bash
   python -m model.captioning.convert_checkpoint   # writes weights/sat_resnet101_coco.pth
   ```

The converter loads the pickled checkpoint through a `models` → `sat_models` alias and drops the optimizer state. It then checks that the converted model gives bit-identical outputs. After that, the `.pth.tar` is no longer needed (this copy's is kept in `../archive/weights/`; copy it back into `weights/` only to re-run the converter).

---

## 🚀 Usage

```bash
cd backend
venv/bin/python run.py          # http://127.0.0.1:8000/docs
```

### Starting the backend and frontend

Start the servers from **your own terminal**. A server started by a tool or an assistant session stops when that session ends.

**Option A: one port (simplest).** `run.py` serves the API and the built web app together:

```bash
cd frontend && npm install && npm run build      # once, and again after any UI change
cd ../backend && venv/bin/python run.py          # open http://127.0.0.1:8000  (#/product-label for the label tab)
```

**Option B: while editing the UI (two terminals, hot reload):**

```bash
# terminal 1: API on :8000
cd backend && venv/bin/python run.py
# terminal 2: Vite dev server on :5173, forwards /api to :8000
cd frontend && npm run dev                        # open http://localhost:5173
```

Vite listens on `localhost`. If `http://127.0.0.1:5173` refuses the connection, use `http://localhost:5173`.

- **Stop:** press `Ctrl+C` in each terminal. If port 8000 or 5173 is still in use, find the process with `lsof -iTCP:8000 -sTCP:LISTEN` and stop it with `kill <PID>`.
- **Check:** `curl http://127.0.0.1:8000/api/health` should report `"product_label": "ready"`.
- **Keys:** copy `.env.example` to `backend/.env` and set `OPENROUTER_API_KEY` (free model), `GROQ_API_KEY` (backup) and `PLANTNET_API_KEY` (plant identification). Without the AI keys the prescription and label features still work: prescriptions show dataset-matched names with "AI busy: details not extracted", labels use the parser only. Without the PlantNet key the plant endpoints return 503.
- **Installing packages:** use `venv/bin/python -m pip install …`, not `venv/bin/pip`. The venv was moved, so its `pip` script points at an old interpreter path.
- **Keeping it running:** to keep the server running after you close the terminal, use `nohup venv/bin/python run.py > server.log 2>&1 &` (stop it with `kill`).

Run the server from the venv. Different Pillow/torch versions resize images slightly differently, and that can change some captions.

| Method | Endpoint | Body | Response |
|---|---|---|---|
| GET | `/api/health` | — | status, device, vocab_size, checkpoint, modes |
| POST | `/api/caption` | `file` (JPEG/PNG/WebP ≤ 10 MB); optional `model` (`sat` default, `catr`, `blip`), `beam_size` (1–10, default 5) and `max_len` | `{model, caption, inference_ms, words, alphas, grid_shape, grid_size, generation_time_ms}`. `alphas` has one map per word, scaled 0–1; shape is 14×14 (SAT), 24×24 (BLIP) or H×W (CATR), given by `grid_shape` (`grid_size` is kept for older clients) |
| POST | `/api/prescription` | `file` (image) | structured prescription JSON (`status`: ok / unreadable / ai_busy) |
| POST | `/api/product-label/interpret` | `image`; optional `product_name` (≤ 100 chars, titles the result and is given to the explanation AI, never used for values or the score) | food label result (`status`: ok / unreadable) |
| GET | `/api/product-label/{id}/explanation` | – | guarded explanation (AI) or the template |
| POST | `/api/plant/identify` | `images` (1–5 photos of one plant), `organ` (auto / leaf / flower / fruit / bark) | PlantNet candidates + medicinal-use context for plants in the local guide (`status`: ok / no_match). Needs `PLANTNET_API_KEY`; photos are sent to Pl@ntNet |
| POST | `/api/plant/uses` | `{scientific_name, common_names}` | medicinal uses (with evidence level), parts used, safety notes, `toxic` flag, written by the AI from the name only; doses are removed (`status`: ok / busy / unavailable) |

The React dev server proxies `/api` to the backend (`frontend/vite.config.js`).

Captioning settings live in `configs.yaml` → `inference:` (`weights`, `image_size`, `beam_size` (default 5), `no_repeat_ngram` (default 2), `max_caption_len`).
At startup the model runs one short warm-up pass (a second or two on Apple MPS). This keeps the first real request fast (about 0.1–0.7 s).

### Tests

```bash
cd backend
venv/bin/python -m unittest discover -s tests -v
```

The tests check:
- the encoder output is `(1, 14, 14, 2048)`
- attention weights add up to 1 at every decoding step
- 14×14 maps scaled 0–1, with special tokens removed
- the vocabulary is the COCO word map (9,490 entries)
- beam size and length limits
- the API response shape and error codes

The 7 test files are: `test_captioning.py`, `test_layout_rows.py`, `test_ocr_candidates.py`,
`test_prescription.py`, `test_prescription_uses.py`, `test_product_label.py` and `test_plant.py`.
They use fake OCR engines and fake AI clients, so no network or model weights are needed
(except `test_captioning.py`, which needs `weights/sat_resnet101_coco.pth`).

### Prescription reader (`model/prescription/`, ADR-005)

```
POST /api/prescription (image)
  1. preprocess   common/preprocess.py: EXIF rotation, resize (steps chosen by scripts/compare_ocr.py)
  2. OCR          common/ocr.py: RapidOCR detection + recognition; every box is also read by the Devanagari
                  model and the more confident reading is kept (printed Hindi, "रात"); sideways/upside-down
                  photos are turned upright. No text -> "Couldn't read this photo. Try a flatter, brighter photo."
  3. redact       common/redact.py: only lines from the first medicine marker (Rx / Tab / 1)) on, without
                  name, age, weight, phone, address or registration lines, ever leave this machine
  4. AI           extract.py: one text request (OpenRouter free model, Groq as backup) -> medicines (name as
                  written, alternatives, strength, form) + already taking. Identification only (2026-11-17):
                  how often, how long, dose counts, dates and follow-up are not extracted.
                  Both busy -> each word of the medicine lines is looked up in the dataset instead, and the
                  result says "AI busy: details not extracted. Try again in a few minutes."
  5. rules        names not found in the OCR text are flagged; a strength on a low-confidence line or not in
                  the text is marked "check". Schedule marks (1-0-1, BD, "x 5 days") are only used as evidence
                  that a row is a medicine line
  6. dataset      medicine_db.py + uses.py (below)
```

The feature **identifies medicines only**: chat, date parsing, dose counts and follow-up were removed in the
2026-11 simplification (the old `chat.py`, `dates.py` and line-cropping modules are gone). No image is sent
to the AI — only the redacted medicine rows, as text.

#### Medicine matching (local, step 6)

- **Data:**
  - Indian Medicine Dataset (`data/medicines/updated_indian_medicine_data.csv`, MIT licence as provided with the file; add the exact source URL here, 45 MB, 253,973 rows). Discontinued rows (7,905) and duplicate names are dropped: **241,867 medicines, 116,502 brands**.
  - **"What it's for" (by ingredient):** derived from a public medicines-usage dataset, licensed **CC BY-SA 4.0**. The raw CSV (89 MB) goes in `data/medicines/250k_raw/` and is git-ignored; `scripts/build_medicine_uses.py` joins it to the Indian Medicine Dataset by name and writes the committed lookup `data/medicines/uses_by_ingredient.json` (0.57 MB, 2,896 ingredient keys), which is **itself CC BY-SA 4.0** (ShareAlike: keep this licence if you redistribute it). Where it has no uses, the Indian Medicine Dataset's own description is the fallback. Names are normalised (dothiepin = dosulepin, acetaminophen = paracetamol, …); combinations without their own entry combine each ingredient's uses. No AI ever writes these lines; with no data the app says "No description available. Ask your pharmacist what this medicine is for."
  - The cleaned index is cached in `data/cache/medicine_index.pkl`. It's built in about 10 s the first time, loads in about 1 s afterwards, loads once at startup, and rebuilds automatically if the CSV changes.
- **Per extracted medicine:**
  1. **Company names never count as medicines.** Exact company names without digits are rejected first ("Cipla"; "Pan 40" is still a medicine).
  2. **Brand similarity (rapidfuzz) decides the status:**
     - **matched:** ≥ 90 and ≥ 5 points ahead of the best *different* brand
     - **verify:** 75–89, or a close different brand
     - **unmatched:** < 75
  3. **Within the winning brand, strength and form pick the variant** ("Dolo 650" → Dolo 650 Tablet). Strengths come from the name *and* the composition. A written strength with no matching product means **verify**; no written strength means no strength is shown.
  4. **Identification guards (2026-11-17):**
     - the brand word is chosen past an OCR-misread "Tab" ("Td SERTA" → Serta, not the brand "TD");
     - a misspelt brand is **matched** only if the written strength or a variant word ("PT", "Plus") confirms it, otherwise **verify**;
     - a written word the product doesn't have ("Colgate Plax") or a form it doesn't come in ("gum paint" vs tablets) means **verify**;
     - oral-care products and devices (Colgate, toothbrush, Sensodyne ...) are reported as not a medicine;
     - "Ativan (Lorazepam)" is matched as the brand Ativan with Lorazepam as an alternative name.
  5. **Confidence level and strength (2026-11-18):** strength written and in the product → good; written and in no product of the brand → Low; not written and the written variant comes in one strength only ("Hexigel", "Sizodon Plus") → neutral "not applicable"; not written and it comes in 2–3 strengths → Medium, 4 or more → Low. A strength inside the name ("Dolo 650") counts as written. High needs every factor good or neutral, no other brand within 5 points, and, when the strength can't confirm it, a name match of exactly 100. A gum paint counts as the same kind of product as a mouth gel, cream or ointment.
  6. **Measured with** `scripts/eval_matcher.py`: 115 cases used while writing the rules and 123 held-out cases generated afterwards (written name → expected ingredient, checked against the dataset). Tuning cases: wrong confident matches 4 → 0, correct or offered for confirmation 107 → 115 of 115. Held-out cases: wrong 0 → 0, correct or offered 96 → 121 of 123 (mostly misread "Tab" prefixes such as "Tub" or "Ta6"). `scripts/eval_prescription.py` scores whole photos the same way (identified / to confirm / wrong).
  7. **Not in the dataset → `data/medicines/known_brands.json`** → **known_brand** (e.g. Zincovit, Electral; never counted as matched).
  8. **The name as read is never overwritten:** `read_as` keeps it, `suggested_name` gives the database name, and a review note explains the difference.
- **Top-3 candidates** (distinct brands first) are returned for the UI's "verify" dropdown.
- **`db_info`** ("From medicine database") holds composition, manufacturer, pack, type, and description, side effects and interactions **only when the dataset has them** (about 2.8% of medicines). Empty fields are omitted. The LLM never adds medical facts.
- **Confidence** = reader confidence × match score/100 × status weight (matched 1.0, verify 0.8, known brand 0.7, unmatched 0.5), all 0–1. Names not found in the text that was read stay **capped at 0.30**.

> **Not medical advice:** handwriting OCR is error-prone. The UI shows a disclaimer, and every result must be checked by a doctor or pharmacist.

### Food label checker (`model/product_label/`, ADR-005)

```
POST /api/product-label/interpret   image
GET  /api/product-label/{session_id}/explanation   explanation, fetched after the results (never blocks them)

  1. preprocess + OCR   same modules as prescriptions (no Hindi pass); no text -> "Couldn't read this photo ..."
  2. parser             regex, per 100 g / 100 mL (per serving converted with the serving size); nutrient names
                        in English, French, German, Spanish, Italian, Dutch, Portuguese, Polish, Czech
  3. AI (only if needed) a nutrient whose name is printed but whose value wasn't parsed, or values that fail the
                        sanity check: the OCR TEXT (never the photo) goes to the AI, which may only copy numbers
                        that are in that text; anything else is rejected. AI busy -> parsed values + a note
  4. analysis           sanity checks, allergens (rules; "unknown" when the ingredients weren't read, never
                        "none"), additives, Nutri-Score 2023 (official tables), tips (rules)
  5. explanation AI from the parsed values only, guarded (numbers, recommendations, allergen claims);
                        template when no AI answers
```

- **Health score (Nutri-Score):** the updated algorithm from Santé publique France, *Conditions of Use*, version 26 June 2024, Exhibit 1-B. All thresholds are in `data/product_label/nutriscore_thresholds.json`, with the source recorded. Category: beverage when the table is per 100 mL, otherwise general food (approximate).
- **Sanity checks:** energy ≈ 4·protein + 4·carbs + 9·fat (±20 %); kJ ≈ kcal × 4.184 (±10 %); nothing above 100 g/100 g; sugars and fibre ≤ carbohydrate; saturated + trans fat ≤ fat; added sugars ≤ sugars. A failure adds "values may be misread".
- **Data files:** `allergens.json` (incl. Indian terms; sulphites per FSSAI Labelling Regulations 2020, clause 5(14)(viii)) and `additives.json` (Open Food Facts additives taxonomy, ODbL, pinned commit).
- **Evaluation:** `scripts/eval_product_label.py --all` scores the verified photos (7 values within ±5 %); `--ai-cache DIR` keeps AI answers between runs so repeated evals don't use the free allowance. `scripts/compare_ocr.py` compares OCR models and preprocessing steps (no AI). `scripts/build_label_eval_set.py` builds the set from the Hugging Face dataset and keeps your own photos separate.

### Plant identification (`model/plant/`)

```
POST /api/plant/identify   1-5 photos of one plant + organ (auto / leaf / flower / fruit / bark)
POST /api/plant/uses       {scientific_name, common_names, lang} -> medicinal uses

  identify.py   1-5 photos -> PlantNet API (PLANTNET_API_KEY) -> candidates + confidence;
                adds context from a small hand-written local guide, with Hindi/Kannada text
  uses.py       AI writes uses from the NAME ONLY (never the photo); JSON-validated, any dose
                is removed, cached in memory per scientific name. A second AI call translates
                the checked English answer to hi/kn and validates script and structure;
                if translation fails the English answer is returned
```

- Photos are sent to **Pl@ntNet** (France); the free tier allows 500 identifications/day.
- Without `PLANTNET_API_KEY` the endpoints return 503 and `/api/health` reports `"plant": "unavailable"`.
- This feature does not use the OCR models or the medicine database. See `tests/test_plant.py` (network mocked).

### AI (`model/common/llm.py`)
- **One free OpenRouter text model** (`OPENROUTER_MODEL`, must end in `:free`), **Groq as the single backup** (`GROQ_MODEL`). The official `openai` SDK talks to both. One try per provider, plus **one short wait-and-retry** only for an HTTP 429 whose `retry-after` is ≤ 10 s. No budget files and no caches in the app. Keys are never logged.
- **Text only:** no image is ever sent to an AI.
- **Configuration:** `.env.example` is the up-to-date template. Older variable names sometimes left in a local `.env` (`TROCR_*`, `LLM_PROVIDER_ORDER`, `OPENROUTER_*_MODELS`, `PRESCRIPTION_*`) are no longer read by any code.

> **Where data goes** (the app shows the same text, with the configured model names):
> - **Caption a photo:** read on this machine. Nothing is sent anywhere else.
> - **Read a prescription:** the photo is read on this machine. Only the medicine lines (no patient, doctor or clinic details) are sent as **text** to OpenRouter, or Groq if OpenRouter is busy.
> - **Check a food label:** the photo is read on this machine. Only if a printed value couldn't be parsed is the label's **text** sent to the AI; explanations send the parsed values.
> - **Identify a plant:** the photos are sent to **Pl@ntNet**. The medicinal-uses AI receives only the plant's name, never a photo.
> - OpenRouter is used with **free models only**, and free models may use what they receive for training.
>
> **Disclaimer:** informational only — not a substitute for a dietitian. Always check the printed label if you have an allergy.

---

## 🏋️ Pretrained weights

| | |
|---|---|
| Architecture | ResNet-101 encoder → 14×14×2048 features; additive soft attention with sigmoid gate; LSTMCell decoder (attention/embed/decoder dim 512) |
| Training data | MS-COCO 2014, Karpathy split, 5 captions/image, min word freq 5 (vocabulary 9,490) |
| Checkpoint | `BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar`, epoch 18, validation BLEU-4 0.2429 |
| Reported test BLEU-4 | ≈ 0.33 at beam size 3 (not re-measured here) |

`model/captioning/sat_models.py` defines the network. Its local changes are marked `LOCAL CHANGE`:
- `resnet101(weights=None)`, so nothing downloads
- an MPS-aware `device`
- a CPU fallback for the 14×14 adaptive pool on Apple MPS

> **History:** this project originally trained its own Show-Attend-Tell model on Flickr8k. That model was replaced on 2026-11-12. Its local "test" split overlapped the images it was trained on, so its reported BLEU-4 of 0.485 wasn't reproducible. The old code is backed up in `../archive/flickr8k_legacy.zip` (git-ignored).

---

### CATR and BLIP weights

- **CATR**: model code in `model/captioning/catr/`, weights `weight493084032.pth` (v3). Local changes, marked `LOCAL CHANGE`: `weights=None` for the backbone (no ImageNet download), and the device is chosen by the service. The service encodes each image once and decodes incrementally; on our tests its greedy output was identical to the reference implementation.
- **BLIP**: loaded through Hugging Face `transformers` (about 1 GB in the HF cache). The deploy script downloads it once during setup, so the first caption request is fast.

---

### Data licences

- Indian Medicine Dataset: MIT (as provided with the file).
- `data/medicines/uses_by_ingredient.json`: CC BY-SA 4.0; credited in the app wherever the descriptions appear and on the About page.
- Open Food Facts additives taxonomy (`data/product_label/additives.json`): ODbL.
- OCR models (`weights/ocr/`, downloaded by `scripts/setup_ocr_models.py`): Apache-2.0.

## 🧩 License
This project is licensed under the [MIT License](LICENSE)