# Product label evaluation set

Real photos only (no synthetic images). Ground truth NEVER comes from this app's own output.
Everything here except this README is git-ignored.

| folder | what | ground truth | counts in eval |
|---|---|---|---|
| `ndl/` | nutrition-table photos from the Hugging Face dataset `openfoodfacts/nutrient-detection-layout` (CC BY-SA 3.0) | per-100 g values tagged by the dataset's professional annotators, converted strictly (OCR noise is skipped, not guessed) | yes if the conversion was clean and passes the sanity check; otherwise after your review |
| `off_india/` | Indian products from Open Food Facts: a nutrition photo + an ingredients photo each | OFF volunteer values, marked `open_food_facts (volunteer, UNVERIFIED)` | only after you tick **verified** |
| `my_photos/` | your own photos | empty skeleton for you to fill in | only after you tick **verified** |

Build / update (run from `backend/`):

    venv/bin/python scripts/build_label_eval_set.py build      # downloads ndl + off_india (pauses between requests)
    venv/bin/python scripts/build_label_eval_set.py skeleton   # adds new files in my_photos/ with empty values
    venv/bin/python scripts/build_label_eval_set.py review     # review page at http://127.0.0.1:8765/

On the review page, compare every value with the photo, fix or clear it, then tick **verified** and press Save.
Save writes `ground_truth.json` (the previous version is kept as `ground_truth.json.bak`).

Evaluate the verified entries:

    venv/bin/python scripts/eval_product_label.py --json product_label_eval.json

`ground_truth.json` (version 2), one entry per product:

    "off_india/8904156707320": {
      "source": "open_food_facts (volunteer, UNVERIFIED)", "licence": "...", "verified": false,
      "images": {"nutrition": "off_india/8904156707320_nutrition.jpg", "ingredients": "off_india/..._ingredients.jpg"},
      "barcode": "8904156707320", "product_name": "...",
      "allergens": ["milk", "wheat_gluten"],          // CONTAINS only; [] = none; null = not scored
      "nutrients_per_100": {"energy_kcal": 520, "sugars_g": 3.1, ...},
      "origin": {...}, "notes": [...]
    }

Allergen names: milk, wheat_gluten, soy, peanut, tree_nuts, sesame, egg, fish, crustaceans, mustard, sulphites.
Scored nutrients (±5 %): energy (kcal, or kJ ÷ 4.184), sugars, fat, saturated fat, carbohydrate, protein, salt (or sodium × 2.5).
Products with two photos are scored as follows: nutrition values come from the nutrition photo and allergens from the ingredients photo.
Each photo is sent to Groq (label reading).
