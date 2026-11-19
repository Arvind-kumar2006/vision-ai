"""Food label tests (unittest; no AI and no network: the OCR and the AI are fakes).

Run from backend/:  venv/bin/python -m unittest tests.test_product_label -v

Nutri-Score reference cases: the official specification (Santé publique France,
Conditions of Use of the Nutri-Score trademark, version 26 June 2024, Exhibit 1-B,
pp. 28-32) publishes the point tables but NO worked examples. The cases below are
worked BY HAND from those tables; each comment shows the table lookup so the
expected result can be checked against the PDF independently of the code.
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.product_label import nutriscore  # noqa: E402
from model.product_label.additives import find_additives  # noqa: E402
from model.product_label.allergens import detect_allergens  # noqa: E402
from model.product_label.parser import parse_nutrition, to_per100  # noqa: E402

DATA = BACKEND_ROOT / "data" / "product_label"


def keys(items):
    return {i["allergen"] for i in items}


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
class DataFileTests(unittest.TestCase):
    def test_nutriscore_file_records_source_and_version(self):
        doc = json.loads((DATA / "nutriscore_thresholds.json").read_text())
        src = doc["_source"]
        self.assertIn("Santé publique France", src["publisher"])
        self.assertIn("26 June 2024", src["version"])
        self.assertIn("Exhibit 1-B".upper(), src["title"].upper())
        for table in ("general", "fats_nuts_seeds", "beverages"):
            self.assertIn(table, doc)

    def test_allergen_file_has_required_groups_and_indian_terms(self):
        doc = json.loads((DATA / "allergens.json").read_text())
        a = doc["allergens"]
        for group in ("milk", "wheat_gluten", "soy", "peanut", "tree_nuts", "sesame", "egg", "fish",
                      "crustaceans", "mustard"):
            self.assertIn(group, a)
        for group, term in (("milk", "paneer"), ("milk", "khoya"), ("wheat_gluten", "maida"),
                            ("wheat_gluten", "atta"), ("peanut", "moongphali"), ("tree_nuts", "kaju"),
                            ("sesame", "til"), ("mustard", "sarson")):
            self.assertIn(term, a[group]["keywords"])


# ---------------------------------------------------------------------------
# Allergens
# ---------------------------------------------------------------------------
class AllergenTests(unittest.TestCase):
    def test_required_mappings(self):
        cases = [("Sugar, whey powder, cocoa", "milk", "whey"),
                 ("Maida, palm oil, salt", "wheat_gluten", "Maida"),
                 ("Jaggery, til", "sesame", "til"),
                 ("Roasted groundnut, salt", "peanut", "groundnut")]
        for text, allergen, trigger in cases:
            r = detect_allergens(ingredients_text=text)
            hit = {i["allergen"]: i for i in r["contains"]}
            self.assertIn(allergen, hit, text)
            self.assertIn(trigger.lower(), hit[allergen]["trigger"].lower(), text)  # e.g. "whey powder"

    def test_may_contain_nuts_is_not_contains(self):
        r = detect_allergens(ingredients_text="Sugar, cocoa butter, emulsifier. May contain nuts.")
        self.assertIn("tree_nuts", keys(r["may_contain"]))
        self.assertNotIn("tree_nuts", keys(r["contains"]))
        r2 = detect_allergens(ingredients_text="Rice, salt", may_contain_statement="May contain traces of milk")
        self.assertEqual(keys(r2["may_contain"]), {"milk"})
        self.assertEqual(r2["contains"], [])

    def test_false_alarms_are_avoided(self):
        for text, not_allergen in (("Peanut butter, sugar", "milk"), ("Coconut milk, water", "milk"),
                                   ("Nutmeg, cardamom", "tree_nuts"), ("Eggless cake mix", "egg"),
                                   ("Rice flour (gluten-free)", "wheat_gluten"),
                                   ("Cocoa, sugar. Free from milk.", "milk")):
            r = detect_allergens(ingredients_text=text)
            self.assertNotIn(not_allergen, keys(r["contains"]) | keys(r["may_contain"]), text)

    def test_words_that_must_not_trigger(self):
        # substrings of keywords: til in lentils/distilled, rai in raisins, malt in maltodextrin,
        # cream in cream of tartar, nut in nutritional
        for text in ("Red lentils, salt", "Distilled vinegar, water", "Raisins, sugar", "Maltodextrin, flavour",
                     "Cream of tartar, starch", "Nutritional yeast, salt"):
            r = detect_allergens(ingredients_text=text)
            self.assertEqual((r["contains"], r["may_contain"]), ([], []), text)

    def test_non_dairy_and_plant_milks(self):
        self.assertEqual(detect_allergens(ingredients_text="Non-dairy creamer (vegetable oil)")["contains"], [])
        self.assertEqual(detect_allergens(ingredients_text="Non dairy whitener")["contains"], [])
        self.assertEqual(keys(detect_allergens(ingredients_text="Almond milk (water, almonds)")["contains"]),
                         {"tree_nuts"})
        self.assertEqual(keys(detect_allergens(ingredients_text="Soy milk (water, soybeans)")["contains"]), {"soy"})
        # real milk next to a non-dairy ingredient is still milk
        self.assertIn("milk", keys(detect_allergens(ingredients_text="Skimmed milk, non-dairy fat")["contains"]))

    def test_sulphites_fssai(self):
        for text in ("Sugar, preservative (INS 223)", "Dried apricots, sulphur dioxide", "Water, E224",
                     "Raisins, preservative (220)", "Sodium metabisulphite"):
            self.assertIn("sulphites", keys(detect_allergens(ingredients_text=text)["contains"]), text)
        for text in ("Sugar, INS 2230 flavour", "Salt, 225 g", "Sulphite-free wine"):
            self.assertNotIn("sulphites", keys(detect_allergens(ingredients_text=text)["contains"]), text)
        src = json.loads((DATA / "allergens.json").read_text())["allergens"]["sulphites"]["source"]
        self.assertIn("5(14)(viii)", src["rule"])
        self.assertIn("fssai.gov.in", src["url"])

    def test_lecithin_counts_as_soy_only_when_stated(self):
        self.assertIn("soy", keys(detect_allergens(ingredients_text="Emulsifier (lecithin (soy))")["contains"]))
        r = detect_allergens(ingredients_text="Emulsifier (INS 322)")
        self.assertNotIn("soy", keys(r["contains"]))
        self.assertTrue(any("322" in n for n in r["notes"]))



# ---------------------------------------------------------------------------
# Additives (Open Food Facts taxonomy)
# ---------------------------------------------------------------------------
class AdditiveTests(unittest.TestCase):
    def test_source_commit_and_licence_recorded(self):
        src = json.loads((DATA / "additives.json").read_text())["_source"]
        self.assertRegex(src["commit"], r"^[0-9a-f]{40}$")
        self.assertIn("ODbL", src["licence"].replace("Open Database License (ODbL)", "ODbL"))
        nns = json.loads((DATA / "additives.json").read_text())["non_nutritive_sweeteners"]
        self.assertTrue(nns["approximate"])
        self.assertNotIn("E420", nns["codes"])  # polyol
        self.assertIn("E955", nns["codes"])

    def test_indian_label_forms(self):
        r = find_additives("Sugar, emulsifiers (322, 471), acidity regulator (INS 330), colour (150d), E211")
        self.assertEqual({a["code"] for a in r["additives"]}, {"E322", "E471", "E330", "E150d", "E211"})
        self.assertEqual(next(a for a in r["additives"] if a["code"] == "E211")["classes"], ["preservative"])

    def test_sweeteners_for_beverage_rule(self):
        r = find_additives("Carbonated water, sweetener (INS 955), acesulfame potassium")
        self.assertEqual(r["non_nutritive_sweeteners"], ["E950", "E955"])
        self.assertFalse(find_additives("Sorbitol (E420), maltitol")["has_non_nutritive_sweetener"])
        self.assertFalse(find_additives("Salt, sugar, spices")["additives"])

    def test_numbers_that_are_not_additives(self):
        self.assertEqual(find_additives("Salt 225 g, sodium 1050 mg")["additives"], [])
        # a well-formed but unknown INS code is reported as unknown, never given a name
        unknown = find_additives("Stabiliser (INS 2230)")["additives"]
        self.assertEqual([(a["code"], a["known"], a["name"]) for a in unknown], [("E2230", False, None)])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
PANEL = """NUTRITIONAL INFORMATION (Approx.)
Serving size: 30 g
Nutrient | Per 100 g | Per serve (30 g) | %RDA*
Energy (kcal) 480 144 7%
Protein (g) 7.5 2.3
Total Sugars (g) 28.0 8.4
Total Fat (g) 20.0 6.0
Saturated Fat (g) 9.1 2.7 12%
Trans Fat (g) <0.1 <0.1
Dietary Fibre (g) 2.1 0.6
Sodium (mg) 450 135 7%"""


class ParserTests(unittest.TestCase):
    def test_slovenian_croatian_nutrient_names_are_normalised(self):
        from model.product_label.parser import parse_nutrition, to_per100
        text = """Energija 203 kJ / 48 kcal
        Masti 1.3 g
        Zasićene masne kiseline 0.9 g
        Ugljikohidrati 5.1 g
        Šećeri 5.1 g
        Bjelančevine 4.0 g
        Sol 0.11 g"""
        parsed = to_per100(parse_nutrition(text), source="test")
        self.assertAlmostEqual(parsed["per100"]["fat_g"], 1.3)
        self.assertAlmostEqual(parsed["per100"]["saturated_fat_g"], 0.9)
        self.assertAlmostEqual(parsed["per100"]["carbohydrate_g"], 5.1)
        self.assertAlmostEqual(parsed["per100"]["sugars_g"], 5.1)
        self.assertAlmostEqual(parsed["per100"]["protein_g"], 4.0)
        self.assertAlmostEqual(parsed["per100"]["salt_g"], 0.11)

    def test_per_100_column_is_used_and_serving_line_is_not_a_header(self):
        r = to_per100(parse_nutrition(PANEL))
        self.assertEqual(r["per100"]["energy_kcal"], 480.0)
        self.assertEqual(r["per100"]["saturated_fat_g"], 9.1)
        self.assertEqual(r["per100"]["sodium_mg"], 450.0)
        self.assertIn("upper bound", r["flags"]["trans_fat_g"][0])
        self.assertEqual(r["field_sources"]["energy_kcal"], "label_regex")

    def test_per_serving_converted_with_serving_size(self):
        r = to_per100(parse_nutrition("Nutrition information per serve (25 g)\nEnergy 120 kcal\n"
                                      "Sugars 9 g\nSodium 60 mg"))
        self.assertEqual(r["per100"]["energy_kcal"], 480.0)   # 120 x 100/25
        self.assertEqual(r["per100"]["sugars_g"], 36.0)       # 9 x 4
        self.assertEqual(r["per100"]["sodium_mg"], 240.0)

    def test_missing_serving_size_means_no_per100_and_no_score(self):
        r = to_per100(parse_nutrition("Nutrition facts per serving\nEnergy 120 kcal\nSugars 9 g"))
        self.assertFalse(r["available"])
        self.assertTrue(any("per-100 g values unavailable" in n for n in r["notes"]))
        score = nutriscore.compute(r["per100"])
        self.assertIsNone(score["grade"])
        self.assertTrue(any("Not scored" in n for n in score["notes"]))

    def test_one_line_beverage_and_number_formats(self):
        r = to_per100(parse_nutrition("Per 100 ml: Energy 180 kJ / 43 kcal, Sugars 10.6 g, Fat 0 g, Salt 0.02 g"))
        self.assertEqual(r["per100"], {"energy_kcal": 43.0, "energy_kj": 180.0, "sugars_g": 10.6,
                                       "fat_g": 0.0, "salt_g": 0.02})
        self.assertEqual(r["basis"], "per 100 mL")
        r2 = to_per100(parse_nutrition("Per 100 g\nEnergy 1,234 kJ\nSodium 1,050 mg\nProtein 7,5 g"))
        self.assertEqual(r2["per100"], {"energy_kj": 1234.0, "sodium_mg": 1050.0, "protein_g": 7.5})

    def test_ingredients_line_is_not_a_nutrition_row(self):
        # regression: "...Sugar, ..., Emulsifier (322), Salt." was parsed as sugars = 322 g
        t = ("INGREDIENTS: Refined Wheat Flour (Maida), Sugar, Palm Oil, Emulsifier (322), Salt.\n"
             "Per 100 g\nTotal Sugars (g) 24.0\nSodium (mg) 380")
        self.assertEqual(to_per100(parse_nutrition(t))["per100"], {"sugars_g": 24.0, "sodium_mg": 380.0})

    # Real transcripts (Open Food Facts photos via nutrient-detection-layout), values = dataset annotations.
    def test_french_inline_paragraph(self):
        t = ("ANALYSE NUTRITIONNELLE MOYENNE POUR 100 G\n"
             "Énergie : 739 kJ soit 177 kcal - Matières grasses : 11 g dont acides gras saturés : 10 g - Glucides : 18 g "
             "dont sucres : 17 g - Protéines : 1,8 g - Sel : 0,05 g - Calcium : 120 mg soit 15% de la VNR")
        self.assertEqual(to_per100(parse_nutrition(t))["per100"], {
            "energy_kcal": 177.0, "energy_kj": 739.0, "fat_g": 11.0, "saturated_fat_g": 10.0, "carbohydrate_g": 18.0,
            "sugars_g": 17.0, "protein_g": 1.8, "salt_g": 0.05})          # calcium never leaks into salt

    def test_spanish_and_multilingual_rows(self):
        es = ("INFORMACIÓN NUTRICIONAL POR 100 g\nValor energético 425 kJ / 101 kcal\nGrasas, de las cuales: 1,16 g\n"
              "Saturadas 0,52 g\nPoliinsaturadas 0,64 g\nHidratos de carbono, de los cuales: 0 g\nAzúcares 0 g\n"
              "Proteínas 22,78 g\nSal 0 g")
        self.assertEqual(to_per100(parse_nutrition(es))["per100"], {
            "energy_kcal": 101.0, "energy_kj": 425.0, "fat_g": 1.16, "saturated_fat_g": 0.52, "carbohydrate_g": 0.0,
            "sugars_g": 0.0, "protein_g": 22.78, "salt_g": 0.0})
        multi = ("Déclaration nutritionnelle / Nährwertdeklaration:\n100 g\n25 g = 1 tranche / Scheibe / pour / per 25 g\n"
                 "énergie / Energie / energie / energia / 1078 kJ / 256 kcal\n269 kJ / 64 kcal\n"
                 "matières grasses / Fett / tuky / tłuszcz 6,2 g\n1,6 g\n"
                 "dont acides gras saturés / davon gesättigte Fettsäuren / z toho nasycené mastné kyseliny 0,8 g\n0,2 g")
        self.assertEqual(to_per100(parse_nutrition(multi))["per100"], {
            "energy_kcal": 256.0, "energy_kj": 1078.0, "fat_g": 6.2, "saturated_fat_g": 0.8})

    def test_unnamed_energy_and_reference_intake(self):
        t = ("Val nut(100g) : 299 Kcal, 1253 KJ, Lipides 12,6g (dont acides gras saturés 2,5g), Protéines 6,6g, "
             "Glucides 39,7g (dont sucres 15,4g), sel 0,8g")
        r = to_per100(parse_nutrition(t))["per100"]
        self.assertEqual((r["energy_kcal"], r["fat_g"], r["saturated_fat_g"], r["sugars_g"], r["salt_g"]),
                         (299.0, 12.6, 2.5, 15.4, 0.8))
        ri = to_per100(parse_nutrition("Per 100 g\nSugars 5 g\nReference intake of an average adult (8400 kJ / 2000 kcal)"))
        self.assertNotIn("energy_kcal", ri["per100"])

    def test_missing_header_assumes_first_column_and_flags_it(self):
        t = ("Nährwerte Ø/Valeurs nutritives moyennes\n"
             "Energie/énergie/energia 660 kJ (158 kcal) 330 kJ (79 kcal) 4 % 8400 kJ (2000 kcal)\n"
             "Fett/lipides/grassi 7,2 g 3,6 g 5 % 70 g\nSalz/sel/sale 3 g 1,5 g 25 % 6 g")
        r = to_per100(parse_nutrition(t))
        self.assertEqual(r["per100"], {"energy_kcal": 158.0, "energy_kj": 660.0, "fat_g": 7.2, "salt_g": 3.0})
        self.assertIn("assumed to be per 100 g", r["flags"]["fat_g"][0])
        self.assertTrue(any("column header" in n for n in r["notes"]))

    def test_foreign_ingredients_line_is_not_nutrition(self):
        t = "Per 100 g\nIngrédients : sucre, sel 2 %, lait\nZutaten: Zucker 40 %, Salz\nSucres 12 g"
        self.assertEqual(to_per100(parse_nutrition(t))["per100"], {"sugars_g": 12.0})



# ---------------------------------------------------------------------------
# Nutri-Score (hand-worked from Exhibit 1-B tables; see module docstring)
# ---------------------------------------------------------------------------
class NutriScoreTests(unittest.TestCase):
    def check(self, result, grade, score, n, p):
        self.assertEqual((result["grade"], result["score"], result["n_points"], result["p_points"]),
                         (grade, score, n, p), result)

    def test_case1_general_food_high_n_excludes_protein(self):
        # Table 5: energy 1800 kJ > 1675 -> 5; sat fat 9.5 > 9 -> 9; sugars 30 > 27 -> 8; salt 1.1 > 1 -> 5
        # N = 27 >= 11 so protein is not counted (1-a); Table 6: fibre 3.5 > 3.0 -> 1; FVL 0 -> 0
        # score = 27 - 1 = 26 -> 2-a: 19..max = E
        r = nutriscore.compute({"energy_kj": 1800, "saturated_fat_g": 9.5, "sugars_g": 30, "salt_g": 1.1,
                                "protein_g": 7, "fibre_g": 3.5}, fvl_percent=0)
        self.check(r, "E", 26, 27, 1)
        self.assertTrue(any("Protein not counted" in n for n in r["notes"]))

    def test_case2_general_food_low_n_counts_protein(self):
        # Table 5: energy 250 < 335 -> 0; sat fat 1.2 > 1 -> 1; sugars 4.0 > 3.4 -> 1; salt 0.1 -> 0 => N = 2
        # Table 6: protein 3.5 > 2.4 -> 1; fibre 0 -> 0 => P = 1; score 1 -> 2-a: 1..2 = B
        r = nutriscore.compute({"energy_kj": 250, "saturated_fat_g": 1.2, "sugars_g": 4.0, "salt_g": 0.1,
                                "protein_g": 3.5, "fibre_g": 0}, fvl_percent=0)
        self.check(r, "B", 1, 2, 1)

    def test_case3_cheese_keeps_protein(self):
        # Table 5: energy 1600 > 1340 -> 4; sat fat 18 > 10 -> 10; sugars 0.5 -> 0; salt 1.7 > 1.6 -> 8 => N = 22
        # 1-a: cheese -> protein counted even though N >= 11; protein 25 > 17 -> 7 => P = 7
        # score 15 -> 11..18 = D   (a non-cheese food would score 22 -> E)
        food = {"energy_kj": 1600, "saturated_fat_g": 18, "sugars_g": 0.5, "salt_g": 1.7,
                "protein_g": 25, "fibre_g": 0}
        self.check(nutriscore.compute(food, category="cheese", fvl_percent=0), "D", 15, 22, 7)
        self.check(nutriscore.compute(food, category="general", fvl_percent=0), "E", 22, 22, 0)

    def test_case4_sugary_beverage(self):
        # Table 9 (per 100 mL): energy 180 > 150 -> 3; sugars 10.6 > 10 -> 9; sat 0 -> 0; salt 0 -> 0;
        # no sweeteners -> 0 => N = 12; Table 10: all 0 => P = 0; score 12 -> 2-c: 10..max = E
        r = nutriscore.compute({"energy_kj": 180, "sugars_g": 10.6, "saturated_fat_g": 0, "salt_g": 0,
                                "protein_g": 0, "fibre_g": 0}, category="beverage", fvl_percent=0)
        self.check(r, "E", 12, 12, 0)

    def test_case5_sweetened_diet_beverage(self):
        # Table 9: energy 2 -> 0; sugars 0 -> 0; sweeteners present -> 4 => N = 4; P = 0
        # score 4 -> 2-c: 3..6 = C
        r = nutriscore.compute({"energy_kj": 2, "sugars_g": 0, "saturated_fat_g": 0, "salt_g": 0,
                                "protein_g": 0, "fibre_g": 0}, category="beverage",
                               non_nutritive_sweeteners=True, fvl_percent=0)
        self.check(r, "C", 4, 4, 0)

    def test_case6_olive_oil_fats_table(self):
        # Table 7: energy from saturates 14 x 37 = 518 > 480 -> 4; sugars 0 -> 0;
        # saturates/lipids 14 % (>= 10, < 16) -> 1; salt 0 -> 0 => N = 5 (< 7, so N - P)
        # Table 8: FVL 100 % > 80 -> 5 (olive oil counts, footnote) => P = 5; score 0 -> 2-b: -5..2 = B
        r = nutriscore.compute({"energy_kj": 3700, "fat_g": 100, "saturated_fat_g": 14, "sugars_g": 0,
                                "salt_g": 0, "protein_g": 0, "fibre_g": 0},
                               category="fats_nuts_seeds", fvl_percent=100)
        self.check(r, "B", 0, 5, 5)

    def test_case7_red_meat_protein_cap(self):
        # Table 5: energy 800 > 670 -> 2; sat 3.5 > 3 -> 3; sugars 0 -> 0; salt 0.15 -> 0 => N = 5
        # Table 6: protein 20 > 17 -> 7, capped to 2 for red meat => P = 2; score 3 -> 3..10 = C
        r = nutriscore.compute({"energy_kj": 800, "saturated_fat_g": 3.5, "sugars_g": 0, "salt_g": 0.15,
                                "protein_g": 20, "fibre_g": 0}, category="red_meat", fvl_percent=0)
        self.check(r, "C", 3, 5, 2)

    def test_water_is_a(self):
        self.assertEqual(nutriscore.compute({}, category="water")["grade"], "A")

    def test_boundary_value_gets_lower_points(self):
        # Table 5 energy: 0 points '< 335', 1 point '> 335'; exactly 335 -> 0 (documented interpretation)
        r = nutriscore.compute({"energy_kj": 335, "saturated_fat_g": 0, "sugars_g": 0, "salt_g": 0,
                                "protein_g": 0, "fibre_g": 0}, fvl_percent=0)
        self.assertEqual(r["breakdown"][0]["points"], 0)

    def test_kcal_and_sodium_conversions(self):
        # 100 kcal x 4.184 = 418.4 kJ > 335 -> 1; sodium 400 mg x 2.5 = 1.0 g salt -> > 0.8 -> 4
        r = nutriscore.compute({"energy_kcal": 100, "saturated_fat_g": 0, "sugars_g": 0, "sodium_mg": 400,
                                "protein_g": 0, "fibre_g": 0}, fvl_percent=0)
        parts = {b["component"]: b["points"] for b in r["breakdown"]}
        self.assertEqual((parts["energy"], parts["salt"]), (1, 4))

    def test_missing_fibre_and_fvl_are_zero_with_notes(self):
        r = nutriscore.compute({"energy_kj": 250, "saturated_fat_g": 1.2, "sugars_g": 4.0, "salt_g": 0.1,
                                "protein_g": 3.5})
        self.assertEqual(r["grade"], "B")
        self.assertTrue(any("Fibre not available" in n and "pessimistic" in n for n in r["notes"]))
        self.assertTrue(any("Fruit/vegetable" in n for n in r["notes"]))

    def test_missing_salt_means_no_score(self):
        r = nutriscore.compute({"energy_kj": 1000, "saturated_fat_g": 1, "sugars_g": 5})
        self.assertIsNone(r["grade"])
        self.assertIn("salt_g", r["missing"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Stage 2: barcodes (check digits + OpenCV detection on a synthetic EAN-13)
# ---------------------------------------------------------------------------

from model.product_label.sanity import NOTE as SANITY_NOTE  # noqa: E402
from model.product_label.sanity import check as sanity_check  # noqa: E402

# Open Food Facts record 6111099000247 "Hule D'olive" as returned on 2026-11-12 (nutriments per 100 g)
OFF_OLIVE_OIL = {"energy_kj": 4, "energy_kcal": 822, "fat_g": 90, "saturated_fat_g": 13, "carbohydrate_g": 0,
                 "sugars_g": 0, "added_sugars_g": 0, "fibre_g": 66, "protein_g": 0, "salt_g": 0, "sodium_mg": 0}


class SanityTests(unittest.TestCase):
    def test_off_olive_oil_record_is_flagged(self):
        r = sanity_check(OFF_OLIVE_OIL, "open_food_facts")
        self.assertFalse(r["ok"])
        self.assertEqual(r["note"], "values may be misread or incorrect in the database")
        self.assertEqual({f["check"] for f in r["failures"]}, {"energy_units", "fibre_above_carbohydrate"})

    def test_plausible_label_passes(self):
        r = sanity_check({"energy_kcal": 482, "protein_g": 7.2, "carbohydrate_g": 66, "fat_g": 21, "sugars_g": 24,
                          "added_sugars_g": 21.5, "fibre_g": 3.4, "saturated_fat_g": 10.2, "trans_fat_g": 0.1}, "label")
        self.assertTrue(r["ok"], r["failures"])

    def test_each_rule(self):
        cases = {
            "energy_balance": {"energy_kcal": 480, "protein_g": 7.5, "carbohydrate_g": 6.8, "fat_g": 20},  # misread 68 -> 6.8
            "over_100g": {"sugars_g": 280},
            "sugars_above_carbohydrate": {"carbohydrate_g": 20, "sugars_g": 28},
            "fat_parts_above_fat": {"fat_g": 10, "saturated_fat_g": 12},
            "added_sugars_above_sugars": {"sugars_g": 12, "added_sugars_g": 14.13},
        }
        for rule, values in cases.items():
            self.assertIn(rule, {f["check"] for f in sanity_check(values, "label")["failures"]}, rule)

    def test_water_and_tiny_values_are_not_flagged(self):
        self.assertTrue(sanity_check({"energy_kcal": 0, "protein_g": 0, "carbohydrate_g": 0, "fat_g": 0}, "label")["ok"])
        self.assertTrue(sanity_check({"energy_kcal": 2, "protein_g": 0, "carbohydrate_g": 0.1, "fat_g": 0}, "label")["ok"])


# ---------------------------------------------------------------------------
# Stage 3: pipeline (reader, Open Food Facts and Groq all faked)
# ---------------------------------------------------------------------------

class EvalSetBuilderTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(BACKEND_ROOT / "scripts"))
        import build_label_eval_set as b
        self.b = b

    def test_value_parsing_never_guesses(self):
        pv = self.b.parse_value
        self.assertEqual(pv("1,3 g ", "g"), (1.3, None))
        self.assertEqual(pv("(158 kcal) ", "kcal"), (158.0, None))
        self.assertEqual(pv("0,38 g", "mg"), (380.0, None))       # sodium in g -> mg
        self.assertEqual(pv("250 mg", "g")[0], 0.25)
        self.assertEqual(pv("294kJ/70kcal\n", "kj"), (294.0, None))   # combined span: unit picks the number
        self.assertEqual(pv("294kJ/70kcal\n", "kcal"), (70.0, None))
        for bad, unit in (("10gl ", "g"), ("<0.5 g", "g"), ("2 345 kJ", "kj"), ("12 kJ", "kcal"), ("0.4", "mg")):
            self.assertIsNone(pv(bad, unit)[0], bad)

    def test_bio_spans_and_conflicting_annotations(self):
        names = ["O", "B-SUGARS_100G", "I-SUGARS_100G", "B-FAT_100G", "I-FAT_100G"]
        row = {"tokens": ["Sugar ", "12,5 ", "g ", "Fat ", "3 ", "g ", "Fat ", "4 g"], "ner_tags": [0, 1, 2, 0, 3, 4, 0, 3]}
        values, notes = self.b.ndl_values(row, names)
        self.assertEqual(values, {"sugars_g": 12.5})
        self.assertTrue(any("annotated twice" in n for n in notes))

    def test_review_edits_apply_and_verify(self):
        tmp = Path(tempfile.mkdtemp())
        gt = {"version": 2, "entries": {"off_india/1": self.b.empty_entry(self.b.SRC_OFF, "x", {"nutrition": "a.jpg"}, False)}}
        gt["entries"]["off_india/1"]["nutrients_per_100"] = {"sugars_g": 30.0}
        (tmp / "ground_truth.json").write_text(json.dumps(gt))
        changed = self.b.apply_edits(tmp, {"off_india/1": {"verified": True, "allergens": ["milk"],
                                                          "nutrients_per_100": {"sugars_g": 24.0, "fat_g": None}}})
        e = json.loads((tmp / "ground_truth.json").read_text())["entries"]["off_india/1"]
        self.assertEqual(changed, 1)
        self.assertEqual((e["verified"], e["nutrients_per_100"], e["allergens"]), (True, {"sugars_g": 24.0}, ["milk"]))
        self.assertEqual(e["verified_by"], "project owner (review page)")
        self.assertTrue((tmp / "ground_truth.json.bak").exists() and (tmp / "review.html").exists())

    def test_skeleton_lists_own_photos_with_empty_values(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "my_photos").mkdir()
        photo().save(tmp / "my_photos" / "biscuit.jpg")
        gt = self.b.load_gt(tmp)
        self.assertEqual(self.b.skeleton(tmp, gt), ["my_photos/biscuit.jpg"])
        e = gt["entries"]["my_photos/biscuit.jpg"]
        self.assertFalse(e["verified"])
        self.assertTrue(all(v is None for v in e["nutrients_per_100"].values()))
        self.assertEqual(e["allergens"], {"contains": None, "may_contain": None})
        page = self.b.review_html(gt)
        self.assertIn('data-k="allergens.contains"', page)
        self.assertIn('data-k="allergens.may_contain"', page)
        (tmp / "ground_truth.json").write_text(json.dumps(gt))
        self.b.apply_edits(tmp, {"my_photos/biscuit.jpg": {"verified": True, "nutrients_per_100": {"sugars_g": 12.0},
                                                          "allergens": {"contains": ["milk"], "may_contain": []}}})
        saved = json.loads((tmp / "ground_truth.json").read_text())["entries"]["my_photos/biscuit.jpg"]
        self.assertEqual(saved["allergens"], {"contains": ["milk"], "may_contain": []})
        self.assertEqual(saved["nutrients_per_100"]["sugars_g"], 12.0)
        self.assertIsNone(saved["nutrients_per_100"]["fat_g"])


# ---------------------------------------------------------------------------
# Real photo (2026-11-12): ingredients not read, multilingual table with a garbled first name
# ---------------------------------------------------------------------------
# Transcript the reader returned for the project owner's photo (yellow multilingual FR/DE/CS/PL panel,
# no ingredients in frame). Printed per 100 g: 1078 kJ, fat 6.2, sat 0.8, carbs 41, sugars 5.8,
# fibre 8.7, protein 4.4, salt 0.91.
REAL_MULTILINGUAL_TRANSCRIPT = [
    "Serving size: 25 g = 1 tranche / Scheibe / pour / per",
    "Déclaration nutritionnelle / Nährwertdeklaration / Výživové údaje na / Informacja o wartości odżywczej:",
    "100 g 25 g = 1 tranche / Scheibe / pour / per % RI**",
    "Énergie / Energie / energia / energia / wartość energetyczna 1078 kJ 269 kJ 256 kJ 64 kJ 3%",
    "matières grasses / Fette / tuky / tłuszcze 6,2 g 1,6 g 2%",
    "dont acides gras saturés / davon gesättigte Fettsäuren / z toho nasycené kyseliny / w tym kwasy tłuszczowe nasycone 0,8 g 0,2 g 1%",
    "glucides / Kohlenhydrate / sacharidy / węglowodany 41 g 10 g 4%",
    "dont sucres / davon Zucker / z toho cukry / w tym cukry 5,8 g 1,5 g 2%",
    "fibres alimentaires / Ballaststoffe / vláknina / błonnik 8,7 g 2,2 g -",
    "protéines / Eiweiß / bílkoviny / białko 4,4 g 1,1 g 2%",
    "sel / Salz / sůl / sól 0,91 g 0,23 g 4%",
]



class MultilingualRowNameTests(unittest.TestCase):
    """A garbled first language must not drop a row whose other names are readable."""

    def test_real_photo_transcript_parses_all_seven_values(self):
        per = to_per100(parse_nutrition("\n".join(REAL_MULTILINGUAL_TRANSCRIPT)))["per100"]
        want = {"fat_g": 6.2, "saturated_fat_g": 0.8, "carbohydrate_g": 41, "sugars_g": 5.8, "fibre_g": 8.7,
                "protein_g": 4.4, "salt_g": 0.91, "energy_kj": 1078}
        for k, v in want.items():
            self.assertAlmostEqual(per.get(k), v, msg=k)

    def test_garbled_first_name_falls_back_to_other_languages(self):
        lines = ["Déclaration nutritionnelle / Nährwertdeklaration", "100 g 25 g % RI",
                 "[illegible] grasses / Fett / tuky / tłuszcz 6,2 g 1,6 g 2%",
                 "atières grasses / Fett / tuky / tłuszcz 6,2 g 1,6 g 2%",
                 "lont sucres / davon Zucker / z toho cukry / w tym cukry 5,8 g 1,5 g 2%",
                 "el / Salz / sůl / sól 0,91 g 0,23 g 4%"]
        rows = parse_nutrition("\n".join(lines))["rows"]
        got = {(r["nutrient"], r["value"]) for r in rows if r["basis"] == "per_100g"}
        self.assertTrue({("fat", 6.2), ("sugars", 5.8), ("salt", 0.91)} <= got, got)

    def test_ingredient_line_with_slashes_is_still_ignored(self):
        self.assertEqual(parse_nutrition("Ingredients: sugar / salt / milk 3 g")["rows"], [])




# ---------------------------------------------------------------------------
# Pipeline: image -> preprocessing -> OCR -> parser (AI only for missing values) -> allergens + Nutri-Score
# ---------------------------------------------------------------------------
from PIL import Image  # noqa: E402

from model.common.llm import LLM, AI_BUSY_NOTE, data_block, numbers_supported  # noqa: E402
from model.product_label.explain import Explainer, facts_for_llm, template_explanation  # noqa: E402
from model.product_label.pipeline import UNREADABLE, ProductLabelPipeline, label_text, missing_values, split_ingredients  # noqa: E402

TABLE = ["Nutrition Information Per 100 g", "Energy 480 kcal", "Fat 20 g", "of which saturates 9 g",
         "Carbohydrate 68 g", "of which sugars 24 g", "Protein 6 g", "Salt 0.8 g"]
INGREDIENTS = ["Ingredients: wheat flour, sugar, palm oil, milk solids, salt.", "May contain nuts."]


class FakeOCR:
    name = "fake OCR"
    available = True
    error = None

    def __init__(self, lines):
        self.lines = lines

    def read(self, bgr, hindi=None):
        rows = [{"text": t, "confidence": 0.95, "box": [0, i, 1, i + 1]} for i, t in enumerate(self.lines)]
        return {"ok": bool(rows), "rows": rows, "lines": list(self.lines), "ms": 1, "error": None if rows else "no text found"}


class FakeCompletions:
    def __init__(self, *answers, error=None):
        self.answers, self.error, self.calls = list(answers), error, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        text = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=text))])


def fake_llm(*answers, error=None):
    c = FakeCompletions(*answers, error=error)
    return LLM.with_client(mock.Mock(chat=mock.Mock(completions=c))), c


def photo():
    return Image.new("RGB", (300, 200), "white")


def run(lines, *answers, error=None, llm=None):
    if llm is None:
        llm, calls = fake_llm(*(answers or ("unused",)), error=error)
    else:
        calls = None
    r = ProductLabelPipeline(ocr=FakeOCR(lines), llm=llm).run(photo())
    return r, calls


class PipelineTests(unittest.TestCase):
    def test_no_text_means_unreadable_message(self):
        r, calls = run([])
        self.assertEqual((r["status"], r["message"]), ("unreadable", UNREADABLE))
        self.assertEqual(calls.calls, [])

    def test_everything_read_means_no_ai_call(self):
        r, calls = run(TABLE + INGREDIENTS)
        self.assertEqual(calls.calls, [])
        self.assertEqual(r["nutrition_per_100"]["sugars_g"], 24.0)
        self.assertEqual(set(r["field_sources"].values()), {"read_on_device"})
        self.assertEqual(r["allergens"]["status"], "found")
        self.assertEqual({a["allergen"] for a in r["allergens"]["contains"]}, {"wheat_gluten", "milk"})
        self.assertEqual({a["allergen"] for a in r["allergens"]["may_contain"]}, {"tree_nuts"})
        self.assertIn(r["nutriscore"]["grade"], list("ABCDE"))
        self.assertIsNone(r["ai"])
        from model.product_label.schemas import ProductLabelResult
        ProductLabelResult(**r)   # the API model accepts the result

    def test_ai_only_for_a_printed_value_that_was_not_parsed(self):
        lines = [l for l in TABLE if not l.startswith("of which sugars")] + ["of which sugars", "24 g"]
        r, calls = run(lines + INGREDIENTS, '{"sugars_g": 24}')
        self.assertEqual(len(calls.calls), 1)
        sent = calls.calls[0]["messages"][1]["content"]
        self.assertIn("sugars_g", sent)
        self.assertNotIn("protein_g,", sent.split("Requested keys:")[1])   # only the missing value is asked for
        self.assertEqual((r["nutrition_per_100"]["sugars_g"], r["field_sources"]["sugars_g"]), (24.0, "ai_from_text"))

    def test_ai_number_not_in_the_text_is_rejected(self):
        lines = [l for l in TABLE if not l.startswith("of which sugars")] + ["of which sugars"]
        r, _ = run(lines, '{"sugars_g": 31}')
        self.assertNotIn("sugars_g", r["nutrition_per_100"])
        self.assertEqual(r["rejected_ai_values"][0]["field"], "sugars_g")

    def test_ai_busy_keeps_parsed_values_and_says_so(self):
        lines = [l for l in TABLE if not l.startswith("of which sugars")] + ["of which sugars"]
        r, _ = run(lines, error=RuntimeError("429"))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["ai_note"], AI_BUSY_NOTE)
        self.assertIn(AI_BUSY_NOTE, r["photo_notes"])
        self.assertEqual(r["nutrition_per_100"]["fat_g"], 20.0)

    def test_backup_provider_is_used_when_the_first_fails(self):
        first = FakeCompletions("x", error=RuntimeError("429"))
        second = FakeCompletions('{"sugars_g": 24}')
        from model.common.llm import Provider
        llm = LLM([Provider("OpenRouter", "m:free", mock.Mock(chat=mock.Mock(completions=first))),
                   Provider("Groq", "q", mock.Mock(chat=mock.Mock(completions=second)))])
        lines = [l for l in TABLE if not l.startswith("of which sugars")] + ["of which sugars", "24 g"]
        r, _ = run(lines, llm=llm)
        self.assertEqual(r["ai"]["provider"], "Groq")
        self.assertEqual(r["nutrition_per_100"]["sugars_g"], 24.0)

    def test_values_that_do_not_add_up_ask_the_ai_and_keep_only_a_consistent_answer(self):
        bad = ["Per 100 g", "Energy 480 kcal", "Fat 2 g", "Carbohydrate 68 g", "of which sugars 24 g", "Protein 6 g",
               "Salt 0.8 g", "Fat 20 g"]
        r, calls = run(bad, '{"fat_g": 20, "energy_kcal": 480, "carbohydrate_g": 68, "protein_g": 6}')
        self.assertEqual(len(calls.calls), 1)
        self.assertEqual(r["nutrition_per_100"]["fat_g"], 20.0)
        self.assertEqual(r["field_sources"]["fat_g"], "ai_from_text")

    def test_missing_values_and_label_text_helpers(self):
        self.assertEqual(missing_values("Energy 480 kcal\nSugars", {"energy_kcal": 480}), ["sugars_g"])
        t = label_text(["Ingredients: oats, sugar,", "salt.", "Nutrition information", "Contains: milk"])
        self.assertEqual(t["ingredients_text"], "oats, sugar, salt.")
        self.assertEqual(t["contains_statement"], "Contains: milk")
        self.assertEqual(split_ingredients("a (b, c), d"), ["a (b, c)", "d"])


class AllergenUnknownTests(unittest.TestCase):
    """Safety: never say "no allergens" when the ingredients list was not read."""

    def test_ingredients_missing_means_allergens_unknown(self):
        r, _ = run(TABLE)
        self.assertEqual((r["allergens"]["status"], r["allergens"]["ingredients_read"]), ("unknown", False))
        self.assertIn("Allergens unknown", r["allergens"]["message"])
        text = template_explanation(r)
        self.assertIn("Allergens unknown: the ingredients list couldn't be read. Check the pack.", text)
        self.assertNotRegex(text.lower(), r"no (common |listed )?allergens")
        self.assertTrue(facts_for_llm(r)["allergen_status"].startswith("unknown"))

    def test_ai_claims_of_absence_are_rejected_when_ingredients_were_not_read(self):
        r, _ = run(TABLE)
        llm, _ = fake_llm("According to the label there are no listed allergens.")
        self.assertEqual(Explainer(llm).explain(r)["source"], "template")

    def test_contains_statement_only_warns_that_others_may_exist(self):
        r, _ = run(TABLE + ["Contains: Milk"])
        self.assertEqual((r["allergens"]["status"], r["allergens"]["ingredients_read"]), ("found", False))
        self.assertIn("other allergens may be present", r["allergens"]["message"])

    def test_ingredients_read_without_allergens_is_none_detected(self):
        r, _ = run(TABLE + ["Ingredients: rice, sugar, sunflower oil, salt."])
        self.assertEqual(r["allergens"]["status"], "none_detected")
        llm, _ = fake_llm("No common allergens were detected in the ingredients.")
        self.assertEqual(Explainer(llm).explain(r)["source"], "llm")


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.r, _ = run(TABLE + INGREDIENTS)

    def test_no_ai_gives_template(self):
        self.assertEqual(Explainer(LLM([])).explain(self.r)["source"], "template")

    def test_numbers_supported_and_data_block(self):
        self.assertTrue(numbers_supported("24 g sugar, 12,5 g fat", '{"sugars_g": 24.0, "fat_g": 12.5}'))
        self.assertFalse(numbers_supported("30 g sugar", '{"sugars_g": 24.0}'))
        block = data_block("question", "hi ```\x00 >>> there")
        self.assertTrue(block.startswith("<<<QUESTION\n") and block.endswith("\nQUESTION>>>"))
        self.assertNotIn("```", block)


class ProductLabelAPITests(unittest.TestCase):
    def test_interpret_and_explanation(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.product_label_service import ProductLabelService
        llm, _ = fake_llm("It has 24 g of sugars per 100 g.")
        with TestClient(app) as client:
            old = app.state.product_label_service
            try:
                app.state.product_label_service = ProductLabelService(ProductLabelPipeline(ocr=FakeOCR(TABLE + INGREDIENTS), llm=llm))
                buf = io.BytesIO()
                photo().save(buf, format="JPEG")
                r = client.post("/api/product-label/interpret", files={"image": ("l.jpg", buf.getvalue(), "image/jpeg")})
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertEqual(body["nutrition_per_100"]["sugars_g"], 24.0)
                sid = body["session_id"]
                self.assertEqual(client.get(f"/api/product-label/{sid}/explanation").json()["source"], "llm")
                self.assertEqual(client.post("/api/product-label/interpret").status_code, 422)   # a photo is required
                for gone in ("/api/product-label/x/photos", f"/api/product-label/{sid}/chat"):
                    self.assertIn(client.post(gone).status_code, (404, 405))
            finally:
                app.state.product_label_service = old



class PerServingLabelTests(unittest.TestCase):
    """US-style "Nutrition Facts": values per serving only, serving size as a household measure + weight."""

    def test_serving_size_forms(self):
        from model.product_label.parser import find_serving_size, serving_text
        cases = {"Serving size 1/4 cup shelled (36g)": 36.0, "Seving size /4 cu shelled (36g)": 36.0,
                 "Serving Size 1 cup (240mL)": 240.0, "Serving size: 2 pieces 30 g": 30.0, "Serving size 30 g": 30.0,
                 "Per serving (40 g)": 40.0, "Serving size 2/3 cup (55 g)": 55.0}
        for text, grams in cases.items():
            self.assertEqual((find_serving_size(text) or {}).get("value"), grams, text)
        for text in ("Serving size: 1 piece", "Serving size 3/4 cup", "Servings per container 8", "About 3.5 servings"):
            self.assertIsNone(find_serving_size(text), text)
        self.assertEqual(serving_text("Serving size:1 piece Gelatin, Lactic"), "1 piece")
        self.assertEqual(serving_text("Serving Size 2 Pieces (30g)"), "2 Pieces (30g)")

    def test_split_large_print_energy_row(self):
        from model.product_label.parser import join_split_energy
        self.assertEqual(join_split_energy(["Amount per serving 100", "Calories", "Total Fat 8g"])[:2],
                         ["Amount per serving", "Calories 100 kcal"])
        self.assertIn("Calories 250 kcal", join_split_energy(["Calories", "250", "Total Fat 8g"]))
        self.assertEqual(join_split_energy(["Protein 5g", "Energy", "Fat 3g"]), ["Protein 5g", "Energy", "Fat 3g"])

    def test_us_label_is_converted_with_its_serving_weight(self):
        lines = ["Nutrition Facts", "Serving size 1/2 cup (50g)", "Amount per serving 200", "Calories", "% Daily Value*",
                 "Total Fat 10g 13%", "Saturated Fat 2g 10%", "Sodium 300mg 13%", "Total Carbohydrate 20g 7%",
                 "Total Sugars 5g", "Protein 6g"]
        r, calls = run(lines)
        self.assertEqual(calls.calls, [])                                    # per-serving label: no AI
        self.assertAlmostEqual(r["nutrition_per_100"]["energy_kcal"], 400.0)
        self.assertAlmostEqual(r["nutrition_per_100"]["fat_g"], 20.0)
        self.assertIn(r["nutriscore"]["grade"], list("ABCDE"))

    def test_no_serving_weight_shows_per_serving_values_and_says_why(self):
        lines = ["Nutrition Facts", "Serving size: 1 piece", "Amount per serving", "Calories 12", "Total Carbohydrate 3g 1%",
                 "Sugars 2g", "Protein 1g"]
        r, calls = run(lines)
        self.assertEqual(calls.calls, [])
        self.assertEqual(r["nutrition_per_100"], {})
        self.assertEqual(r["nutrition_per_serving"]["sugars_g"], 2.0)
        self.assertTrue(any("1 piece" in n and "per-100 g" in n for n in r["photo_notes"]))
        self.assertFalse(any("doesn't show the nutrition table" in n for n in r["photo_notes"]))


class MidLineIngredientsTests(unittest.TestCase):
    def test_side_column_ingredients_are_found_but_treated_as_partial(self):
        r, _ = run(TABLE + ["Trans Fat 0g INGREDIENTS: Peanuts, Water, Salt and", "Cholesterol 0mg 0%"])
        self.assertEqual(r["ingredients_text"], "Peanuts, Water, Salt and")
        self.assertEqual(r["allergens"]["status"], "found")
        self.assertIn("part of the ingredients", r["allergens"]["message"])
        r2, _ = run(TABLE + ["Fat 8g Ingrédients : sucre, sel", "Protein 6 g"])
        self.assertEqual(r2["allergens"]["status"], "unknown")             # nothing found in a partial list: never "none"
        self.assertFalse(r2["allergens"]["ingredients_read"])

    def test_contains_needs_a_colon_mid_line(self):
        from model.product_label.pipeline import label_text
        self.assertEqual(label_text(["Total Fat 8g CONTAINS: Milk."])["contains_statement"], "CONTAINS: Milk.")
        self.assertIsNone(label_text(["This pack contains 10 portions"])["contains_statement"])


class ProductNameTests(unittest.TestCase):
    def test_name_cleaned_and_capped(self):
        from app.services.product_label_service import clean_product_name
        self.assertEqual(clean_product_name("  Maggi\n 2-Minute\tNoodles "), "Maggi 2-Minute Noodles")
        self.assertIsNone(clean_product_name("   "))
        self.assertIsNone(clean_product_name(None))
        self.assertEqual(len(clean_product_name("x" * 500)), 100)

    def test_name_reaches_the_ai_facts(self):
        from model.product_label.explain import facts_for_llm
        self.assertEqual(facts_for_llm({"product_name": "Parle-G"})["product_name_typed_by_shopper"], "Parle-G")
        self.assertIsNone(facts_for_llm({})["product_name_typed_by_shopper"])
