"""'What it's for' (ingredient lookup from the 250k Medicines dataset). No AI is called. Run from backend/:
    venv/bin/python -m unittest tests.test_prescription_uses -v
"""
import json
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.prescription import uses  # noqa: E402
from model.prescription.medicine_db import match_all  # noqa: E402
from model.prescription.schemas import MedicineItem, PrescriptionResult  # noqa: E402

HAVE_TABLE = uses.PATH.exists()


def result_for(items, **kw):
    meds, _ = match_all(items)
    return PrescriptionResult(id="rx1", medicines=[MedicineItem(**m) for m in meds], **kw)


@unittest.skipUnless(HAVE_TABLE, "uses_by_ingredient.json not built")
class IngredientLookupTests(unittest.TestCase):
    def test_brand_names_share_their_ingredient(self):
        self.assertEqual(uses.ingredients_of("Domperidone (30mg) + Rabeprazole (20mg)"), ["domperidone", "rabeprazole"])
        self.assertEqual(uses.ingredients_of("Dothiepin (75mg)"), ["dosulepin"])          # synonym -> one key
        self.assertEqual(uses.lookup("Dosulepin (75mg)"), uses.lookup("Dothiepin (25mg)"))
        self.assertEqual(uses.what_for(uses.lookup("Dothiepin")), "Anxiety disorder and depression")

    def test_combination_falls_back_to_each_ingredient(self):
        info = uses.lookup("Zolpidem + Clonazepam")   # not a real combination product
        self.assertIsNotNone(info)
        line = uses.what_for(info)
        self.assertIn("insomnia", line.lower())
        self.assertLessEqual(len(line.split()), uses.MAX_WORDS)

    def test_line_is_short_whole_phrases_from_the_data(self):
        info = uses.lookup("Aspirin + Atorvastatin")
        line = uses.what_for(info)
        self.assertLessEqual(len(line.split()), uses.MAX_WORDS + 4)   # never cut inside a phrase
        for phrase in line.replace(" and ", ",").split(","):
            phrase = phrase.strip().lower()
            self.assertTrue(any(phrase in u.lower() for u in info["uses"]), phrase)

    def test_licence_recorded_in_the_file(self):
        doc = json.loads(uses.PATH.read_text())
        self.assertEqual(doc["_licence"], "CC BY-SA 4.0")
        self.assertIn("Shudhanshu Singh", doc["_source"])


class MissingDescriptionTests(unittest.TestCase):
    def test_unknown_ingredient(self):
        self.assertIsNone(uses.lookup("Notarealium (5mg)"))
        self.assertIsNone(uses.what_for(None))
        self.assertIsNone(uses.lookup(None))



@unittest.skipUnless(HAVE_TABLE, "uses_by_ingredient.json not built")
class VerifyRowTests(unittest.TestCase):
    def test_matched_row_has_what_for_from_database(self):
        m = result_for([{"name": "Dothepin / Prothiaden", "strength": "75mg", "confidence": 0.8}]).medicines[0]
        self.assertEqual(m.match_status, "matched")
        self.assertEqual(m.what_for, "Anxiety disorder and depression")
        self.assertIn("CC BY-SA 4.0", m.what_for_source)

    def test_verify_row_has_what_for_per_candidate_only(self):
        r = result_for([{"name": "Xocalm", "strength": "5mg", "confidence": 0.8}])
        m = r.medicines[0]
        self.assertEqual(m.match_status, "verify")
        self.assertIsNone(m.what_for)                                   # not before the person picks
        self.assertTrue(all(c.what_for for c in m.candidates))          # ready per candidate (shown in the browser)



if __name__ == "__main__":
    unittest.main()
