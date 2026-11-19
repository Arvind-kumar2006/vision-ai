"""Prescription pipeline tests (ADR-005). No AI and no network: the OCR and the AI are fakes.

    venv/bin/python -m unittest tests.test_prescription -v
"""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.common.llm import AI_BUSY_NOTE, LLM, Provider  # noqa: E402
from model.common.redact import lines_for_ai  # noqa: E402
from model.prescription import medicine_db  # noqa: E402
from model.prescription.extract import basic_medicines, ground, grounding_score, validate  # noqa: E402
from model.prescription.pipeline import UNREADABLE, PrescriptionPipeline, mark_checks  # noqa: E402


HAS_DATASET = medicine_db.CSV_PATH.exists()

SYNTH_CSV = """id,name,price,Is_discontinued,manufacturer_name,type,pack_size_label,short_composition1,short_composition2,salt_composition,medicine_desc,side_effects,drug_interactions
1,Dolo 650 Tablet,30,FALSE,Micro Labs Ltd,allopathy,strip of 15 tablets,Paracetamol (650mg),,Paracetamol (650mg),Pain reliever.,Nausea,"{""drug"": [""Warfarin""], ""brand"": [""x""], ""effect"": [""MODERATE""]}"
2,Dolo 500 Tablet,20,FALSE,Micro Labs Ltd,allopathy,strip of 15 tablets,Paracetamol (500mg),,,,,
3,Oldcure 10 Tablet,10,TRUE,Some Pharma Ltd,allopathy,strip of 10 tablets,Oldium (10mg),,,,,
4,Rabemac-DSR Capsule,90,FALSE,Macleods Pharmaceuticals Pvt Ltd,allopathy,strip of 15 capsules,Rabeprazole (20mg),Domperidone (30mg),,,,
"""


def synthetic_index(tmpdir: Path):
    csv_path = tmpdir / "meds.csv"
    csv_path.write_text(SYNTH_CSV)
    return medicine_db.load_index(csv_path, tmpdir / "cache.pkl"), csv_path


class MedicineDbSyntheticTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_discontinued_never_indexed(self):
        idx, _ = synthetic_index(self.tmp)
        self.assertNotIn("Oldcure 10 Tablet", {r["name"] for r in idx.rows})
        r = medicine_db.match_medicine("Oldcure 10", "tab", idx)
        self.assertFalse(any(c["name"].startswith("Oldcure") for c in r["candidates"]))
        self.assertNotEqual(r["match_status"], "matched")

    def test_strength_picks_variant_and_details_have_no_empty_sections(self):
        idx, _ = synthetic_index(self.tmp)
        r = medicine_db.match_medicine("Dolo 650", "1 tab", idx)
        self.assertEqual(r["match_status"], "matched")
        self.assertEqual(r["db_info"]["composition"], "Paracetamol (650mg)")
        self.assertEqual(r["db_info"]["interactions"], [{"drug": "Warfarin", "effect": "MODERATE"}])
        r500 = medicine_db.match_medicine("Dolo 500", "tab", idx)
        self.assertEqual(r500["suggested_name"], "Dolo 500 Tablet")
        for key in ("description", "side_effects", "interactions"):
            self.assertNotIn(key, r500["db_info"])  # absent, not empty

    def test_cache_is_reused_and_rebuilt_when_csv_changes(self):
        _, csv_path = synthetic_index(self.tmp)
        cache = self.tmp / "cache.pkl"
        first = cache.stat().st_mtime_ns
        medicine_db.load_index(csv_path, cache)
        self.assertEqual(cache.stat().st_mtime_ns, first)  # reused
        import os, time as _t
        csv_path.write_text(SYNTH_CSV + "5,Newmed 5 Tablet,1,FALSE,X Ltd,allopathy,strip,Y (5mg),,,,,\n")
        os.utime(csv_path, (_t.time() + 5, _t.time() + 5))
        idx = medicine_db.load_index(csv_path, cache)
        self.assertIn("Newmed 5 Tablet", {r["name"] for r in idx.rows})


@unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
class MedicineDbDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = medicine_db.get_index()

    def match(self, name, dosage=None):
        return medicine_db.match_medicine(name, dosage, self.idx)

    def test_rabemar_offers_rabemac(self):
        r = self.match("Rabemar", "25k Cap")
        self.assertIn(r["match_status"], ("verify", "matched"))
        self.assertTrue(any(c["name"].lower().startswith("rabemac") for c in r["candidates"]),
                        r["candidates"])
        self.assertIsNotNone(r["suggested_name"])  # what was read is not silently replaced
        self.assertTrue(any("Read as 'Rabemar'" in n for n in r["notes"]))

    def test_dolo_650_prefers_650_strength(self):
        r = self.match("Dolo 650", "1 tab")
        self.assertEqual(r["match_status"], "matched")
        self.assertEqual(r["suggested_name"], "Dolo 650 Tablet")
        self.assertIn("650", r["db_info"]["composition"])

    def test_company_names_never_match(self):
        for company in ("Cipla", "Sun Pharma", "Cipla Ltd"):
            r = self.match(company)
            self.assertEqual(r["match_status"], "unmatched", company)
            self.assertEqual(r["candidates"], [], company)
            self.assertTrue(any("pharmaceutical company" in n for n in r["notes"]), company)
        self.assertEqual(self.match("Pan 40", "tab")["match_status"], "matched")  # a medicine

    def test_discontinued_medicines_never_appear(self):
        import csv as _csv
        _csv.field_size_limit(sys.maxsize)
        active, disc = set(), set()
        with open(medicine_db.CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
            for row in _csv.DictReader(f):
                (disc if row["Is_discontinued"].upper() == "TRUE" else active).add(row["name"].strip().lower())
        only_disc = sorted(disc - active)
        self.assertGreater(len(only_disc), 1000)
        names = {r["name"].lower() for r in self.idx.rows}
        self.assertFalse(names & set(only_disc))
        r = self.match(only_disc[0].title())
        self.assertFalse(any(c["name"].lower() in only_disc for c in r["candidates"]))

    def test_known_brands_zincovit_and_electral(self):
        for name, dosage in (("Zincovit", "tab"), ("Electral", "powder")):
            r = self.match(name, dosage)
            self.assertEqual(r["match_status"], "known_brand", name)
            self.assertIsNone(r["db_info"], name)
            self.assertTrue(r["known_brand"]["composition"], name)

    def test_strength_not_written_is_not_invented(self):
        r = self.match("Crocin")
        self.assertEqual(r["match_status"], "matched")
        self.assertNotIn("(", r["db_info"]["composition"])
        self.assertIn("variant_note", r["db_info"])


class CombinedConfidenceTests(unittest.TestCase):
    def test_formula(self):
        c = medicine_db.combined_confidence
        self.assertAlmostEqual(c(0.9, "matched", 100.0, True), 0.9)
        self.assertAlmostEqual(c(0.9, "verify", 80.0, True), round(0.9 * 0.8 * 0.8, 3))
        self.assertAlmostEqual(c(0.9, "unmatched", None, True), 0.45)
        self.assertTrue(0.0 <= c(5.0, "matched", 100.0, True) <= 1.0)

    def test_grounding_cap_applies_even_when_matched(self):
        self.assertLessEqual(medicine_db.combined_confidence(0.95, "matched", 100.0, False), 0.30)


# ---------------------------------------------------------------------------
# Full pipeline with a fake OCR engine and a fake LLM


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class RedactionTests(unittest.TestCase):
    def test_only_medicine_part_without_personal_details_is_sent(self):
        rows = [{"text": t, "confidence": 1.0} for t in [
            "PSYCHIATRIC CLINIC", "Bula Paul Raipur", "NS 72/M Weight 69 kg", "Mobile 78692-36766",
            "1) Tab. Dothepin / Prothiaden 75mg", "Name: someone", "2) Tab. Clonazepam 2mg", "Raipur 492001"]]
        kept, withheld = lines_for_ai(rows)
        self.assertEqual([r["text"] for r in kept], ["1) Tab. Dothepin / Prothiaden 75mg", "2) Tab. Clonazepam 2mg"])
        self.assertEqual(withheld, 6)

    def test_section_ends_at_signature_and_footer_and_doctor_is_never_sent(self):
        for footer in (["Dr. Somename Surname MBBS MD"], ["Regd No 1234"], ["Signature"], ["In emergency call 040 1234567"],
                       ["Plot No 12, Some Street"], ["Tel: 98765 43210"], ["clinic@example.com"]):
            rows = [{"text": t, "confidence": 1.0} for t in
                    ["Dr. Somename Surname", "Mrs Anyname 40 yrs", "1) Tab Brandx 10mg 1-0-1", "x — x — 1",
                     *footer, "2) Tab Brandy 5mg"]]
            sent = " ".join(r["text"] for r in lines_for_ai(rows)[0])
            self.assertIn("Brandx", sent, footer)
            self.assertNotIn("Somename", sent, footer)          # the doctor's name, header or signature
            self.assertNotIn("Anyname", sent, footer)
            self.assertNotIn("Brandy", sent, footer)             # nothing after the footer is sent
        rows = [{"text": t} for t in ["Rx", "1) Tab Brandx", "Dr Otherdoc", "2) Tab Brandy"]]
        self.assertNotIn("Otherdoc", " ".join(r["text"] for r in lines_for_ai(rows)[0]))

    def test_clinical_history_is_never_sent_and_its_numbers_are_dropped(self):
        def row(*boxes):
            return {"text": " ".join(boxes), "confidence": 0.9, "boxes": [{"text": b} for b in boxes]}
        rows = [row("1) Tab Brandx 10mg"), row("1 — x — 1", "Mother expired 3 months back"),
                row("2) Tab Brandy", "k/c/o DM HTN since 5 yrs"), row("Chr Schizophrenia"),
                row("x x 1", "H/O seizures 2 years"), row("Depresion since 2 years")]
        kept, _ = lines_for_ai(rows)
        sent = "\n".join(r["text"] for r in kept)
        for gone in ("expired", "Mother", "HTN", "Schizophrenia", "seizures", "Depresion"):
            self.assertNotIn(gone, sent)
        for number in ("3 months", "5 yrs", "2 years"):
            self.assertNotIn(number, sent)
        self.assertIn("1 — x — 1", sent)                     # the timing on the same row is kept
        self.assertIn("x x 1", sent)
        self.assertIn("2) Tab Brandy", sent)
        self.assertIs(kept[1]["_src"], rows[1])              # the original row is known (for "sent to AI")

    def test_no_medicine_lines_means_nothing_is_sent(self):
        kept, _ = lines_for_ai([{"text": "Bula Paul"}, {"text": "Raipur"}])
        self.assertEqual(kept, [])

    def test_unprefixed_medicine_rows_use_strength_and_schedule_evidence(self):
        rows = [{"text": t, "confidence": 0.8} for t in [
            "Patient Name: Someone", "Dolo 650", "1-0-1", "Clonazepam 0.5 mg", "Dr Clinic"]]
        kept, _ = lines_for_ai(rows)
        sent = [r["text"] for r in kept]
        self.assertEqual(sent, ["Dolo 650", "1-0-1", "Clonazepam 0.5 mg"])

    def test_known_medicine_before_a_prefixed_row_starts_the_section(self):
        rows = [{"text": t, "confidence": 0.8} for t in [
            "Clinic header", "X flagyl 400 101", "Tab Dotin M", "Tab Pan 40"]]
        kept, _ = lines_for_ai(rows)
        self.assertEqual([r["text"] for r in kept], ["X flagyl 400 101", "Tab Dotin M", "Tab Pan 40"])

    def test_safety_net_recovers_strong_unprefixed_medicine_name(self):
        from model.prescription.extract import strong_entries
        rows = [{"text": "X flagyl 400 101", "confidence": 0.6}]
        self.assertEqual(strong_entries(rows)[0]["name"].lower(), "flagyl")

    def test_layout_candidates_keep_unreadable_medicine_rows_for_verification(self):
        from model.prescription.extract import supplement_layout_candidates
        rows = [{"text": t, "confidence": 0.5} for t in [
            "T-mimic dupp 0.31-0.31-0.3", "Anthatecind dupy 0.9p", "Advent dup 0.81 0.8 3day",
            "Nanoden meraldeys 0.8-1", "Hirzole cream twice daily"]]
        added = supplement_layout_candidates([], rows)
        self.assertEqual(len(added), 5)
        self.assertTrue(all("name" in m and "name" in m["checks"] for m in added))

    def test_instruction_rows_are_not_promoted_to_medicines(self):
        from model.prescription.extract import supplement_layout_candidates
        rows = [{"text": t, "confidence": 0.8} for t in [
            "after meals 1-0-1 x 5 days", "massage 1-0-1 x 1 week", "Tab Augmentin 625mg"]]
        added = supplement_layout_candidates([], rows)
        self.assertEqual([m["name"] for m in added], ["Augmentin"])


# ---------------------------------------------------------------------------
# AI output: validation, grounding, check markers; basic parsing when the AI is busy
# ---------------------------------------------------------------------------
class ExtractTests(unittest.TestCase):
    def test_validate_keeps_name_alternatives_strength_and_form_only(self):
        out = validate({"medicines": [{"name": "Dothepin", "alternatives": "Prothiaden", "strength": "75 mg",
                                       "form": "tablet", "frequency": "① रात", "duration": "8 months",
                                       "confidence": 0.9}, {"name": ""}, "junk"],
                        "already_taking": ["Ecosprin AV"]})
        self.assertEqual(len(out["medicines"]), 1)
        m = out["medicines"][0]
        self.assertEqual((m["alternatives"], m["strength"], m["form"]), (["Prothiaden"], "75mg", "tablet"))
        for gone in ("dosage", "frequency", "duration", "instructions"):
            self.assertNotIn(gone, m)
        self.assertNotIn("followup_date", out)
        self.assertIsNone(validate({"medicines": [{"name": "Dolo", "strength": "1 tab"}]})["medicines"][0]["strength"])
        self.assertEqual(out["already_taking"], ["Ecosprin AV"])
        both = validate({"medicines": [{"name": "Ecosprin AV"}, {"name": "Dolo"}], "already_taking": ["Ecosprin AV"]})
        self.assertEqual([m["name"] for m in both["medicines"]], ["Dolo"])   # already taking is not prescribed

    def test_ocr_fixups_and_ai_safety_net(self):
        from model.prescription.extract import fix_ocr
        self.assertEqual(fix_ocr("on7ab, Ecospuin Av"), "On Tab, Ecospuin Av")
        self.assertEqual(fix_ocr("2118|25 31 9ab zocalm smg So5"), "2118|25 31 Tab zocalm 5mg SOS")
        self.assertEqual(fix_ocr("Tablet 5 days, Stop"), "Tablet 5 days, Stop")

    def test_grounding_flags_invented_names(self):
        meds = [{"name": "Zincovit", "confidence": 0.9}, {"name": "Augmentin", "confidence": 0.9}]
        notes = ground(meds, "1) Tab Zincovet 1-0-1")
        self.assertEqual([m["in_ocr_text"] for m in meds], [True, False])
        self.assertLessEqual(meds[1]["confidence"], 0.3)
        self.assertEqual(len(notes), 1)
        self.assertGreater(grounding_score("Clonazepam", "Tob Clonszepam"), 0.8)

    def test_check_markers_low_confidence_or_number_not_in_text(self):
        rows = [{"text": "1) Tab Dothepin 75mg ① रात", "confidence": 0.95}, {"text": "2) Tab Zocalm smg SOS", "confidence": 0.6}]
        text = "\n".join(r["text"] for r in rows)
        a = {"name": "Dothepin", "strength": "75mg"}
        mark_checks(a, rows, text)
        self.assertEqual(a["checks"], [])
        b = {"name": "Zocalm", "strength": "5mg"}
        mark_checks(b, rows, text)
        self.assertEqual(b["checks"], ["strength"])                        # low-confidence line, number not read
        c = {"name": "Dothepin", "strength": "50mg"}
        mark_checks(c, rows, text)
        self.assertEqual(c["checks"], ["strength"])                        # 50 isn't in the text

    @unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
    def test_basic_medicines_use_the_dataset(self):
        from model.prescription.extract import fix_ocr   # the pipeline applies the OCR fix-ups first
        rows = [{"text": fix_ocr(t), "confidence": 0.9} for t in
                ["On Tab Ecosprin AV", "() Tab. dothepin /Prothiaden", "X 2) Tob, Clonszepam( or", "8months",
                 "2118|25 31 9ab zocalm smg So5"]]
        meds, taking = basic_medicines(rows)
        self.assertEqual([m["name"].lower() for m in meds], ["dothepin", "clonszepam", "zocalm"])
        self.assertEqual(taking, ["Ecosprin"])


# ---------------------------------------------------------------------------
# Step A: the safety net only adds names that open a medicine-style line, with a strong match, not duplicates
# ---------------------------------------------------------------------------
class EntryNameTests(unittest.TestCase):
    def test_medicine_style_lines_in_many_layouts(self):
        from model.prescription.extract import entry_name
        cases = {"① Tab Brandx Plus 50mg": "Brandx Plus", "Tab. Brandx 10 mg 1-0-1": "Brandx",
                 "(2) Syp. Brandy DS 5ml": "Brandy DS", "Cap: Novaxa SR 1-0-1": "Novaxa SR", "3) Brandx Forte": "Brandx Forte",
                 "12/3/24 4) Tab Brandq 5mg": "Brandq", "✓ 1. Inj Brandz 1 amp": "Brandz", "tabBrandx (generic) 2mg": "Brandx",
                 "Capsules  Omeprax 20": "Omeprax", "⑤ Brandw 0.5mg": "Brandw", "1- Brandr 25mg": "Brandr"}
        for text, name in cases.items():
            self.assertEqual(entry_name(text), name, text)

    def test_abbreviated_forms_and_bullets(self):
        from model.prescription.extract import entry_name
        from model.common.redact import MEDICINE_START
        cases = {"- T. Brandx 1 Po TDS": "Brandx", "[-T. Brandy 40 1 Po OD": "Brandy", "—T. Brandz XL 10mg Po BD": "Brandz",
                 "• C. Brandq DSR 1 Po OD": "Brandq DSR", "C: Brandw 20": "Brandw", "* Tab Brandv": "Brandv",
                 "> T.Brandu 5mg": "Brandu", "(T. Brandt)": "Brandt"}
        for text, name in cases.items():
            self.assertEqual(entry_name(text), name, text)
            self.assertTrue(MEDICINE_START.search(text), text)
        for text in ("T.C. count 9000", "Temp.: 98 F", "Dr. Somename", "C/O fever", "Total 3 tabs"):
            self.assertIsNone(entry_name(text), text)

    def test_no_medicine_section_gives_a_clear_message(self):
        llm, calls = fake_llm("{}")
        r = PrescriptionPipeline(ocr=FakeOCR(["SOME HOSPITAL", "Temp.: Pulse:", "Fever since 2 days"]), llm=llm).run(page())
        self.assertEqual(r.status, "no_medicines")
        self.assertIn("no medicine lines were recognised", r.message)
        self.assertEqual(calls.calls, [])

    def test_other_lines_are_never_medicine_entries(self):
        from model.prescription.extract import entry_name
        for text in ("Call me in between if any problem", "Dr. Somename Surname", "Morning      Night", "x — x — 1",
                     "Review after 2 weeks", "On Tab Brandx", "Tablet 5 days", "Mother expired 3 months back",
                     "Chr Schizophrenia DM HTN", "रात", "Continue other medicines"):
            self.assertIsNone(entry_name(text), text)

    @unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
    def test_safety_net_needs_a_strong_match_and_skips_duplicates(self):
        from model.prescription.extract import missed_by_ai
        rows = [{"text": t, "confidence": 0.9} for t in [
            "1) Tab Dolo 650 1-0-1",           # strong match, not listed by the AI -> added
            "2) Tab Qwertyx 10mg",              # no strong match -> never added
            "3) Tab Pantop 40",                 # the AI listed "Pantop-40": same brand -> duplicate
            "4) Tab Crocin 500",                # the AI listed "Paracetamol": same ingredient as Dolo/Crocin -> duplicate
            "Call me if any problem"]]          # not a medicine-style line
        ai = [{"name": "Pantop-40"}, {"name": "Paracetamol"}]
        added = [m["name"] for m in missed_by_ai(ai, rows)]
        self.assertEqual(added, [])            # Dolo is paracetamol too -> duplicate of the AI's "Paracetamol"
        self.assertEqual([m["name"] for m in missed_by_ai([{"name": "Pantop-40"}], rows[:2])], ["Dolo"])


@unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
class VariantSuffixTests(unittest.TestCase):
    def test_written_form_does_not_hide_the_variant(self):
        from model.prescription.medicine_db import match_all
        out, _ = match_all([{"name": "Sizodon Plus", "form": "tablet", "confidence": 0.9}])
        self.assertIn("plus", (out[0]["candidates"][0]["name"]).lower())

    def test_suffix_mismatch_is_verify_never_matched(self):
        from model.prescription.medicine_db import check_variant
        base = {"match_status": "matched", "candidates": [{"name": "Brandx 1 Tablet"}], "notes": []}
        self.assertEqual(check_variant(base, "Brandx Plus")["match_status"], "verify")
        self.assertEqual(check_variant(base, "Brandx")["match_status"], "matched")
        sr = {"match_status": "matched", "candidates": [{"name": "Brandy SR 100 Tablet"}], "notes": []}
        self.assertEqual(check_variant(sr, "Brandy")["match_status"], "verify")        # SR written nowhere
        self.assertEqual(check_variant(sr, "brandy sr")["match_status"], "matched")    # case doesn't matter
        ds = {"match_status": "known_brand", "known_brand": {"name": "Brandz DS"}, "notes": []}
        self.assertEqual(check_variant(ds, "Brandz Forte")["match_status"], "verify")
        self.assertIn("different products", check_variant(ds, "Brandz Forte")["notes"][-1])


# ---------------------------------------------------------------------------
# Steps D + E: one block per medicine, margin notes separately, slot grids
# ---------------------------------------------------------------------------
def page_row(*boxes, y=0, h=40, conf=0.9):
    """A row of text boxes: (text, x0, x1) in pixels on a 1000 px wide page."""
    bs = [{"text": t, "box": [x0, y, x1, y + h]} for t, x0, x1 in boxes]
    return {"text": " ".join(t for t, _, _ in boxes), "confidence": conf, "boxes": bs,
            "box": [min(b["box"][0] for b in bs), y, max(b["box"][2] for b in bs), y + h]}


class LayoutTests(unittest.TestCase):
    def test_blocks_and_margin_notes_in_different_layouts(self):
        from model.prescription.extract import build_blocks, format_blocks
        rows = [page_row(("On Tab Brandq", 100, 380), y=0),
                page_row(("① Tab Brandx Plus", 100, 500), ("Note in margin", 780, 980), y=60),
                page_row(("1 — x — 1", 260, 520), y=110),
                page_row(("Morning", 250, 360), ("Night", 470, 560), y=160),
                page_row(("2) Cap Brandy 20mg", 120, 560), y=220),
                page_row(("x x 1", 300, 480), ("6 months", 700, 860), y=270),    # far right, big gap -> margin
                page_row(("3. Syp Brandz 5ml", 110, 520), ("1-0-1 after food", 560, 820), y=330),   # small gap: same line
                page_row(("aside", 850, 940), y=390)]                              # alone, right part -> margin
        layout = build_blocks(rows, 1000)
        self.assertEqual([r["text"] for r in layout["before"]], ["On Tab Brandq"])
        self.assertEqual([[r["text"] for r in b] for b in layout["blocks"]],
                         [["① Tab Brandx Plus", "1 — x — 1", "Morning Night"], ["2) Cap Brandy 20mg", "x x 1"],
                          ["3. Syp Brandz 5ml 1-0-1 after food"]])
        self.assertEqual([(n["text"], n["beside"]) for n in layout["margin"]],
                         [("Note in margin", [1]), ("6 months", [2]), ("aside", [3])])
        text = format_blocks(layout)
        for part in ("OTHER LINES:", "MEDICINE 1:", "MEDICINE 3:", "MARGIN NOTES:", "6 months [0.90] (beside medicine 2)"):
            self.assertIn(part, text)

    def test_thresholds_are_relative_to_the_page(self):
        from model.prescription.extract import build_blocks
        small = [page_row(("1) Tab Brandx", 50, 250), ("6 months", 350, 430), y=0, h=20)]   # 500 px page
        big = [page_row(("1) Tab Brandx", 200, 1000), ("6 months", 1400, 1720), y=0, h=80)]  # 2000 px page
        self.assertEqual(build_blocks(small, 500)["margin"][0]["text"], "6 months")
        self.assertEqual(build_blocks(big, 2000)["margin"][0]["text"], "6 months")
        no_boxes = [{"text": "1) Tab Brandx 6 months", "confidence": 0.9}]                 # no geometry: one line
        self.assertEqual(build_blocks(no_boxes, 1000)["margin"], [])


class SlotGridTests(unittest.TestCase):
    def test_grid_spellings(self):
        from model.prescription.extract import slot_code
        cases = {"1 — x — 1": "1-0-1", "x x 1": "0-0-1", "×—×—1": "0-0-1", "1-0-1": "1-0-1", "① 0 ①": "1-0-1",
                 "½ - 0 - 1": "½-0-1", "X–1–X": "0-1-0", "o o 1 night": "0-0-1", "2_0_2": "2-0-2", "x——x—1nisw": "0-0-1",
                 "1 . 1 . 1": "1-1-1", "Tab Brandx 650 1 0 1": "1-0-1"}
        for text, code in cases.items():
            self.assertEqual(slot_code(text), code, text)

    def test_things_that_are_not_grids(self):
        from model.prescription.extract import slot_code
        for text in ("21/8/25", "12-3-24", "0.5mg", "x x", "1 x", "0-0-0", "Tab Brandx 10mg", "3 months", "Rs 1 2 34"):
            self.assertIsNone(slot_code(text), text)

class DurationRowTests(unittest.TestCase):
    """A row that is only a duration is never a medicine (durations are not extracted any more)."""

    def test_pure_duration_notes(self):
        from model.prescription.extract import is_pure_duration
        for text in ("6 months", "(Six) 6 mths", "10 days", "x 5 days", "3mths", "8months", "one month", "2 wk", "1 yr"):
            self.assertTrue(is_pure_duration(text), text)
        for text in ("Mother expired 3 months back", "review after 2 weeks", "since 5 yrs", "BP 140/90", "4967/2013",
                     "Brother died 2 years ago", "sugar 180", "", "night"):
            self.assertFalse(is_pure_duration(text), text)

class ConfidenceLevelTests(unittest.TestCase):
    def item(self, **kw):
        base = {"match_status": "matched", "match_score": 100.0, "strength": "10mg", "in_ocr_text": True,
                "ocr_confidence": 0.95, "candidates": [{"name": "Brandx 10mg Tablet"}], "variant_mismatch": False}
        return {**base, **kw}

    def test_levels(self):
        from model.prescription.medicine_db import confidence_level
        level = lambda **kw: confidence_level(self.item(**kw))[0]
        self.assertEqual(level(), "high")
        self.assertEqual(level(strength=None), "high")                                    # no strength written: neutral
        self.assertEqual(level(ocr_confidence=0.75), "medium")                            # handwriting only fair
        self.assertEqual(level(match_score=90.0), "medium")
        self.assertEqual(level(match_status="verify", match_score=85.0), "medium")
        self.assertEqual(level(strength="20mg"), "low")                                   # strength not in the product
        self.assertEqual(level(variant_mismatch=True, match_status="verify"), "low")      # Plus vs plain
        self.assertEqual(level(ocr_confidence=0.4), "low")
        self.assertEqual(level(in_ocr_text=False), "low")
        self.assertEqual(level(match_status="unmatched", match_score=None, candidates=[]), "low")
        self.assertEqual(level(match_status="known_brand", candidates=[], known_brand={"name": "Brandk"},
                               strength=None), "high")

    def test_the_four_strength_cases(self):
        from model.prescription.medicine_db import confidence_level
        level = lambda **kw: confidence_level(self.item(**kw))
        strength = lambda **kw: next(f for f in level(**kw)[1] if f["factor"] == "Strength")
        # 1. not written, one strength (or none) in the dataset: neutral, never lowers it
        self.assertEqual(strength(strength=None, strength_variants=1)["effect"], "neutral")
        self.assertEqual(level(strength=None, strength_variants=1)[0], "high")
        self.assertEqual(level(strength=None, strength_variants=0)[0], "high")
        # 2. not written, several strengths: ambiguous; 2-3 -> medium, 4+ -> low
        self.assertEqual(level(strength=None, strength_variants=2)[0], "medium")
        self.assertEqual(level(strength=None, strength_variants=4)[0], "low")
        # 3. written and in the product: good
        self.assertEqual(strength()["effect"], "good")
        # 4. written and not in the product (or in no product of the brand): bad
        self.assertEqual(level(strength="20mg")[0], "low")
        self.assertEqual(level(written_strength_in_brand=False)[0], "low")

    def test_high_without_strength_needs_an_exact_name_and_no_close_rival(self):
        from model.prescription.medicine_db import confidence_level
        level = lambda **kw: confidence_level(self.item(strength=None, strength_variants=1, **kw))[0]
        self.assertEqual(level(), "high")                                   # Hexigel: 100/100, clear, one strength
        self.assertEqual(level(match_score=96.0), "medium")                 # strength can't confirm a near spelling
        self.assertEqual(level(ocr_confidence=0.75), "medium")              # handwriting only fair
        self.assertEqual(level(in_ocr_text=False), "low")                   # not grounded in the text read
        self.assertEqual(level(rival_score=97.0), "low")                    # a different brand nearly as close
        self.assertEqual(level(rival_score=80.0), "high")                   # a distant one doesn't count

    def test_factors_are_words_never_percentages(self):
        from model.prescription.medicine_db import confidence_level
        _, factors = confidence_level(self.item(ocr_confidence=0.72))
        self.assertEqual([f["factor"] for f in factors], ["Medicine list", "Name similarity", "Other similar names", "Strength",
                                                          "Variant (Plus, SR, DS ...)", "Handwriting read", "Name in the text read"])
        self.assertNotIn("%", " ".join(str(f["value"]) for f in factors))
        self.assertIn("fair", [f["value"] for f in factors])


# ---------------------------------------------------------------------------
# Pipeline with a fake OCR and a fake AI
# ---------------------------------------------------------------------------
P5_LINES = ["PSYCHIATRIC CLINIC", "रजि. क्र. 4967/2013", "Bula Paul Raipur", "69 Kgs", "On Tab Ecosprin AV",
            "1) Tab. Dothepin / Prothiaden 75mg", "① रात", "2) Tab. Clonazepam / Lonazep 2mg ① रात", "8 months",
            "21/8/25 3) Tab Zocalm 5mg SOS"]
AI_JSON = """{"medicines": [
  {"name": "Dothepin", "alternatives": ["Prothiaden"], "strength": "75mg", "form": "tablet", "confidence": 0.9},
  {"name": "Clonazepam", "alternatives": ["Lonazep"], "strength": "2mg", "form": "tablet", "confidence": 0.9},
  {"name": "Zocalm", "strength": "5mg", "confidence": 0.8}],
 "already_taking": ["Ecosprin AV"]}"""


class FakeOCR:
    name, available, error = "fake OCR", True, None

    def __init__(self, lines, conf=0.95):
        self.lines, self.conf = lines, conf

    def read(self, bgr, hindi=None, profile=None, preprocessing=None):
        rows = [{"text": t, "confidence": self.conf, "box": [0, i, 1, i + 1]} for i, t in enumerate(self.lines)]
        return {"ok": bool(rows), "rows": rows, "lines": list(self.lines), "ms": 1, "error": None if rows else "no text found"}


class FakeCompletions:
    def __init__(self, text, error=None):
        self.text, self.error, self.calls = text, error, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=self.text))])


def fake_llm(text="{}", error=None):
    c = FakeCompletions(text, error)
    return LLM.with_client(mock.Mock(chat=mock.Mock(completions=c))), c


def page():
    return Image.new("RGB", (400, 600), "white")


@unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
class PipelineTests(unittest.TestCase):
    def test_full_result(self):
        llm, calls = fake_llm(AI_JSON)
        r = PrescriptionPipeline(ocr=FakeOCR(P5_LINES), llm=llm).run(page())
        self.assertEqual(r.status, "ok")
        self.assertEqual(len(calls.calls), 1)                                  # one AI call per scan
        sent = calls.calls[0]["messages"][1]["content"]
        for private in ("Bula Paul", "PSYCHIATRIC", "4967", "69 Kgs"):
            self.assertNotIn(private, sent)                                    # header/patient never sent
        self.assertIn("Dothepin", sent)
        meds = {m.name: m for m in r.medicines}
        self.assertEqual(set(meds), {"Dothepin", "Clonazepam", "Zocalm"})
        self.assertEqual({n: m.strength for n, m in meds.items()}, {"Dothepin": "75mg", "Clonazepam": "2mg", "Zocalm": "5mg"})
        self.assertEqual(meds["Dothepin"].match_status, "matched")
        self.assertEqual(meds["Dothepin"].what_for, "Anxiety disorder and depression")
        self.assertEqual(r.already_taking, ["Ecosprin AV"])
        body = r.model_dump()
        for gone in ("date", "doctor_reg_no", "followup_date", "general_instructions"):
            self.assertNotIn(gone, body)
        for gone in ("dosage", "frequency", "duration", "instructions"):
            self.assertNotIn(gone, body["medicines"][0])
        self.assertEqual(r.ai["provider"], "test")
        self.assertFalse(any(l.sent_to_ai for l in r.ocr_lines if "Bula" in l.text))

    def test_a_medicine_the_ai_left_out_is_still_listed_and_flagged(self):
        no_zocalm = AI_JSON.replace(""",
  {"name": "Zocalm", "strength": "5mg", "confidence": 0.8}]""", "]")
        llm, _ = fake_llm(no_zocalm)
        r = PrescriptionPipeline(ocr=FakeOCR(P5_LINES), llm=llm).run(page())
        self.assertIn("Zocalm", [m.name for m in r.medicines])
        self.assertTrue(any("didn't list it" in n for n in r.review_notes))
        self.assertNotIn("Ecosprin", [m.name for m in r.medicines])   # already taking is not added back

    def test_the_ai_is_asked_for_identification_only(self):
        lines = ["Rx", "1) Tab Dolo 650", "x — x — 1", "Night"]
        llm, calls = fake_llm('{"medicines": [{"block": 1, "name": "Dolo", "strength": "650mg", "confidence": 0.9}]}')
        r = PrescriptionPipeline(ocr=FakeOCR(lines), llm=llm).run(page())
        self.assertEqual((r.medicines[0].name, r.medicines[0].strength), ("Dolo", "650mg"))
        system, sent = calls.calls[0]["messages"][0]["content"], calls.calls[0]["messages"][1]["content"]
        self.assertIn("MEDICINE 1:", sent)
        for gone in ('"frequency"', '"duration"', '"dosage"', "followup_date", "DURATION EVIDENCE"):
            self.assertNotIn(gone, system + sent)

    def test_no_text_means_unreadable(self):
        llm, calls = fake_llm()
        r = PrescriptionPipeline(ocr=FakeOCR([]), llm=llm).run(page())
        self.assertEqual((r.status, r.message), ("unreadable", UNREADABLE))
        self.assertEqual(r.medicines, [])
        self.assertEqual(calls.calls, [])

    def test_both_ai_providers_busy_gives_dataset_matched_names(self):
        first, second = FakeCompletions("", RuntimeError("429")), FakeCompletions("", RuntimeError("timeout"))
        llm = LLM([Provider("OpenRouter", "m:free", mock.Mock(chat=mock.Mock(completions=first))),
                   Provider("Groq", "q", mock.Mock(chat=mock.Mock(completions=second)))])
        r = PrescriptionPipeline(ocr=FakeOCR(P5_LINES), llm=llm).run(page())
        self.assertEqual((len(first.calls), len(second.calls)), (1, 1))       # each tried once
        self.assertEqual((r.status, r.message), ("ai_busy", AI_BUSY_NOTE))
        self.assertIn(AI_BUSY_NOTE, r.review_notes)
        self.assertEqual({m.name.lower() for m in r.medicines} >= {"dothepin", "zocalm"}, True)
        self.assertEqual(r.already_taking, ["Ecosprin"])
        self.assertIsNone(r.ai)

    def test_ai_answer_not_json_is_treated_as_busy(self):
        llm, _ = fake_llm("Sorry, I can't help with that.")
        r = PrescriptionPipeline(ocr=FakeOCR(P5_LINES), llm=llm).run(page())
        self.assertEqual(r.status, "ai_busy")

    def test_low_confidence_lines_get_check_markers(self):
        llm, _ = fake_llm(AI_JSON)
        r = PrescriptionPipeline(ocr=FakeOCR(P5_LINES, conf=0.6), llm=llm).run(page())
        self.assertTrue(all("strength" in m.checks for m in r.medicines))
        self.assertTrue(any("check" in n for n in r.review_notes))


class FreeOnlyProviderTests(unittest.TestCase):
    def test_paid_openrouter_model_is_never_used(self):
        from model.common.llm import providers_from_env
        ps = providers_from_env({"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "openai/gpt-5", "GROQ_API_KEY": "g"})
        self.assertEqual([p.name for p in ps], ["Groq"])
        ps = providers_from_env({"OPENROUTER_API_KEY": "k", "GROQ_API_KEY": "g"})
        self.assertEqual([p.name for p in ps], ["OpenRouter", "Groq"])           # OpenRouter first, Groq backup
        self.assertTrue(ps[0].model.endswith(":free"))
        self.assertNotIn("k", repr(ps[0]))                                       # keys never in repr / logs
        self.assertEqual(providers_from_env({}), [])


class PrescriptionAPITests(unittest.TestCase):
    def test_routes(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.prescription_service import PrescriptionService
        llm, _ = fake_llm(AI_JSON)
        with TestClient(app) as client:
            old = app.state.prescription_service
            try:
                app.state.prescription_service = PrescriptionService(PrescriptionPipeline(ocr=FakeOCR(P5_LINES), llm=llm))
                buf = io.BytesIO()
                page().save(buf, format="JPEG")
                r = client.post("/api/prescription", files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")})
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertEqual(len(body["medicines"]), 3)
                self.assertNotIn("patient_name", body)
                for gone in (f"/api/prescription/{body['id']}/chat", f"/api/prescription/{body['id']}/values", f"/api/prescription/{body['id']}/fill-gaps"):
                    self.assertIn(client.post(gone, json={}).status_code, (404, 405))
                self.assertEqual(client.post("/api/prescription", files={"file": ("x.txt", b"hi", "text/plain")}).status_code, 415)
            finally:
                app.state.prescription_service = old


# ---------------------------------------------------------------------------
# Fixes from the 24-photo check of data/medicines/Images (2026-11-14)
# ---------------------------------------------------------------------------
class RealPhotoRegressionTests(unittest.TestCase):
    def sent(self, lines):
        kept, _ = lines_for_ai([{"text": t, "confidence": 0.9} for t in lines])
        return [r["text"] for r in kept]

    def test_pt_variant_is_not_a_patient_detail(self):
        self.assertIn("Tab. Zerodol PT", self.sent(["Rx", "Tab. Zerodol PT", "1-0-1 x5days"]))
        self.assertNotIn("Pt: Ramesh", self.sent(["Rx", "Pt: Ramesh", "Tab Dolo 650"]))

    def test_letterhead_above_the_patient_line_does_not_start_the_medicine_section(self):
        sent = self.sent(["HOSPITAL", "Veer Savarkar Marg, Mahim, Mumbai - 400 016", "Name. Rahul 29m",
                          "Rx", "- T. Azee 500mg x 7 days"])
        self.assertEqual(sent[0], "Rx")
        self.assertNotIn("Veer Savarkar Marg, Mahim, Mumbai - 400 016", sent)

    def test_allergy_list_is_never_sent_even_misspelt(self):
        sent = self.sent(["Rx", "Patient is allengic to : onset, Naproxen", "amikacin, Levofloxacin,",
                          "Azomax. Tramal", "Inj Risek 40mg IV SOS"])
        self.assertEqual(sent, ["Rx", "Inj Risek 40mg IV SOS"])
        self.assertIn("Tab Cetirizine 10mg for allergy", self.sent(["Rx", "Tab Cetirizine 10mg for allergy"]))

    def test_consultant_signature_line_ends_the_section(self):
        sent = self.sent(["Rx", "T. Ultracet BD x 5 days", "Consultant Tanmoy Kumar Mandal Regn No: WBMC-63430"])
        self.assertEqual(sent, ["Rx", "T. Ultracet BD x 5 days"])

    def test_glued_duration_in_layout_candidates_does_not_crash(self):
        from model.prescription.extract import layout_candidates
        rows = [{"text": t, "confidence": 0.8} for t in ["Syp Sumol DS 4ml 3d", "Syp Mucolite LS 2.5ml tds 3d"]]
        self.assertEqual([c["name"] for c in layout_candidates(rows)], ["Sumol DS", "Mucolite LS"])

    def test_layout_candidates_need_dosing_evidence_not_any_number(self):
        from model.prescription.extract import layout_candidates
        rows = [{"text": t, "confidence": 0.9} for t in ["V (Cervical Cancer) 10-45yrs", "Code No: 0457/2009",
                                                         "Advent drops 0.8ml"]]
        self.assertEqual([c["name"] for c in layout_candidates(rows)], ["Advent"])

    def test_schedule_row_under_its_medicine_is_a_continuation(self):
        from model.prescription.extract import build_blocks
        rows = [{"text": t, "confidence": 0.9} for t in [
            "Tab. Augmentin 625mg", "meals 1-0-1x5days", "Tab. Enzoflam 1-0-1x5days", "Adv: Hexigel gum paint",
            "massage 1-0-1 x1week", "Colgate Plax mouth wash - 1week."]]
        blocks = build_blocks(rows)["blocks"]
        self.assertEqual([b[0]["text"] for b in blocks], ["Tab. Augmentin 625mg", "Tab. Enzoflam 1-0-1x5days",
                                                          "Adv: Hexigel gum paint", "Colgate Plax mouth wash - 1week."])

    def test_orphan_schedule_marks_a_missed_medicine_line(self):
        from model.prescription.extract import missed_line_gaps, schedule_only
        self.assertTrue(schedule_only("1-0-0x5days"))
        self.assertFalse(schedule_only("Tab. Enzoflam 1-0-1x5days"))
        row = lambda t, y: {"text": t, "boxes": [{"text": t, "box": [100, y, 600, y + 50]}]}
        rows = [row("Tab. Enzoflam 1-0-1x5days", 100), row("1-0-0x5days", 230)]
        self.assertEqual(missed_line_gaps(rows), [(150, 230)])
        self.assertEqual(missed_line_gaps([row("Tab. Zerodol PT", 100), row("1-0-1 x5days", 230)]), [])

    def test_iweek_is_read_as_one_week(self):
        from model.prescription.extract import fix_ocr
        self.assertEqual(fix_ocr("Colgate Plax mouth wash -Iweek."), "Colgate Plax mouth wash -1week.")
        self.assertEqual(fix_ocr("Ivermectin 12mg"), "Ivermectin 12mg")

    def test_missed_line_is_read_again_from_its_strip(self):
        import numpy as np

        class StripOCR(FakeOCR):
            def __init__(self):
                self.calls = 0

            def read(self, bgr, hindi=None, profile=None, preprocessing=None):
                self.calls += 1
                if self.calls > 1:   # the strip
                    b = {"text": "Tab. PanD 40mg", "raw": "Tab. PanD 40mg", "confidence": 0.94, "box": [40, 30, 400, 70]}
                    return {"ok": True, "rows": [b], "boxes": [b], "ms": 1}
                boxes = [{"text": t, "raw": t, "confidence": 0.95, "box": [100, y, 600, y + 50]} for t, y in
                         [("Rx", 0), ("Tab. Enzoflam 1-0-1x5days", 100), ("1-0-0x5days", 230)]]
                return {"ok": True, "rows": boxes, "boxes": boxes, "ms": 1, "rotated": 0}

        pipe = PrescriptionPipeline(ocr=StripOCR(), llm=LLM([]))
        bgr = np.full((400, 800, 3), 255, np.uint8)
        from model.prescription.layout import reconstruct_rows
        rows, _ = reconstruct_rows(pipe.ocr.read(bgr)["boxes"], 800)
        pipe.ocr.calls = 1
        found = pipe._recover_missed_lines(bgr, {"boxes": rows and pipe.ocr.read(bgr)["boxes"], "rotated": 0}, rows)
        self.assertEqual([b["text"] for b in found], ["Tab. PanD 40mg"])
        self.assertEqual(found[0]["box"][1], 30 - pipe.STRIP_PAD + 150)


@unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
class RealPhotoDatasetTests(unittest.TestCase):
    def test_generic_names_match_their_ingredient_not_a_lookalike_brand(self):
        for written, ingredient in [("Remdesivir", "Remdesivir"), ("HYDROXYCHLOROQUINE 400mg", "Hydroxychloroquine"),
                                    ("Liposomal Amphotericin B", "Liposomal Amphotericin B"),
                                    ("Isavuconazole", "Isavuconazole"), ("REMDESINVIR", "Remdesivir")]:
            r = medicine_db.match_medicine(written)
            self.assertEqual(r["match_status"], "matched", written)
            self.assertEqual(r["db_info"]["composition"], ingredient, written)
        self.assertEqual(medicine_db.match_medicine("Augmentin")["db_info"]["composition"],
                         "Amoxycillin + Clavulanic Acid")   # a brand still matches as the brand

    def test_ordinary_words_are_not_recovered_as_medicines(self):
        from model.prescription.extract import strong_entries
        rows = [{"text": t, "confidence": 0.9} for t in ["Medicine Prescribed : Plan", "Bone Kung Liver",
                                                         "Further plan of treatment", "relived by Bugen"]]
        self.assertEqual(strong_entries(rows), [])

    def test_strength_tells_a_medicine_from_a_company(self):
        from model.prescription.extract import strong_entries
        rows = [{"text": "- T. Pan 40mg", "confidence": 0.9}]
        self.assertEqual([e["name"] for e in strong_entries(rows)], ["Pan"])


class MergedRowTests(unittest.TestCase):
    def test_two_entries_on_one_row_are_split(self):
        from model.prescription.extract import split_merged_entries
        split = lambda t: [r["text"] for r in split_merged_entries([{"text": t, "confidence": 0.9}])]
        self.assertEqual(split("T. Rantac 1 tab OD BBF . T. Diclo 1 tab BD"), ["T. Rantac 1 tab OD BBF", "T. Diclo 1 tab BD"])
        self.assertEqual(split("- T. Ondem (4) BD D2-D4 —T. Ultracet BD x 5 days"),
                         ["- T. Ondem (4) BD D2-D4", "—T. Ultracet BD x 5 days"])
        self.assertEqual(split("after meals Tab. Augmentin 625mg"), ["after meals Tab. Augmentin 625mg"])

    def test_capsule_letter_without_its_dot_starts_the_section(self):
        kept, _ = lines_for_ai([{"text": t, "confidence": 0.9} for t in ["Age: 30 yrs", "C Neurokind plus",
                                                                          "Oro T gargles 2 tsp"]])
        self.assertEqual([r["text"] for r in kept], ["C Neurokind plus", "Oro T gargles 2 tsp"])

    @unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
    def test_generic_name_split_by_the_ocr_is_recovered_whole(self):
        from model.prescription.extract import strong_entries
        found = strong_entries([{"text": "$$$ HYDROXY CHLORO QUINE 400Mg once a week", "confidence": 0.7}])
        self.assertEqual([e["name"] for e in found], ["HYDROXY CHLORO QUINE"])
        r = medicine_db.match_medicine("HYDROXY CHLORO QUINE 400mg")
        self.assertEqual((r["match_status"], r["db_info"]["composition"]), ("matched", "Hydroxychloroquine"))


class RateLimitRetryTests(unittest.TestCase):
    class Limited(Exception):
        def __init__(self, wait):
            self.status_code = 429
            self.response = mock.Mock(status_code=429, headers={"retry-after": str(wait)})

    def client(self, effects):
        create = mock.Mock(side_effect=effects)
        return mock.Mock(chat=mock.Mock(completions=mock.Mock(create=create))), create

    def test_short_rate_limit_is_retried_once(self):
        ok = mock.Mock(choices=[mock.Mock(message=mock.Mock(content="{}"))])
        client, create = self.client([self.Limited(0.01), ok])
        out = LLM([Provider("Groq", "q", client)]).complete("s", "u")
        self.assertEqual((out["text"], create.call_count), ("{}", 2))

    def test_long_rate_limit_fails_fast(self):
        from model.common.llm import AIBusy
        client, create = self.client([self.Limited(3600)])
        with self.assertRaises(AIBusy):
            LLM([Provider("Groq", "q", client)]).complete("s", "u")
        self.assertEqual(create.call_count, 1)


class AINameGuardTests(unittest.TestCase):
    def test_placeholder_and_instruction_names_are_dropped(self):
        out = validate({"medicines": [{"name": "Unknown", "strength": "300mg"}, {"name": "In case If fever"},
                                      {"name": "Dolo 650"}, {"name": "Pan D"}]})
        self.assertEqual([m["name"] for m in out["medicines"]], ["Dolo 650", "Pan D"])


class UngroundedNameTests(unittest.TestCase):
    def test_lab_tests_are_not_medicines(self):
        out = validate({"medicines": [{"name": "CBC / CKMB / Pro-BNP / Troponin"}, {"name": "Complete course"},
                                      {"name": "Montair FX"}]})
        self.assertEqual([m["name"] for m in out["medicines"]], ["Montair FX"])

    @unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
    def test_an_ai_name_not_in_the_text_is_never_confirmed(self):
        llm, _ = fake_llm('{"medicines": [{"block": 1, "name": "Dothepin", "strength": "75mg"}, '
                          '{"block": 2, "name": "Risperidone"}], "already_taking": []}')
        r = PrescriptionPipeline(ocr=FakeOCR(["Rx", "Tab Dothepin 75mg 0-0-1", "Tab Xyzqw 2mg"]), llm=llm).run(page())
        risp = next(m for m in r.medicines if m.name == "Risperidone")
        self.assertEqual((risp.match_status, risp.confidence_level), ("verify", "low"))


# ---------------------------------------------------------------------------
# Identification guards (2026-11-17): scripts/eval_matcher.py measures them on 115 + 123 cases
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DATASET, "Indian Medicine Dataset CSV missing")
class IdentificationGuardTests(unittest.TestCase):
    def ingredient(self, written, strength=None):
        from model.prescription.medicine_db import match_all
        m = match_all([{"name": written, "strength": strength, "confidence": 0.9}])[0][0]
        return m["match_status"], ((m.get("db_info") or {}).get("composition") or "").lower(), m

    def test_misread_tab_prefix_is_not_the_brand(self):
        for written, strength, want in [("Td SERTA", "50mg", "sertraline"), ("Tob Dolo 650", None, "paracetamol"),
                                        ("Tub METOLAR", "25mg", "metoprolol"), ("7ab Pan 40", None, "pantoprazole")]:
            status, comp, _ = self.ingredient(written, strength)
            self.assertEqual(status, "matched", written)
            self.assertIn(want, comp, written)

    def test_short_real_brands_keep_their_first_word(self):
        for written, want in [("Pan 40", "pantoprazole"), ("Pan D", "domperidone"), ("Avil 25", "pheniramine")]:
            self.assertIn(want, self.ingredient(written)[1], written)

    def test_spelling_alone_never_confirms_a_brand(self):
        status, _, m = self.ingredient("Colate")             # exact brand: matched
        self.assertEqual(status, "matched")
        status, _, m = self.ingredient("Clopilat")           # misspelt, nothing else written: verify
        self.assertEqual(status, "verify")
        status, comp, _ = self.ingredient("Rivotil", "0.5mg")   # misspelt but the strength confirms it
        self.assertEqual((status, "clonazepam" in comp), ("matched", True))

    def test_non_medicines_are_never_forced_onto_a_drug(self):
        for written in ("Colgate Plax", "Colgate", "Oral B brush", "Sensodyne"):
            status, _, m = self.ingredient(written)
            self.assertEqual(status, "unmatched", written)
            self.assertTrue(m["non_medicine"], written)

    def test_written_words_and_form_that_do_not_fit_stop_a_match(self):
        self.assertNotEqual(self.ingredient("Stolin gum paint")[0], "matched")
        status, _, m = self.ingredient("Dolo Wibble")
        self.assertEqual(status, "verify")

    def test_benchmarks_have_no_wrong_confident_match(self):
        import json
        from scripts.eval_matcher import CASES, HELDOUT, run
        for path in (CASES, HELDOUT):
            rows = run(json.loads(path.read_text())["cases"])
            wrong = [(r["written"], r["product"]) for r in rows if r["outcome"] == "wrong"]
            self.assertEqual(wrong, [], path.name)
            ok = sum(r["outcome"] in ("correct", "verify") for r in rows) / len(rows)
            self.assertGreaterEqual(ok, 0.95, path.name)

    def test_hexigel_gum_paint_is_high_and_not_a_form_clash(self):
        status, _, m = self.ingredient("Hexigel gum paint")
        self.assertEqual((status, m["confidence_level"]), ("matched", "high"))
        f = next(x for x in m["confidence_factors"] if x["factor"] == "Strength")
        self.assertEqual((f["value"], f["effect"]), ("not applicable (one strength only)", "neutral"))
        self.assertEqual(self.ingredient("Dolo")[2]["confidence_level"], "low")          # 11 strengths, none written
        self.assertEqual(self.ingredient("Dolo", "650mg")[2]["confidence_level"], "high")
        self.assertEqual(self.ingredient("Dolo", "999mg")[2]["confidence_level"], "low")  # no Dolo is 999mg
        self.assertEqual(self.ingredient("Shelcal", "500mg")[2]["confidence_level"], "high")   # nothing to compare

    def test_strengths_are_counted_within_the_written_variant(self):
        level = lambda w: self.ingredient(w)[2]["confidence_level"]
        self.assertEqual(level("Sizodon Plus"), "high")       # one product, although "Sizodon" comes in 1-4 mg
        self.assertEqual(level("Zerodol PT"), "high")
        self.assertEqual(level("Sizodon"), "low")             # plain Sizodon: which strength?
        self.assertEqual(level("Augmentin 625 Duo"), "high")  # the strength is part of the written name
