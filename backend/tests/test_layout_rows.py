"""Focused Phase 2A tests for spatial rows and explainable OCR evidence."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.prescription.extract import build_blocks, entry_name, fix_ocr, format_blocks, starts_entry  # noqa: E402
from model.prescription.layout import (arbitrate_box, entry_boundary_decision,
                                       reconstruct_rows)  # noqa: E402


def box(text, x0, y0, x1, y1, confidence=0.8, candidates=None, **extra):
    value = {
        "id": extra.pop("id", text), "text": text, "raw": text,
        "confidence": confidence, "box": [x0, y0, x1, y1],
    }
    if candidates is not None:
        value["candidates"] = candidates
    value.update(extra)
    return value


class SpatialRowTests(unittest.TestCase):
    def test_same_medicine_row_groups_boxes_in_reading_order(self):
        rows, _ = reconstruct_rows([
            box("1)", 20, 100, 45, 130),
            box("Tab Brandx", 55, 102, 220, 132),
            box("10mg", 235, 101, 290, 131),
            box("1-0-1", 305, 102, 380, 132),
        ], 1000, normalize=fix_ocr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "1) Tab Brandx 10mg 1-0-1")

    def test_different_rows_remain_separate(self):
        rows, _ = reconstruct_rows([
            box("Tab Brandx", 30, 100, 220, 130),
            box("10mg", 230, 102, 280, 132),
            box("Tab Brandy", 30, 155, 220, 185),
            box("20mg", 230, 157, 280, 187),
        ], 1000)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["text"] for r in rows], ["Tab Brandx 10mg", "Tab Brandy 20mg"])

    def test_low_confidence_noise_is_evidence_but_not_row_text(self):
        rows, _ = reconstruct_rows([
            box("Tab Brandx", 30, 100, 220, 130),
            box("G", 230, 102, 245, 132, confidence=0.209),
            box("2", 250, 102, 265, 132, confidence=0.264),
        ], 1000)
        self.assertEqual(rows[0]["text"], "Tab Brandx 2")
        self.assertEqual([e["text"] for e in rows[0]["evidence"]["excluded"]], ["G"])
        self.assertEqual(rows[0]["evidence"]["excluded"][0]["id"], "G")

    def test_dose_and_frequency_stay_on_nearby_row(self):
        rows, _ = reconstruct_rows([
            box("Tab Brandx", 30, 100, 220, 130),
            box("1-0-1", 235, 101, 310, 131),
            box("Tab Brandy", 30, 170, 220, 200),
            box("SOS", 235, 171, 285, 201),
        ], 1000)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Tab Brandx 1-0-1")
        self.assertEqual(rows[1]["text"], "Tab Brandy SOS")


class ArbitrationTests(unittest.TestCase):
    def test_all_candidates_are_retained_with_provenance(self):
        result = arbitrate_box(box("Brandx", 0, 0, 100, 20, candidates=[
            {"text": "Brandx", "confidence": 0.70, "engine": "latin", "orientation": "0"},
            {"text": "Brandy", "confidence": 0.68, "engine": "devanagari", "orientation": "180"},
        ], preprocessing="resize"))
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual({c["engine"] for c in result["candidates"]}, {"latin", "devanagari"})
        self.assertEqual(result["preprocessing"], "resize")
        self.assertIn("orientation", result["candidates"][0])

    def test_arbitration_can_prefer_structure_over_slightly_higher_confidence(self):
        result = arbitrate_box(box("9x8", 0, 0, 100, 20, candidates=[
            {"text": "9x8", "confidence": 0.88, "engine": "latin"},
            {"text": "2mg", "confidence": 0.854, "engine": "devanagari"},
        ]))
        self.assertEqual(result["text"], "2mg")
        self.assertIn("structure beat", " ".join(result["reasons"]))

    def test_raw_ocr_is_not_replaced_by_database_name(self):
        result = arbitrate_box(box("Panlocid 2104 0f", 0, 0, 100, 20))
        self.assertEqual(result["raw"], "Panlocid 2104 0f")
        self.assertEqual(result["text"], "Panlocid 2104 0f")

    def test_geometry_is_page_scale_relative(self):
        small, _ = reconstruct_rows([
            box("Tab Brandx", 10, 20, 100, 30), box("10mg", 105, 20, 130, 30),
            box("Tab Brandy", 10, 45, 100, 55),
        ], 200)
        large, _ = reconstruct_rows([
            box("Tab Brandx", 20, 40, 200, 60), box("10mg", 210, 40, 260, 60),
            box("Tab Brandy", 20, 90, 200, 110),
        ], 400)
        self.assertEqual([r["text"] for r in small], [r["text"] for r in large])


class EntryBoundaryTests(unittest.TestCase):
    def test_standalone_numbered_medicine_becomes_new_entry(self):
        decision = entry_boundary_decision({"text": "2 Rivotil0.sms(cloncypem)", "confidence": 0.89})
        self.assertEqual(decision["decision"], "new medicine")
        self.assertIn("numbered marker", decision["reasons"])
        self.assertTrue(starts_entry("2 Rivotil0.sms(cloncypem)"))
        self.assertEqual(entry_name("2 Rivotil0.sms(cloncypem)"), "Rivotil")

    def test_corrupted_prefix_with_strength_becomes_new_entry(self):
        decision = entry_boundary_decision({"text": "Td SERTA 50mg -1Nisw", "confidence": 0.75})
        self.assertEqual(decision["decision"], "new medicine")
        self.assertIn("short OCR prefix with strength", decision["reasons"])
        self.assertTrue(starts_entry("Td SERTA 50mg -1Nisw"))
        self.assertEqual(entry_name("Td SERTA 50mg -1Nisw"), "SERTA")

    def test_ordinary_continuation_does_not_open_entry(self):
        decision = entry_boundary_decision({"text": "take after food", "confidence": 0.9})
        self.assertEqual(decision["decision"], "continuation")
        self.assertFalse(starts_entry("take after food"))
        self.assertIsNone(entry_name("take after food"))

    def test_dose_frequency_continuation_stays_with_previous_block(self):
        layout = build_blocks([
            {"text": "Tab Brandx 10mg", "confidence": 0.9},
            {"text": "1-0-1", "confidence": 0.8},
            {"text": "2 Rivotil0.sms(cloncypem)", "confidence": 0.8},
        ])
        self.assertEqual(len(layout["blocks"]), 2)
        self.assertEqual([r["text"] for r in layout["blocks"][0]], ["Tab Brandx 10mg", "1-0-1"])
        self.assertEqual(layout["blocks"][1][0]["text"], "2 Rivotil0.sms(cloncypem)")

    def test_corrupted_form_prefix_is_not_required_when_other_evidence_is_strong(self):
        decision = entry_boundary_decision({"text": "Tob Brandx 10mg", "confidence": 0.8})
        self.assertEqual(decision["decision"], "new medicine")
        self.assertEqual(entry_name("Tob Brandx 10mg"), "Brandx")

    def test_boundary_decision_is_page_scale_invariant(self):
        small = entry_boundary_decision(
            {"text": "2 Rivotil 0.5mg", "confidence": 0.8, "box": [10, 40, 100, 60]},
            {"text": "Tab Prior 10mg", "confidence": 0.8, "box": [10, 10, 100, 30]},
            line_h=20, page_width=200)
        large = entry_boundary_decision(
            {"text": "2 Rivotil 0.5mg", "confidence": 0.8, "box": [20, 80, 200, 120]},
            {"text": "Tab Prior 10mg", "confidence": 0.8, "box": [20, 20, 200, 60]},
            line_h=40, page_width=400)
        self.assertEqual(small["decision"], large["decision"])
        self.assertEqual(small["score"], large["score"])

    def test_boundary_logic_contains_no_medicine_names(self):
        import inspect
        from model.prescription import layout
        source = inspect.getsource(layout.entry_boundary_decision)
        self.assertNotIn("Rivot", source)
        self.assertNotIn("SERTA", source)


if __name__ == "__main__":
    unittest.main()
