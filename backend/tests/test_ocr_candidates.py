"""OCR layer tests for the prescription profile (Phase 1). No real models are loaded.

    venv/bin/python -m unittest tests.test_ocr_candidates -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.common.ocr import (PROFILE_LABEL, PROFILE_PRESCRIPTION, OCREngine, _crop)  # noqa: E402
from model.common.preprocess import prescription_variants  # noqa: E402


class FakeResult:
    def __init__(self, boxes, txts, scores):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


class FakeRapidOCR:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)
        return self.result


def quad(x0=0, y0=0, x1=100, y1=20):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class OCRProfileTests(unittest.TestCase):
    def engine(self, result):
        eng = OCREngine("v6-small", hindi=False)
        eng._main = FakeRapidOCR(result)
        return eng

    def test_prescription_full_pass_turns_off_cls_and_keeps_low_scores(self):
        eng = self.engine(FakeResult([], [], []))
        eng._full(np.zeros((10, 10, 3), dtype=np.uint8), PROFILE_PRESCRIPTION)
        call = eng._main.calls[-1]
        self.assertIs(call["use_cls"], False)
        self.assertEqual(call["text_score"], 0.0)
        self.assertIs(call["use_det"], True)
        self.assertIs(call["use_rec"], True)

    def test_label_full_pass_keeps_current_behaviour(self):
        eng = self.engine(FakeResult([], [], []))
        eng._full(np.zeros((10, 10, 3), dtype=np.uint8), PROFILE_LABEL)
        call = eng._main.calls[-1]
        self.assertIs(call["use_cls"], True)
        self.assertEqual(call["text_score"], 0.5)
        self.assertIs(call["use_det"], True)
        self.assertIs(call["use_rec"], True)

    def test_label_profile_restores_default_score_after_prescription_call(self):
        """RapidOCR remembers the last text_score, so label must set 0.5 explicitly."""
        eng = self.engine(FakeResult([], [], []))
        eng._full(np.zeros((10, 10, 3), dtype=np.uint8), PROFILE_PRESCRIPTION)
        eng._full(np.zeros((10, 10, 3), dtype=np.uint8), PROFILE_LABEL)
        self.assertEqual(eng._main.calls[-1]["text_score"], 0.5)
        eng._rec_only(np.zeros((10, 10, 3), dtype=np.uint8), PROFILE_LABEL)
        self.assertEqual(eng._main.calls[-1]["text_score"], 0.5)

    def test_label_hindi_pass_also_restores_default_score(self):
        main = FakeResult([quad()], ["Tab X"], [0.9])
        eng = self.engine(main)
        eng._hi = FakeRapidOCR(FakeResult([], ["टैब एक्स"], [0.8]))
        with mock.patch.object(eng, "_upright", return_value=(np.zeros((20, 100, 3), dtype=np.uint8), main, 0)):
            eng.read(np.zeros((20, 100, 3), dtype=np.uint8), hindi=True, profile=PROFILE_LABEL)
        self.assertEqual(eng._hi.calls[-1]["text_score"], 0.5)

    def test_unknown_profile_is_rejected(self):
        eng = self.engine(FakeResult([], [], []))
        with self.assertRaises(ValueError):
            eng._full(np.zeros((10, 10, 3), dtype=np.uint8), "typo")

    def test_low_confidence_candidate_survives_in_prescription_profile(self):
        result = FakeResult([quad()], ["ANdip 1040D"], [0.42])
        eng = self.engine(result)
        with mock.patch.object(eng, "_upright", return_value=(np.zeros((20, 100, 3), dtype=np.uint8), result, 0)):
            r = eng.read(np.zeros((20, 100, 3), dtype=np.uint8), hindi=False,
                         profile=PROFILE_PRESCRIPTION, preprocessing="exif+resize")
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["boxes"]), 1)
        self.assertEqual(r["rows"][0]["text"], "ANdip 1040D")
        self.assertEqual(r["rows"][0]["confidence"], 0.42)
        self.assertEqual(r["boxes"][0]["candidates"][0]["confidence"], 0.42)

    def test_truly_empty_text_is_still_discarded(self):
        result = FakeResult([quad(), quad(0, 30, 100, 50)], ["", "Tab X"], [0.9, 0.8])
        eng = self.engine(result)
        with mock.patch.object(eng, "_upright", return_value=(np.zeros((60, 100, 3), dtype=np.uint8), result, 0)):
            r = eng.read(np.zeros((60, 100, 3), dtype=np.uint8), hindi=False, profile=PROFILE_PRESCRIPTION)
        self.assertEqual(len(r["boxes"]), 1)
        self.assertEqual(r["boxes"][0]["text"], "Tab X")

    def test_candidate_metadata_is_preserved(self):
        result = FakeResult([quad(3, 4, 80, 24)], ["Tab X"], [0.61])
        eng = self.engine(result)
        with mock.patch.object(eng, "_upright", return_value=(np.zeros((30, 100, 3), dtype=np.uint8), result, 90)):
            r = eng.read(np.zeros((30, 100, 3), dtype=np.uint8), hindi=False,
                         profile=PROFILE_PRESCRIPTION, preprocessing="exif+resize")
        b = r["boxes"][0]
        self.assertEqual(b["id"], "B1")
        self.assertEqual(b["text"], "Tab X")
        self.assertEqual(b["confidence"], 0.61)
        self.assertEqual(b["box"], [3.0, 4.0, 80.0, 24.0])
        self.assertEqual(b["engine"], "RapidOCR v6-small")
        self.assertEqual(b["orientation"], "90")
        self.assertEqual(b["preprocessing"], "exif+resize")
        self.assertEqual(b["candidates"][0]["text"], "Tab X")
        self.assertEqual(r["profile"], PROFILE_PRESCRIPTION)

    def test_devanagari_reading_is_kept_as_an_alternative_not_destroyed(self):
        main = FakeResult([quad()], ["Tab X"], [0.40])
        eng = self.engine(main)
        eng._hi = FakeRapidOCR(FakeResult([], ["टैब एक्स"], [0.90]))
        with mock.patch.object(eng, "_upright", return_value=(np.zeros((20, 100, 3), dtype=np.uint8), main, 0)):
            r = eng.read(np.zeros((20, 100, 3), dtype=np.uint8), hindi=True, profile=PROFILE_PRESCRIPTION)
        texts = [c["text"] for c in r["boxes"][0]["candidates"]]
        self.assertEqual(texts, ["Tab X", "टैब एक्स"])
        self.assertEqual(r["boxes"][0]["text"], "टैब एक्स")
        self.assertEqual(r["boxes"][0]["selected"], 1)


class CropTests(unittest.TestCase):
    def test_prescription_variants_keep_bgr_shape(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        variants = prescription_variants(image)
        self.assertEqual([name for name, _ in variants], ["clahe", "otsu"])
        self.assertTrue(all(v.shape == image.shape for _, v in variants))

    def test_tall_crop_is_not_rotated_by_default(self):
        img = np.full((200, 100, 3), 255, dtype=np.uint8)
        crop = _crop(img, quad(0, 0, 100, 200))
        self.assertGreater(crop.shape[0], crop.shape[1])

    def test_rotation_is_explicit(self):
        img = np.full((200, 100, 3), 255, dtype=np.uint8)
        crop = _crop(img, quad(0, 0, 100, 200), rotate_tall=True)
        self.assertGreater(crop.shape[1], crop.shape[0])


if __name__ == "__main__":
    unittest.main()
