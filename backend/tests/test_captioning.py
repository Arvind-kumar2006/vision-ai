"""Tests for the pretrained Show, Attend and Tell captioning path.

Run from backend/:   python -m unittest discover -s tests -v

Requires backend/weights/sat_resnet101_coco.pth
(create it with: python -m model.captioning.convert_checkpoint).
"""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

WEIGHTS = BACKEND_ROOT / "weights" / "sat_resnet101_coco.pth"
SAMPLE = BACKEND_ROOT / "data" / "samples" / "prescription_test.jpeg"


def _test_image(seed: int = 0, size=(320, 240)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


@unittest.skipUnless(WEIGHTS.exists(), f"missing {WEIGHTS}")
class SATModelTests(unittest.TestCase):
    service = None

    @classmethod
    def setUpClass(cls):
        from app.services.captioners.sat import SATCaptioner

        cls.service = SATCaptioner()
        assert cls.service.ready, cls.service.error

    # TEST-01 -------------------------------------------------------------
    def test_01_encoder_output_shape(self):
        """Encoder maps a 256x256 image to a (1, 14, 14, 2048) feature grid."""
        x = self.service._preprocess(_test_image())
        self.assertEqual(tuple(x.shape), (1, 3, 256, 256))
        with torch.no_grad():
            out = self.service.encoder(x)
        self.assertEqual(tuple(out.shape), (1, 14, 14, 2048))

    def test_02_attention_weights_sum_to_one_each_step(self):
        """Soft attention over the 196 positions is a distribution at every decode step."""
        svc = self.service
        dec = svc.decoder
        with torch.no_grad():
            enc = svc.encoder(svc._preprocess(_test_image(1))).view(1, -1, 2048)
            self.assertEqual(enc.shape[1], 196)
            h, c = dec.init_hidden_state(enc)
            word = torch.tensor([svc.word_map["<start>"]], device=svc.device)
            for _ in range(10):
                awe, alpha = dec.attention(enc, h)
                self.assertEqual(tuple(alpha.shape), (1, 196))
                self.assertTrue(bool((alpha >= 0).all()))
                self.assertAlmostEqual(alpha.sum().item(), 1.0, places=4)
                awe = dec.sigmoid(dec.f_beta(h)) * awe
                h, c = dec.decode_step(torch.cat([dec.embedding(word), awe], dim=1), (h, c))
                word = dec.fc(h).argmax(dim=1)

    def test_03_caption_output_contract(self):
        """caption/words/alphas line up; maps are 14x14 scaled 0-1; specials stripped."""
        r = self.service.caption(Image.open(SAMPLE) if SAMPLE.exists() else _test_image(2))
        self.assertTrue(r["caption"])
        self.assertEqual(r["caption"], " ".join(r["words"]))
        self.assertEqual(len(r["words"]), len(r["alphas"]))
        self.assertEqual(r["grid_size"], 14)
        for grid in r["alphas"]:
            a = np.asarray(grid)
            self.assertEqual(a.shape, (14, 14))
            self.assertGreaterEqual(a.min(), 0.0)
            self.assertAlmostEqual(float(a.max()), 1.0, places=2)
        self.assertFalse({"<start>", "<end>", "<pad>", "<unk>"} & set(r["words"]))
        self.assertGreater(r["generation_time_ms"], 0)

    def test_04_vocabulary_is_coco_word_map(self):
        wm = self.service.word_map
        self.assertEqual(len(wm), 9490)
        self.assertEqual(wm["<pad>"], 0)
        for tok in ("<start>", "<end>", "<unk>"):
            self.assertIn(tok, wm)
        self.assertEqual(self.service.decoder.fc.out_features, len(wm))

    def test_05_beam_size_and_max_len(self):
        img = _test_image(3)
        for beam in (1, 3, 5):
            self.assertTrue(self.service.caption(img, beam_size=beam)["words"])
        self.assertLessEqual(len(self.service.caption(img, max_len=4)["words"]), 4)


class RepeatBlockingTests(unittest.TestCase):
    def test_blocks_repeated_bigram(self):
        """After '<start> a dog ... a', the token 'dog' must be banned (would repeat 'a dog')."""
        from app.services.captioners.sat import SATCaptioner as CaptionService

        START, A, DOG, IN, GRASS = 0, 1, 2, 3, 4
        scores = torch.zeros(1, 5)
        CaptionService._block_repeated_ngrams(scores, [[START, A, DOG, IN, A]], n=2)
        self.assertEqual(scores[0, DOG].item(), float("-inf"))
        self.assertEqual(scores[0, GRASS].item(), 0.0)  # new pairs still allowed

    def test_start_token_not_counted(self):
        from app.services.captioners.sat import SATCaptioner as CaptionService

        scores = torch.zeros(1, 5)
        CaptionService._block_repeated_ngrams(scores, [[0, 1]], n=2)
        self.assertTrue(bool((scores == 0).all()))


CATR_WEIGHTS = BACKEND_ROOT / "weights" / "catr_v3_coco.pth"


class SharedHelperTests(unittest.TestCase):
    def test_merge_wordpieces_averages_maps(self):
        from app.services.captioners.common import merge_wordpieces

        maps = [np.full((2, 2), v, dtype=np.float32) for v in (1.0, 3.0, 5.0)]
        words, merged = merge_wordpieces(["a", "fr", "##isbee"], maps)
        self.assertEqual(words, ["a", "frisbee"])
        self.assertEqual(float(merged[1][0, 0]), 4.0)

    def test_block_repeated_ngrams_skips_start(self):
        from app.services.captioners.common import block_repeated_ngrams

        scores = torch.zeros(1, 6)
        block_repeated_ngrams(scores, [[5, 1, 2, 3, 1]], n=2, skip=1)
        self.assertEqual(scores[0, 2].item(), float("-inf"))


@unittest.skipUnless(CATR_WEIGHTS.exists(), f"missing {CATR_WEIGHTS}")
class CATRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.services.caption_service import load_inference_cfg
        from app.services.captioners.catr import CATRCaptioner

        cls.engine = CATRCaptioner(load_inference_cfg())
        assert cls.engine.ready, cls.engine.error

    def test_output_contract_non_square_grid(self):
        r = self.engine.caption(_test_image(5, size=(400, 240)))
        rows, cols = r["grid_shape"]
        self.assertNotEqual(rows, cols)  # follows the image aspect ratio
        self.assertEqual(len(r["words"]), len(r["alphas"]))
        for grid in r["alphas"]:
            a = np.asarray(grid)
            self.assertEqual(a.shape, (rows, cols))
            self.assertAlmostEqual(float(a.max()), 1.0, places=2)
        self.assertFalse(any(w.startswith("##") or w in ("[CLS]", "[SEP]", "[UNK]") for w in r["words"]))
        self.assertEqual(r["model"], "catr")


@unittest.skipUnless(WEIGHTS.exists(), f"missing {WEIGHTS}")
class CaptionAPITests(unittest.TestCase):
    def test_caption_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import app

        buf = io.BytesIO()
        _test_image(4).save(buf, format="JPEG")
        with TestClient(app) as client:
            health = client.get("/api/health").json()
            self.assertEqual(health["modes"]["caption"], "ready")
            self.assertIn(health["caption_model"], ("sat", "catr", "blip"))

            r = client.post("/api/caption?beam_size=3&model=sat",
                            files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            for key in ("mode", "caption", "inference_ms", "words", "alphas",
                        "grid_size", "grid_shape", "generation_time_ms", "model"):
                self.assertIn(key, body)
            self.assertEqual(body["grid_shape"], [14, 14])
            self.assertEqual(body["model"], "sat")
            self.assertIn("caption_models", health)

            bad_model = client.post("/api/caption?model=gpt",
                                    files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
            self.assertEqual(bad_model.status_code, 422)

            bad = client.post("/api/caption", files={"file": ("x.txt", b"hi", "text/plain")})
            self.assertEqual(bad.status_code, 415)
            bad_beam = client.post("/api/caption?beam_size=0",
                                   files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
            self.assertEqual(bad_beam.status_code, 422)


if __name__ == "__main__":
    unittest.main()
