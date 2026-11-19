"""Plant identification tests (no network: PlantNet is mocked).

Run from backend/:  venv/bin/python -m unittest tests.test_plant -v
"""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from PIL import Image  # noqa: E402

from model.plant import identify as plant  # noqa: E402


def result(name, score, common=()):
    return {"score": score, "species": {"scientificNameWithoutAuthor": name, "commonNames": list(common),
                                        "family": {"scientificNameWithoutAuthor": "Lamiaceae"}}}


class PreprocessTests(unittest.TestCase):
    def size(self, w, h):
        return Image.open(io.BytesIO(plant.preprocess_image(Image.new("RGB", (w, h))))).size

    def test_small_photo_upscaled_to_800_short_side(self):
        self.assertEqual(min(self.size(150, 300)), 800)

    def test_large_photo_capped_at_1600_long_side(self):
        self.assertEqual(max(self.size(4000, 3000)), 1600)


class BuildResultTests(unittest.TestCase):
    def test_guide_plant_gets_medicinal_context(self):
        r = plant.build_result([result("Ocimum tenuiflorum", 0.81, ["Holy basil"])], "leaf", 1)
        self.assertTrue(r["guide"]["in_guide"])
        self.assertEqual(r["guide"]["common_name"], "Tulsi / Holy basil")
        self.assertFalse(r["low_confidence"])
        self.assertEqual(r["best"]["family"], "Lamiaceae")

    def test_unknown_plant_has_no_medicinal_claim(self):
        r = plant.build_result([result("Mentha spicata", 0.33, ["Spearmint"])], "auto", 1)
        self.assertFalse(r["guide"]["in_guide"])
        self.assertEqual(r["guide"]["use"], plant.NO_GUIDE_USE)
        self.assertTrue(r["low_confidence"])

    def test_no_results(self):
        r = plant.build_result([], "auto", 2)
        self.assertEqual(r["status"], "no_match")
        self.assertIsNone(r["best"])


class IdentifyTests(unittest.TestCase):
    def test_sends_one_organ_per_image(self):
        resp = mock.Mock(status_code=200, json=lambda: {"results": [result("Aloe vera", 0.9)]})
        with mock.patch.object(plant.httpx, "post", return_value=resp) as post:
            r = plant.identify([Image.new("RGB", (900, 900))] * 2, "flower", key="k")
        self.assertEqual(post.call_args.kwargs["data"], {"organs": ["flower", "flower"]})
        self.assertEqual(len(post.call_args.kwargs["files"]), 2)
        self.assertEqual(r["guide"]["common_name"], "Aloe vera")

    def test_404_is_no_match(self):
        with mock.patch.object(plant.httpx, "post", return_value=mock.Mock(status_code=404, text="Species not found")):
            self.assertEqual(plant.identify([Image.new("RGB", (900, 900))], key="k")["status"], "no_match")

    def test_rate_limit_raises_with_status(self):
        with mock.patch.object(plant.httpx, "post", return_value=mock.Mock(status_code=429, text="Too many")):
            with self.assertRaises(plant.PlantNetError) as ctx:
                plant.identify([Image.new("RGB", (900, 900))], key="k")
        self.assertEqual(ctx.exception.status, 429)

    def test_bad_organ_and_too_many_photos_rejected(self):
        img = Image.new("RGB", (10, 10))
        with self.assertRaises(ValueError):
            plant.identify([img], "root", key="k")
        with self.assertRaises(ValueError):
            plant.identify([img] * 6, key="k")


if __name__ == "__main__":
    unittest.main()


class FakeLLM:
    available = True

    def __init__(self, text=None, busy=False):
        self.text, self.busy, self.calls = text, busy, 0

    def complete(self, system, user, max_tokens=500, json_mode=False):
        from model.common.llm import AIBusy
        self.calls += 1
        if self.busy:
            raise AIBusy([("test", "RateLimitError")])
        return {"text": self.text, "provider": "test", "model": "m", "ms": 1}


class UsesTests(unittest.TestCase):
    REPLY = ('{"known_medicinal": true, "summary": "Used for digestion.", "parts_used": ["Leaf"], "toxic": false,'
             ' "uses": [{"use": "Relieves indigestion", "detail": "Tea of the leaves.", "evidence": "moderate"},'
             '          {"use": "Calms cough", "detail": "Take 2 cups a day.", "evidence": "traditional"},'
             '          {"use": "Sleep", "detail": "x", "evidence": "proven"}],'
             ' "safety": ["Can worsen reflux.", "Do not exceed 500 mg."]}')

    def test_reply_cleaned_doses_dropped_and_cached(self):
        from model.plant.uses import PlantUses
        llm = FakeLLM(self.REPLY)
        u = PlantUses(llm)
        r = u.get("Mentha x piperita", ["Peppermint"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual([x["use"] for x in r["uses"]], ["Relieves indigestion", "Sleep"])   # dose item dropped
        self.assertEqual(r["uses"][1]["evidence"], "traditional")                          # unknown level -> traditional
        self.assertEqual(r["safety"], ["Can worsen reflux."])
        self.assertEqual(r["parts_used"], ["leaf"])
        u.get("mentha X piperita")
        self.assertEqual(llm.calls, 1)

    def test_busy_and_unparseable_are_not_cached(self):
        from model.plant.uses import PlantUses
        self.assertEqual(PlantUses(FakeLLM(busy=True)).get("Aloe vera")["status"], "busy")
        llm = FakeLLM("sorry, no")
        u = PlantUses(llm)
        self.assertEqual(u.get("Aloe vera")["status"], "busy")
        u.get("Aloe vera")
        self.assertEqual(llm.calls, 2)


class SeqLLM(FakeLLM):
    """Answers each call with the next reply in the list (an Exception instance is raised instead)."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    def complete(self, system, user, max_tokens=500, json_mode=False):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply, "provider": "test", "model": "m", "ms": 1}


class TranslationTests(unittest.TestCase):
    EN = ('{"known_medicinal": true, "summary": "Used for digestion.", "parts_used": ["leaf"], "toxic": false,'
          ' "uses": [{"use": "Relieves indigestion", "detail": "Leaf tea is used.", "evidence": "moderate"}],'
          ' "safety": ["Can worsen reflux."]}')
    HI = ('{"summary": "पाचन के लिए उपयोग होता है।", "parts_used": ["पत्ती"], "local_name": "पुदीना",'
          ' "uses": [{"use": "अपच में राहत", "detail": "पत्तियों की चाय उपयोग होती है।", "evidence": "बहुत"}],'
          ' "safety": ["एसिडिटी बढ़ा सकता है।"]}')

    def uses(self, *replies):
        from model.plant.uses import PlantUses
        llm = SeqLLM(*replies)
        return PlantUses(llm), llm

    def test_hindi_translation_keeps_evidence_and_is_cached(self):
        u, llm = self.uses(self.EN, self.HI)
        r = u.get("Mentha x piperita", lang="hi")
        self.assertEqual(r["lang"], "hi")
        self.assertEqual(r["local_name"], "पुदीना")
        self.assertEqual(r["uses"][0]["use"], "अपच में राहत")
        self.assertEqual(r["uses"][0]["evidence"], "moderate")      # taken from the English answer, not the AI
        self.assertEqual(u.get("Mentha x piperita", lang="en")["summary"], "Used for digestion.")
        u.get("Mentha x piperita", lang="hi")
        self.assertEqual(llm.calls, 2)

    def test_dropped_safety_point_retried_once_then_english(self):
        bad = self.HI.replace('"safety": ["एसिडिटी बढ़ा सकता है।"]', '"safety": []')
        u, llm = self.uses(self.EN, bad, self.HI)
        self.assertEqual(u.get("Mentha x piperita", lang="hi")["lang"], "hi")   # second try accepted
        u, llm = self.uses(self.EN, bad, bad, self.HI)
        self.assertEqual(u.get("Mentha x piperita", lang="hi")["lang"], "en")
        self.assertEqual(u.get("Mentha x piperita", lang="hi")["lang"], "hi")   # not cached, so asked again

    def test_foreign_letters_rejected(self):
        mixed = self.HI.replace("पाचन", "पाচন")   # Bengali letters slipped into Hindi
        u, _ = self.uses(self.EN, mixed, mixed)
        self.assertEqual(u.get("Mentha x piperita", lang="hi")["lang"], "en")

    def test_dose_in_translation_rejected(self):
        dosed = self.HI.replace("पत्तियों की चाय उपयोग होती है।", "दिन में २ कप चाय लें।")
        u, _ = self.uses(self.EN, dosed, dosed)
        self.assertEqual(u.get("Mentha x piperita", lang="hi")["lang"], "en")

    def test_wrong_script_rejected(self):
        u, _ = self.uses(self.EN, self.HI, self.HI)
        self.assertEqual(u.get("Mentha x piperita", lang="kn")["lang"], "en")   # Hindi text for a Kannada request

    def test_translation_busy_falls_back_to_english(self):
        from model.common.llm import AIBusy
        busy = AIBusy([("test", "RateLimitError")])
        u, _ = self.uses(self.EN, busy, busy)
        r = u.get("Mentha x piperita", lang="kn")
        self.assertEqual((r["status"], r["lang"]), ("ok", "en"))

    def test_unknown_language_rejected(self):
        u, _ = self.uses()
        with self.assertRaises(ValueError):
            u.get("Aloe vera", lang="ta")

    def test_guide_has_hand_written_translations(self):
        r = plant.build_result([result("Azadirachta indica", 0.9)], "leaf", 1)
        self.assertEqual(r["guide"]["translations"]["kn"]["common_name"], "ಬೇವು")
        self.assertEqual(r["guide"]["translations"]["hi"]["common_name"], "नीम")
