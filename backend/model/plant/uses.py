"""Medicinal uses of an identified plant, written by the project's text AI (model/common/llm.py).

Only the plant's name is sent (never the photo). The AI answers as JSON; the reply is validated, trimmed,
and anything that looks like a dose is dropped, because doses must come from a doctor, not from this app.
Answers are cached per scientific name for the life of the server.

Hindi and Kannada: the checked English answer is translated by a second AI call (same facts in every
language), validated (same number of uses and safety points, right script, no dose), then cached per
(name, language). If translation fails the English answer is returned with ``lang: "en"``.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from model.common.llm import AIBusy, LLM, data_block, default_llm, parse_json_loose

log = logging.getLogger("plant.uses")

SYSTEM = """You are a careful pharmacognosy assistant. You get the scientific name of a plant identified from a photo.
Describe its medicinal uses for a general reader in India. Rules:
- Only state uses that are documented in traditional medicine (Ayurveda, Unani, Siddha, folk use) or in scientific studies. Do not invent uses.
- Say how strong the evidence is: "traditional" (use only), "limited" (small or lab/animal studies), or "moderate" (several human studies).
- Never give doses, amounts, frequencies or recipes with quantities.
- Always list real safety concerns: toxicity, pregnancy/breastfeeding, children, allergies, interactions with medicines.
- If the plant has no known medicinal use, say so and return empty "uses".
- If the plant is poisonous, set "toxic" to true and say so first in "safety".
Return only JSON:
{"known_medicinal": true|false, "summary": "1-2 plain sentences",
 "uses": [{"use": "short phrase, e.g. Relieves indigestion", "detail": "one sentence", "evidence": "traditional|limited|moderate"}],
 "parts_used": ["leaf", ...], "safety": ["one sentence each"], "toxic": true|false}
At most 6 uses and 5 safety points."""

TRANSLATE = """You translate notes about a plant's medicinal uses from English into {language} for a general reader in India.
Rules:
- Translate only the text values. Keep every JSON key exactly as given, in English.
- Keep the same number of items in the same order. Do not add, drop, soften or strengthen anything, above all the safety points.
- Never add doses, amounts or frequencies.
- Write simple everyday {language} in {script} script. Well-known medical loanwords are fine.
- Keep the scientific name in Latin letters; do not translate it.
- Add "local_name": the plant's common name in {language} ({script} script), or "" if it has no well-known one.
Return only JSON with the same shape as the input plus "local_name"."""

# lang code -> (language, script, Unicode block the translated text must use)
LANGS = {
    "en": ("English", "Latin", None),
    "hi": ("Hindi", "Devanagari", re.compile(r"[\u0900-\u097F]")),
    "kn": ("Kannada", "Kannada", re.compile(r"[\u0C80-\u0CFF]")),
}
EVIDENCE = {"traditional", "limited", "moderate"}
# Anything dose-like: "500 mg", "2 cups", "3 times a day", "10 ml", and the same in Hindi and Kannada.
# \d also matches Devanagari (०-९) and Kannada (೦-೯) digits.
DOSE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg|g\b|grams?|ml|cups?|teaspoons?|tsp|tablespoons?|tbsp|drops?|times|capsules?|tablets?"
    r"|मिलीग्राम|मि\.?ग्रा|ग्राम|मिली|कप|चम्मच|बूँद|बूंद|बार|गोली|गोलियाँ|कैप्सूल"
    r"|ಮಿಲಿಗ್ರಾಂ|ಮಿ\.?ಗ್ರಾಂ|ಗ್ರಾಂ|ಮಿಲಿ|ಕಪ್|ಚಮಚ|ಹನಿ|ಬಾರಿ|ಮಾತ್ರೆ|ಕ್ಯಾಪ್ಸೂಲ್)", re.I)
MAX_CACHE = 300


def _text(v: Any, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


def _clean(raw: Dict[str, Any]) -> Dict[str, Any]:
    uses: List[Dict[str, str]] = []
    for u in raw.get("uses") or []:
        if not isinstance(u, dict):
            continue
        use, detail = _text(u.get("use"), 80), _text(u.get("detail"), 240)
        if not use or DOSE.search(use + " " + detail):
            continue
        ev = _text(u.get("evidence"), 20).lower()
        uses.append({"use": use, "detail": detail, "evidence": ev if ev in EVIDENCE else "traditional"})
    safety = [s for s in (_text(x, 240) for x in raw.get("safety") or []) if s and not DOSE.search(s)]
    parts = [p for p in (_text(x, 30).lower() for x in raw.get("parts_used") or []) if p]
    summary = _text(raw.get("summary"), 400)
    if DOSE.search(summary):
        summary = ""
    return {
        "known_medicinal": bool(raw.get("known_medicinal")) and bool(uses),
        "summary": summary,
        "uses": uses[:6],
        "parts_used": parts[:6],
        "safety": safety[:5] or ["Ask a doctor before using any plant medicinally."],
        "toxic": bool(raw.get("toxic")),
    }


def _script_ok(text: str, block: "re.Pattern[str]") -> bool:
    """Some text in the target script, and every letter or vowel sign either Latin or in that script.
    Free models sometimes slip in letters from another script (Tamil, Bengali, even Korean)."""
    if not block.search(text):
        return False
    return all(ch < "\u0250" or block.match(ch) for ch in text if unicodedata.category(ch)[0] in "LM")


def _clean_translation(en: Dict[str, Any], raw: Dict[str, Any], lang: str) -> Optional[Dict[str, Any]]:
    """The translated answer, or None if it doesn't match the English one item for item (or adds a dose)."""
    uses, safety = raw.get("uses") or [], raw.get("safety") or []
    if not isinstance(uses, list) or not isinstance(safety, list):
        return None
    if len(uses) != len(en["uses"]) or len(safety) != len(en["safety"]):
        return None
    out_uses: List[Dict[str, str]] = []
    for e, t in zip(en["uses"], uses):
        if not isinstance(t, dict):
            return None
        use, detail = _text(t.get("use"), 160), _text(t.get("detail"), 480)
        if not use or (e["detail"] and not detail):
            return None
        out_uses.append({"use": use, "detail": detail, "evidence": e["evidence"]})
    out_safety = [_text(x, 480) for x in safety]
    summary = _text(raw.get("summary"), 800)
    if not all(out_safety) or (en["summary"] and not summary):
        return None
    parts = [_text(x, 60) for x in raw.get("parts_used") or []] if isinstance(raw.get("parts_used"), list) else []
    if len(parts) != len(en["parts_used"]) or not all(parts):
        parts = en["parts_used"]
    text = " ".join([summary, *out_safety, *(u["use"] + " " + u["detail"] for u in out_uses)])
    if DOSE.search(text) or not _script_ok(text + " " + " ".join(parts), LANGS[lang][2]):
        return None
    return {**en, "summary": summary, "uses": out_uses, "parts_used": parts, "safety": out_safety,
            "local_name": _text(raw.get("local_name"), 60), "lang": lang}


class PlantUses:
    def __init__(self, llm: Optional[LLM] = None) -> None:
        self._llm = llm
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def llm(self) -> LLM:
        return self._llm if self._llm is not None else default_llm()

    def get(self, scientific_name: str, common_names: Sequence[str] = (), lang: str = "en") -> Dict[str, Any]:
        if lang not in LANGS:
            raise ValueError(f"lang must be one of {', '.join(LANGS)}")
        name = _text(scientific_name, 120)
        if not name:
            raise ValueError("scientific_name is required")
        en = self._english(name, common_names)
        if lang == "en" or en["status"] != "ok":
            return {**en, "lang": "en"}
        key = f"{lang}:{name.lower()}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        translated = self._translate(en, lang)
        if translated is None:
            return {**en, "lang": "en"}   # the page says the translation isn't available right now
        self._store(key, translated)
        return translated

    def _cached(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _store(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > MAX_CACHE:
                self._cache.popitem(last=False)

    def _translate(self, en: Dict[str, Any], lang: str) -> Optional[Dict[str, Any]]:
        language, script, _ = LANGS[lang]
        notes = {"summary": en["summary"], "uses": [{"use": u["use"], "detail": u["detail"]} for u in en["uses"]],
                 "parts_used": en["parts_used"], "safety": en["safety"]}
        user = (data_block("PLANT", f"Scientific name: {en['scientific_name']}", 200) + "\n"
                + data_block("NOTES", json.dumps(notes, ensure_ascii=False), 5000))
        # Each provider gets one try; a rejected or busy reply moves on to the next one. (The free OpenRouter
        # model mixes Devanagari/Bengali letters into Kannada; the Groq backup writes it cleanly.)
        providers = getattr(self.llm, "providers", None)
        tries = [LLM([p]) for p in providers] if providers else [self.llm, self.llm]
        for llm in tries:
            try:
                # Indic scripts take several times more tokens than English.
                out = llm.complete(TRANSLATE.format(language=language, script=script), user,
                                   max_tokens=2500, json_mode=True)
            except AIBusy as exc:
                log.info("Plant uses: translation to %s busy (%s)", lang, exc)
                continue
            raw = parse_json_loose(out["text"])
            result = _clean_translation(en, raw, lang) if raw is not None else None
            if result is not None:
                return result
            log.info("Plant uses: translation to %s rejected", lang)
        return None

    def _english(self, name: str, common_names: Sequence[str]) -> Dict[str, Any]:
        key = f"en:{name.lower()}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        if not self.llm.available:
            return {"status": "unavailable", "scientific_name": name}
        user = data_block("PLANT", f"Scientific name: {name}\nCommon names: {', '.join(common_names[:5]) or 'unknown'}", 400)
        try:
            out = self.llm.complete(SYSTEM, user, max_tokens=900, json_mode=True)
        except AIBusy as exc:
            log.info("Plant uses: AI busy (%s)", exc)
            return {"status": "busy", "scientific_name": name}
        raw = parse_json_loose(out["text"])
        if raw is None:
            log.info("Plant uses: AI reply was not JSON")
            return {"status": "busy", "scientific_name": name}
        result = {"status": "ok", "scientific_name": name, **_clean(raw),
                  "source": {"provider": out["provider"], "model": out["model"]}}
        self._store(key, result)
        return result
