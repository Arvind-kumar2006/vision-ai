"""Identify a plant from up to five photos with the PlantNet API.

Ported from the tested Streamlit app (plants_classification/app.py): same preprocessing, same request,
same local guide. The LangChain template there only formatted text, so the fields are returned as JSON here
and the React page lays them out.
"""
from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

import httpx
from PIL import Image, ImageEnhance

# PlantNet takes the key as a URL parameter, and httpx logs every request URL at INFO: keep the key out of logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

PLANTNET_ENDPOINT = "https://my-api.plantnet.org/v2/identify/all"
ORGANS = ("auto", "leaf", "flower", "fruit", "bark")
MAX_IMAGES = 5
LOW_CONFIDENCE = 0.60

# scientific name (lower case) -> (common name, traditional/researched context, safety)
GUIDE = {
    "ocimum tenuiflorum": ("Tulsi / Holy basil", "Traditionally used in some herbal practices.", "Do not assume concentrated products are safe or free of drug interactions."),
    "ocimum sanctum": ("Tulsi / Holy basil", "Traditionally used in some herbal practices.", "Do not assume concentrated products are safe or free of drug interactions."),
    "azadirachta indica": ("Neem", "Traditionally used in some topical and herbal preparations.", "Oral use can be toxic, especially for children and during pregnancy."),
    "aloe vera": ("Aloe vera", "The gel is traditionally used for some topical applications.", "Oral aloe latex can cause adverse effects and interactions."),
    "withania somnifera": ("Ashwagandha", "Used in Ayurvedic traditions and studied for possible effects.", "May interact with medicines and may not be suitable for everyone."),
    "phyllanthus emblica": ("Amla / Indian gooseberry", "Used traditionally as a food and herbal ingredient.", "Evidence and product quality vary."),
}
# The same guide entries in Hindi and Kannada, written by hand (not by the AI). Keyed like GUIDE.
GUIDE_LOCAL = {
    "ocimum tenuiflorum": {
        "hi": ("तुलसी", "कुछ पारंपरिक जड़ी-बूटी पद्धतियों में इसका उपयोग होता है।",
               "यह न मानें कि इसके सांद्र (कंसन्ट्रेटेड) उत्पाद सुरक्षित हैं या दवाओं के साथ प्रतिक्रिया नहीं करते।"),
        "kn": ("ತುಳಸಿ", "ಕೆಲವು ಸಾಂಪ್ರದಾಯಿಕ ಗಿಡಮೂಲಿಕೆ ಪದ್ಧತಿಗಳಲ್ಲಿ ಇದನ್ನು ಬಳಸಲಾಗುತ್ತದೆ.",
               "ಇದರ ಸಾಂದ್ರೀಕೃತ ಉತ್ಪನ್ನಗಳು ಸುರಕ್ಷಿತವೆಂದು ಅಥವಾ ಔಷಧಿಗಳೊಂದಿಗೆ ಪ್ರತಿಕ್ರಿಯಿಸುವುದಿಲ್ಲವೆಂದು ಭಾವಿಸಬೇಡಿ."),
    },
    "azadirachta indica": {
        "hi": ("नीम", "पारंपरिक रूप से त्वचा पर लगाने वाली और कुछ जड़ी-बूटी की तैयारियों में इसका उपयोग होता है।",
               "मुँह से लेना ज़हरीला हो सकता है, खासकर बच्चों के लिए और गर्भावस्था में।"),
        "kn": ("ಬೇವು", "ಸಾಂಪ್ರದಾಯಿಕವಾಗಿ ಚರ್ಮಕ್ಕೆ ಹಚ್ಚುವ ಮತ್ತು ಕೆಲವು ಗಿಡಮೂಲಿಕೆ ತಯಾರಿಕೆಗಳಲ್ಲಿ ಇದನ್ನು ಬಳಸಲಾಗುತ್ತದೆ.",
               "ಬಾಯಿಯಿಂದ ಸೇವಿಸುವುದು ವಿಷಕಾರಿಯಾಗಬಹುದು, ವಿಶೇಷವಾಗಿ ಮಕ್ಕಳಿಗೆ ಮತ್ತು ಗರ್ಭಾವಸ್ಥೆಯಲ್ಲಿ."),
    },
    "aloe vera": {
        "hi": ("एलोवेरा / घृतकुमारी", "इसका जेल पारंपरिक रूप से त्वचा पर कुछ उपयोगों के लिए लगाया जाता है।",
               "एलो का लेटेक्स मुँह से लेने पर दुष्प्रभाव और दवाओं से प्रतिक्रिया हो सकती है।"),
        "kn": ("ಲೋಳೆಸರ", "ಇದರ ಜೆಲ್ ಅನ್ನು ಸಾಂಪ್ರದಾಯಿಕವಾಗಿ ಚರ್ಮದ ಕೆಲವು ಬಳಕೆಗಳಿಗೆ ಹಚ್ಚಲಾಗುತ್ತದೆ.",
               "ಲೋಳೆಸರದ ಲೇಟೆಕ್ಸ್ ಅನ್ನು ಬಾಯಿಯಿಂದ ಸೇವಿಸಿದರೆ ಅಡ್ಡಪರಿಣಾಮಗಳು ಮತ್ತು ಔಷಧಿ ಪ್ರತಿಕ್ರಿಯೆಗಳು ಉಂಟಾಗಬಹುದು."),
    },
    "withania somnifera": {
        "hi": ("अश्वगंधा", "आयुर्वेदिक परंपरा में इसका उपयोग होता है और इसके संभावित प्रभावों पर अध्ययन हुए हैं।",
               "दवाओं के साथ प्रतिक्रिया कर सकता है और हर किसी के लिए उपयुक्त नहीं हो सकता।"),
        "kn": ("ಅಶ್ವಗಂಧ", "ಆಯುರ್ವೇದ ಪರಂಪರೆಯಲ್ಲಿ ಬಳಸಲಾಗುತ್ತದೆ ಮತ್ತು ಇದರ ಸಂಭಾವ್ಯ ಪರಿಣಾಮಗಳ ಬಗ್ಗೆ ಅಧ್ಯಯನಗಳು ನಡೆದಿವೆ.",
               "ಔಷಧಿಗಳೊಂದಿಗೆ ಪ್ರತಿಕ್ರಿಯಿಸಬಹುದು ಮತ್ತು ಎಲ್ಲರಿಗೂ ಸೂಕ್ತವಾಗದಿರಬಹುದು."),
    },
    "phyllanthus emblica": {
        "hi": ("आंवला", "पारंपरिक रूप से भोजन और जड़ी-बूटी के रूप में इसका उपयोग होता है।",
               "इसके प्रमाण और उत्पादों की गुणवत्ता अलग-अलग होती है।"),
        "kn": ("ನೆಲ್ಲಿಕಾಯಿ", "ಸಾಂಪ್ರದಾಯಿಕವಾಗಿ ಆಹಾರವಾಗಿ ಮತ್ತು ಗಿಡಮೂಲಿಕೆ ಪದಾರ್ಥವಾಗಿ ಬಳಸಲಾಗುತ್ತದೆ.",
               "ಇದರ ಪುರಾವೆಗಳು ಮತ್ತು ಉತ್ಪನ್ನಗಳ ಗುಣಮಟ್ಟ ಬದಲಾಗುತ್ತದೆ."),
    },
}
GUIDE_LOCAL["ocimum sanctum"] = GUIDE_LOCAL["ocimum tenuiflorum"]
NO_GUIDE_USE = "No verified medicinal entry is available in this app."
NO_GUIDE_SAFETY = "Do not infer medicinal use or safety from the identification alone."


class PlantNetError(RuntimeError):
    """PlantNet refused or failed the request. ``status`` is the HTTP status (None if unreachable)."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def api_key() -> str:
    return os.getenv("PLANTNET_API_KEY", "").strip()


def preprocess_image(image: Image.Image) -> bytes:
    """Upscale small photos to 800 px short side, cap at 1600 px long side, mild contrast/sharpness, JPEG q90."""
    image = image.convert("RGB")
    if min(image.size) < 800:
        f = 800 / min(image.size)
        image = image.resize((round(image.width * f), round(image.height * f)), Image.Resampling.LANCZOS)
    if max(image.size) > 1600:
        f = 1600 / max(image.size)
        image = image.resize((round(image.width * f), round(image.height * f)), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.05)
    image = ImageEnhance.Sharpness(image).enhance(1.10)
    out = BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()


def call_plantnet(images: Sequence[Image.Image], organ: str, key: str, timeout: float = 120) -> List[Dict[str, Any]]:
    files = [("images", (f"plant{i}.jpg", preprocess_image(img), "image/jpeg")) for i, img in enumerate(images)]
    try:
        r = httpx.post(
            PLANTNET_ENDPOINT,
            params={"api-key": key, "lang": "en", "nb-results": 5},
            data={"organs": [organ] * len(files)},
            files=files,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise PlantNetError(f"Could not reach PlantNet: {exc.__class__.__name__}") from exc
    if r.status_code == 404:   # PlantNet's answer for "no species found"
        return []
    if r.status_code != 200:
        raise PlantNetError(f"PlantNet returned HTTP {r.status_code}: {r.text[:300]}", r.status_code)
    return r.json().get("results", [])


def _candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    s = item.get("species", {}) or {}
    return {
        "scientific_name": s.get("scientificNameWithoutAuthor") or "Unknown",
        "author": s.get("scientificNameAuthorship") or "",
        "common_names": list(s.get("commonNames") or [])[:5],
        "family": ((s.get("family") or {}).get("scientificNameWithoutAuthor")) or "",
        "genus": ((s.get("genus") or {}).get("scientificNameWithoutAuthor")) or "",
        "score": float(item.get("score", 0) or 0),
    }


def build_result(results: List[Dict[str, Any]], organ: str, n_images: int) -> Dict[str, Any]:
    if not results:
        return {"status": "no_match", "organ": organ, "images": n_images, "best": None, "candidates": [],
                "low_confidence": True, "guide": None}
    candidates = [_candidate(r) for r in results]
    best = candidates[0]
    sci = best["scientific_name"].lower()
    key = next((k for k in GUIDE if k in sci), None)
    if key:
        entry = GUIDE[key]
        local = {lang: {"common_name": n, "use": u, "safety": s} for lang, (n, u, s) in GUIDE_LOCAL.get(key, {}).items()}
        guide = {"in_guide": True, "common_name": entry[0], "use": entry[1], "safety": entry[2], "translations": local}
    else:
        guide = {"in_guide": False, "common_name": ", ".join(best["common_names"][:3]) or "Not provided",
                 "use": NO_GUIDE_USE, "safety": NO_GUIDE_SAFETY}
    return {"status": "ok", "organ": organ, "images": n_images, "best": best, "candidates": candidates,
            "low_confidence": best["score"] < LOW_CONFIDENCE, "guide": guide}


def identify(images: Sequence[Image.Image], organ: str = "auto", key: Optional[str] = None) -> Dict[str, Any]:
    if organ not in ORGANS:
        raise ValueError(f"organ must be one of {', '.join(ORGANS)}")
    if not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError(f"Send 1 to {MAX_IMAGES} photos of the same plant")
    key = key if key is not None else api_key()
    if not key:
        raise PlantNetError("PLANTNET_API_KEY is not set")
    return build_result(call_plantnet(images, organ, key), organ, len(images))
