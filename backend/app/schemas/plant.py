"""API models for plant identification."""
from typing import Dict, List, Optional

from pydantic import BaseModel


class PlantCandidate(BaseModel):
    scientific_name: str
    author: str = ""
    common_names: List[str] = []
    family: str = ""
    genus: str = ""
    score: float                      # PlantNet confidence 0-1


class PlantGuide(BaseModel):
    in_guide: bool                    # True only for the few plants in the app's local guide
    common_name: str
    use: str
    safety: str
    translations: Dict[str, Dict[str, str]] = {}   # "hi"/"kn" -> {common_name, use, safety}, hand-written


class PlantResult(BaseModel):
    status: str                       # ok | no_match
    organ: str
    images: int
    best: Optional[PlantCandidate] = None
    candidates: List[PlantCandidate] = []
    low_confidence: bool = True       # best score below 60 %
    guide: Optional[PlantGuide] = None


class PlantUsesRequest(BaseModel):
    scientific_name: str
    common_names: List[str] = []
    lang: str = "en"                  # en | hi | kn


class PlantUse(BaseModel):
    use: str
    detail: str = ""
    evidence: str = "traditional"     # traditional | limited | moderate


class PlantUsesResponse(BaseModel):
    status: str                       # ok | busy (AI limit or error) | unavailable (no AI key)
    scientific_name: str
    known_medicinal: bool = False
    summary: str = ""
    uses: List[PlantUse] = []
    parts_used: List[str] = []
    safety: List[str] = []
    toxic: bool = False
    source: Optional[dict] = None     # {"provider", "model"} that wrote it
    lang: str = "en"                  # language of the text; "en" when a translation wasn't available
    local_name: str = ""              # the plant's name in that language (translations only)
