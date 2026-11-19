"""Pydantic models shared by the perception pipeline and the API layer."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    """One line read by the OCR, with its confidence. sent_to_ai=False: kept on this machine (header,
    patient details) or not part of the medicine section."""

    text: str
    confidence: Optional[float] = None
    sent_to_ai: bool = False


class MedicineCandidate(BaseModel):
    """A medicine from the database that the extracted name may refer to."""

    name: str
    score: float  # brand similarity 0-100 (decides the status)
    variant_score: Optional[float] = None  # score incl. strength/form bonus (picks the variant)
    manufacturer: Optional[str] = None
    composition: Optional[str] = None
    pack_size: Optional[str] = None
    what_for: Optional[str] = None          # from the ingredient lookup, shown after the person picks this one
    what_for_source: Optional[str] = None


class InteractionItem(BaseModel):
    drug: str
    effect: str = ""


class MedicineInfo(BaseModel):
    """Facts copied from the Indian Medicine Dataset. Empty fields are left out."""

    source: str = "From medicine database"
    composition: Optional[str] = None
    type: Optional[str] = None
    manufacturer: Optional[str] = None
    pack_size: Optional[str] = None
    description: Optional[str] = None
    side_effects: Optional[List[str]] = None
    interactions: Optional[List[InteractionItem]] = None
    variant_note: Optional[str] = None  # e.g. strength not written, several products


class KnownBrandInfo(BaseModel):
    name: str
    composition: Optional[str] = None
    source: str = "Known brand list (no database details)"


class MedicineItem(BaseModel):
    """One medicine parsed from a prescription."""

    name: str
    # Combined 0-1: reader confidence x match score/100 x status weight; capped at 0.30
    # when the name is not in the text that was read (grounding).
    confidence: float = 0.0
    reader_confidence: float = 0.0  # what the AI reported for the name (before matching)
    read_as: str = ""  # exactly what the reader returned; never overwritten
    suggested_name: Optional[str] = None  # database name, when it differs from read_as
    match_status: str = "unmatched"  # matched | verify | known_brand | unmatched
    non_medicine: bool = False       # oral-care product or device ("Colgate Plax", "toothbrush"): not a medicine
    match_score: Optional[float] = None  # brand similarity 0-100
    candidates: List[MedicineCandidate] = Field(default_factory=list)  # top 3, distinct brands first
    db_info: Optional[MedicineInfo] = None  # "From medicine database"
    known_brand: Optional[KnownBrandInfo] = None
    # False = the LLM returned a name that does not appear (even approximately)
    # in the OCR text, i.e. it may have been invented. Always review these.
    in_ocr_text: bool = True
    strength: Optional[str] = None  # e.g. "75mg"
    form: Optional[str] = None      # as written: tablet, capsule, syrup, gum paint, ...
    # "Dothepin / Prothiaden 75mg" is one medicine with alternative brand names, matched together
    alternatives: List[str] = Field(default_factory=list)
    alternative_matches: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    as_written: Optional[str] = None  # the medicine's line as written / read
    checks: List[str] = Field(default_factory=list)   # "name" / "strength": check on the paper (hard to read)
    # "What it's for": database text by ingredient (250k Medicines dataset, CC BY-SA 4.0; fallback Indian
    # Medicine Dataset). None -> "No description available. Ask your pharmacist what this medicine is for."
    what_for: Optional[str] = None
    what_for_source: Optional[str] = None
    # High / Medium / Low from match score, strength and variant agreement, OCR confidence (medicine_db.confidence_level)
    confidence_level: str = "medium"
    confidence_factors: List[Dict[str, object]] = Field(default_factory=list)
    ocr_confidence: Optional[float] = None
    variant_mismatch: bool = False
    strength_variants: Optional[int] = None   # distinct strengths of the matched brand in the dataset (None: unknown)
    written_strength_in_brand: Optional[bool] = None   # False: no product of the brand has the written strength
    rival_score: Optional[float] = None       # best DIFFERENT brand's name similarity (close = conflicting candidate)
    other_uses: List[str] = Field(default_factory=list)
    common_side_effects: List[str] = Field(default_factory=list)
    drug_class: Optional[str] = None


class PrescriptionResult(BaseModel):
    """Output of image -> preprocessing -> RapidOCR -> redaction -> AI -> medicine dataset (ADR-005)."""

    id: str
    status: str = "ok"            # ok | unreadable (no text) | no_medicines (text, but no medicine lines) | ai_busy
    message: Optional[str] = None
    medicines: List[MedicineItem] = Field(default_factory=list)
    already_taking: List[str] = Field(default_factory=list)   # "On Tab Ecosprin AV": not part of this prescription
    review_notes: List[str] = Field(default_factory=list)    # plain words for the person checking the result
    ai: Optional[Dict[str, object]] = None                   # {"provider", "model", "ms"} when the AI was used
    ocr_lines: List[OcrLine] = Field(default_factory=list)
    lines_sent_to_ai: int = 0
    lines_kept_private: int = 0
    timings: Dict[str, int] = Field(default_factory=dict)    # preprocess_ms, ocr_ms, ai_ms, match_ms, total_ms
    preprocessing: List[str] = Field(default_factory=list)
    ocr_engine: str = ""
