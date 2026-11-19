"""Pydantic models for the Product Label Interpreter API (mirrors model/prescription/schemas.py)."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AllergenHit(BaseModel):
    allergen: str
    label: str
    trigger: str  # the exact word that triggered the rule
    source: str   # ingredients | contains_statement | may_contain_statement


class Allergens(BaseModel):
    contains: List[AllergenHit] = Field(default_factory=list)
    may_contain: List[AllergenHit] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    # found | none_detected (ingredients read, nothing found) | unknown (ingredients not read: never "none")
    status: str = "unknown"
    ingredients_read: bool = False
    message: Optional[str] = None


class NutriScoreOut(BaseModel):
    label: str = "Health score (Nutri-Score)"
    grade: Optional[str] = None
    points: Optional[int] = None
    n_points: Optional[int] = None
    p_points: Optional[int] = None
    category: str = "general"
    basis: Optional[str] = None
    breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    algorithm: Optional[str] = None


class ProductLabelResult(BaseModel):
    """Output of image -> preprocessing -> RapidOCR -> parser (AI only for missing values) -> allergens +
    Nutri-Score (ADR-005)."""

    session_id: str
    status: str                      # ok | unreadable (no text found)
    product_name: Optional[str] = None   # typed by the user (optional); never used for values or the score
    message: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    ingredients_text: Optional[str] = None
    nutrition_per_100: Dict[str, float] = Field(default_factory=dict)
    nutrition_basis: Optional[str] = None
    nutrition_per_serving: Dict[str, float] = Field(default_factory=dict)   # shown when per 100 g isn't possible
    nutrition_flags: Dict[str, List[str]] = Field(default_factory=dict)
    field_sources: Dict[str, str] = Field(default_factory=dict)   # read_on_device | ai_from_text
    rejected_ai_values: List[Dict[str, Any]] = Field(default_factory=list)
    allergens: Allergens = Field(default_factory=Allergens)
    additives: List[Dict[str, Any]] = Field(default_factory=list)
    nutriscore: NutriScoreOut = Field(default_factory=NutriScoreOut)
    sanity: Optional[Dict[str, Any]] = None
    sanity_notes: List[str] = Field(default_factory=list)
    photo_notes: List[str] = Field(default_factory=list)
    ai: Optional[Dict[str, Any]] = None        # {"provider", "model", "ms"} when the AI filled values
    ai_note: Optional[str] = None              # "AI busy: details not extracted. Try again in a few minutes."
    transcript: List[str] = Field(default_factory=list)
    # Open-ended facts preserve useful printed label content outside the normalized
    # nutrition fields used by Nutri-Score (for example strength, expiry, batch,
    # warnings, storage, and manufacturer).
    label_type: str = "unknown"              # food | medicine | other | unknown
    label_facts: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None   # llm | template
    explanation_status: str = "pending"        # pending (GET .../explanation) | ready | not_applicable
    explanation_note: Optional[str] = None
    tips: List[str] = Field(default_factory=list)
    disclaimer: str = ""
    preprocessing: List[str] = Field(default_factory=list)
    ocr_engine: str = ""
    timings: Dict[str, int] = Field(default_factory=dict)


class ExplanationResponse(BaseModel):
    session_id: str
    status: str                  # ready
    explanation: str
    source: str                  # llm | template
    note: Optional[str] = None   # e.g. "detailed explanation unavailable right now" when the AI is busy

