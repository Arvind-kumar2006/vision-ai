"""API response models: health and captioning here; prescription.py and product_label.py re-export the feature models."""
from typing import Any, Dict, List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str
    vocab_size: int
    checkpoint: str
    modes: Dict[str, str]
    caption_model: str = "sat"                      # default captioning model
    caption_models: Dict[str, str] = {}             # name -> ready | unavailable | not_loaded
    caption_model_labels: Dict[str, str] = {}
    ai: Dict[str, Any] = {}                         # AI providers and models in order (no keys)


class CaptionResponse(BaseModel):
    mode: str = "caption"
    caption: str
    inference_ms: int  # wall-clock time of the whole request handler
    model: str = "sat"  # which captioning model produced this caption
    # Attention: one grid_shape[0] x grid_shape[1] map per word, scaled 0-1.
    words: List[str] = []
    alphas: List[List[List[float]]] = []
    grid_size: int = 0  # rows (kept for older clients; maps may be non-square, see grid_shape)
    grid_shape: List[int] = [0, 0]
    generation_time_ms: float = 0.0  # preprocessing + encoder + decoding
