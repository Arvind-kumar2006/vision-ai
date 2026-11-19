"""
CaptionService: one entry point for all captioning models.

    sat   Show, Attend and Tell, ResNet-101 + LSTM, COCO 2014      
    -> 14x14 attention
    catr  CATR, ResNet-101 + transformer, COCO 2017               
    -> H x W attention
    blip  BLIP-base, ViT-B/16 + text decoder, COCO fine-tuned 
    -> 24x24 attention

`configs.yaml` -> `inference.model` picks the default (loaded at startup);
the others load on first use (``?model=`` on the API). Every model returns the
same shape: caption, words, one attention map per word (0-1), grid_shape.
"""
from __future__ import annotations

import logging
import threading

import yaml
from PIL import Image

from ..config import settings
from .captioners.blip import BLIPCaptioner
from .captioners.catr import CATRCaptioner
from .captioners.sat import SATCaptioner

log = logging.getLogger("backend.caption")

ENGINES = {"sat": SATCaptioner, "catr": CATRCaptioner, "blip": BLIPCaptioner}
LABELS = {name: cls.label for name, cls in ENGINES.items()}


def load_inference_cfg() -> dict:
    try:
        with open(settings.config_path, "r") as f:
            return (yaml.safe_load(f) or {}).get("inference", {}) or {}
    except FileNotFoundError:
        return {}


class CaptionService:
    def __init__(self) -> None:
        self.cfg = load_inference_cfg()
        default = str(self.cfg.get("model", "sat")).lower()
        self.default_model = default if default in ENGINES else "sat"
        self._engines: dict = {}
        self._lock = threading.Lock()
        self.get(self.default_model)

    # --- engine registry ------------------------------------------------
    def get(self, name: str | None = None):
        name = (name or self.default_model).lower()
        if name not in ENGINES:
            raise KeyError(name)
        engine = self._engines.get(name)
        if engine is None:
            with self._lock:
                engine = self._engines.get(name)
                if engine is None:
                    log.info("Loading caption model '%s'", name)
                    engine = ENGINES[name](self.cfg)
                    self._engines[name] = engine
        return engine

    def status(self) -> dict:
        """name -> ready | unavailable | not_loaded."""
        out = {}
        for name in ENGINES:
            engine = self._engines.get(name)
            out[name] = "not_loaded" if engine is None else ("ready" if engine.ready else "unavailable")
        return out

    # --- attributes the health route reads (from the default model) -------
    @property
    def _default(self):
        return self._engines[self.default_model]

    @property
    def ready(self) -> bool:
        return self._default.ready

    @property
    def error(self):
        return self._default.error

    @property
    def device(self):
        return self._default.device

    @property
    def vocab_size(self) -> int:
        return self._default.vocab_size

    @property
    def checkpoint_path(self):
        return self._default.checkpoint_path

    # --- inference ----------------------------------------------------------
    def caption(self, image: Image.Image, beam_size: int | None = None,
                max_len: int | None = None, model: str | None = None) -> dict:
        engine = self.get(model)
        if not engine.ready:
            raise RuntimeError(engine.error or f"Caption model '{engine.name}' is not available.")
        return engine.caption(image, beam_size=beam_size, max_len=max_len)
