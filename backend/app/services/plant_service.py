"""PlantService: PlantNet identification (needs PLANTNET_API_KEY) and AI-written medicinal uses (model.plant.uses)."""
from __future__ import annotations

import sys
from typing import Any, Dict, Sequence

from ..config import BACKEND_ROOT

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.plant import identify as plant  # noqa: E402
from model.plant.uses import PlantUses  # noqa: E402


class PlantService:
    def __init__(self, uses: PlantUses | None = None) -> None:
        self.uses = uses or PlantUses()

    @property
    def ready(self) -> bool:
        return bool(plant.api_key())

    @property
    def status(self) -> str:
        return "ready" if self.ready else "unavailable"

    def identify(self, images: Sequence, organ: str) -> Dict[str, Any]:
        return plant.identify(images, organ)

    def medicinal_uses(self, scientific_name: str, common_names: Sequence[str] = (), lang: str = "en") -> Dict[str, Any]:
        return self.uses.get(scientific_name, common_names, lang)
