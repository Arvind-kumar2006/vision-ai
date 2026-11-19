"""PrescriptionService: runs the prescription pipeline, one photo at a time."""
from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

from PIL import Image

from ..config import BACKEND_ROOT

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.prescription.medicine_db import get_index  # noqa: E402
from model.prescription.pipeline import PrescriptionPipeline  # noqa: E402
from model.prescription.schemas import PrescriptionResult  # noqa: E402

log = logging.getLogger("backend.prescription")


class PrescriptionService:
    def __init__(self, pipeline: Optional[PrescriptionPipeline] = None) -> None:
        index = get_index()   # medicine dataset, loaded once (cached index, ~1 s)
        self.medicine_db = f"{len(index.rows)} medicines, {len(index.known)} known brands" if index else "unavailable"
        self.pipeline = pipeline or PrescriptionPipeline()
        self._run_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.pipeline.ocr.available

    @property
    def status(self) -> str:
        return "ready" if self.ready else "unavailable"

    @property
    def error(self) -> Optional[str]:
        return self.pipeline.ocr.error

    def analyze(self, image: Image.Image) -> PrescriptionResult:
        with self._run_lock:   # one OCR run at a time
            return self.pipeline.run(image)
