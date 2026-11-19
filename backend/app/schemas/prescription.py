"""API response models for the perception/prescription feature.

Kept separate from ``app/schemas/__init__.py`` so the captioning schemas stay
untouched. The canonical models live in ``model.prescription.schemas``.
"""
import sys

from ..config import BACKEND_ROOT

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.prescription.schemas import MedicineItem, OcrLine, PrescriptionResult  # noqa: E402

__all__ = ["MedicineItem", "OcrLine", "PrescriptionResult"]
