"""API models for the Product Label Interpreter (canonical models in model.product_label.schemas)."""
import sys

from ..config import BACKEND_ROOT

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.product_label.schemas import ExplanationResponse, ProductLabelResult  # noqa: E402

__all__ = ["ExplanationResponse", "ProductLabelResult"]
