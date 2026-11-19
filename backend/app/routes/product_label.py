"""Food label endpoints.

POST /api/product-label/interpret                 label photo (+ optional product_name) -> result (no AI for values that were read)
GET  /api/product-label/{session_id}/explanation  explanation, fetched after the results
"""
import logging

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..dependencies import read_image
from ..schemas.product_label import ExplanationResponse, ProductLabelResult

router = APIRouter(prefix="/api/product-label", tags=["product-label"])
log = logging.getLogger("backend.product_label")


@router.post("/interpret", response_model=ProductLabelResult)
def interpret(request: Request, image: UploadFile = File(..., description="Label photo (ingredients + nutrition table)"),
              product_name: Optional[str] = Form(None, description="Optional product name typed by the user")
              ) -> ProductLabelResult:
    pil = read_image(image)
    service = request.app.state.product_label_service
    if not service.ready:
        raise HTTPException(status_code=503, detail="The label reader is not available right now (OCR models installed?).")
    try:
        return service.interpret(pil, product_name)
    except Exception:
        log.exception("Product label interpretation failed")
        raise HTTPException(status_code=500, detail="Product label interpretation failed.")


@router.get("/{session_id}/explanation", response_model=ExplanationResponse)
def explanation(session_id: str, request: Request) -> ExplanationResponse:
    try:
        r = request.app.state.product_label_service.explanation(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found or expired. Check the label again.")
    except Exception:
        log.exception("Product label explanation failed")
        raise HTTPException(status_code=500, detail="Explanation failed.")
    return ExplanationResponse(session_id=session_id, status=r["explanation_status"], explanation=r["explanation"],
                               source=r["explanation_source"], note=r.get("explanation_note"))

