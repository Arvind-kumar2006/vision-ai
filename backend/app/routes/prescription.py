"""Prescription endpoint: read a photo."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from PIL import Image

from ..dependencies import read_image
from ..schemas.prescription import PrescriptionResult

router = APIRouter(prefix="/api", tags=["prescription"])
log = logging.getLogger("backend.prescription")


@router.post("/prescription", response_model=PrescriptionResult)
def analyze_prescription(request: Request, image: Image.Image = Depends(read_image)) -> PrescriptionResult:
    service = request.app.state.prescription_service
    if not service.ready:
        log.error("Prescription reader unavailable: %s", service.error)
        raise HTTPException(status_code=503, detail="The prescription reader is not available right now. "
                                                    "Check the server logs (OCR models installed?).")
    try:
        return service.analyze(image)
    except Exception:
        log.exception("Prescription analysis failed")   # details only in the server log
        raise HTTPException(status_code=500, detail="Prescription analysis failed.")

