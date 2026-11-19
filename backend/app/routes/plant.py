"""Plant identification endpoint.

POST /api/plant/identify   1-5 photos of the same plant (+ organ) -> PlantNet candidates and guide context
POST /api/plant/uses       {scientific_name, common_names, lang} -> medicinal uses written by the AI (name only, no photo),
                           in English, Hindi (hi) or Kannada (kn)
"""
import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..dependencies import read_image
from ..schemas.plant import PlantResult, PlantUsesRequest, PlantUsesResponse
from model.plant.identify import MAX_IMAGES, ORGANS, PlantNetError

router = APIRouter(prefix="/api/plant", tags=["plant"])
log = logging.getLogger("backend.plant")


@router.post("/identify", response_model=PlantResult)
def identify(request: Request,
             images: List[UploadFile] = File(..., description=f"1-{MAX_IMAGES} photos of the same plant"),
             organ: str = Form("auto", description=f"What the photos show: {', '.join(ORGANS)}")) -> PlantResult:
    if organ not in ORGANS:
        raise HTTPException(status_code=422, detail=f"organ must be one of: {', '.join(ORGANS)}.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"Send at most {MAX_IMAGES} photos of the same plant.")
    service = request.app.state.plant_service
    if not service.ready:
        raise HTTPException(status_code=503, detail="Plant identification isn't set up (PLANTNET_API_KEY missing).")
    pils = [read_image(f) for f in images]
    try:
        return service.identify(pils, organ)
    except PlantNetError as exc:
        log.warning("PlantNet failed: %s", exc)
        if exc.status == 429:
            raise HTTPException(status_code=429, detail="PlantNet's daily limit is reached. Try again tomorrow.")
        if exc.status in (401, 403):
            raise HTTPException(status_code=502, detail="PlantNet rejected the API key.")
        raise HTTPException(status_code=502, detail="PlantNet couldn't be reached. Please try again.")
    except Exception:
        log.exception("Plant identification failed")
        raise HTTPException(status_code=500, detail="Plant identification failed.")


@router.post("/uses", response_model=PlantUsesResponse)
def uses(body: PlantUsesRequest, request: Request) -> PlantUsesResponse:
    try:
        return request.app.state.plant_service.medicinal_uses(body.scientific_name, body.common_names, body.lang)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        log.exception("Plant uses failed")
        raise HTTPException(status_code=500, detail="Medicinal uses couldn't be loaded.")
