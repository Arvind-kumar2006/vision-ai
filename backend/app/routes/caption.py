"""Caption endpoint: upload an image, get a caption plus per-word attention maps."""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from PIL import Image

from ..dependencies import read_image
from ..schemas import CaptionResponse

router = APIRouter(prefix="/api", tags=["caption"])
log = logging.getLogger("backend.caption")


@router.post("/caption", response_model=CaptionResponse)
def create_caption(
    request: Request,
    image: Image.Image = Depends(read_image),
    max_len: int | None = Query(None, ge=1, le=60, description="Maximum caption length in words"),
    beam_size: int | None = Query(
        None, ge=1, le=10,
        description="Beam search width (default from configs.yaml inference.beam_size)",
    ),
    model: str | None = Query(
        None, pattern="^(sat|catr|blip)$",
        description="sat = Show-Attend-Tell, catr = CATR transformer, blip = BLIP-base. "
                    "Default: configs.yaml inference.model",
    ),
) -> CaptionResponse:
    service = request.app.state.caption_service
    try:
        engine = service.get(model)
    except Exception:
        log.exception("Could not load caption model %s", model)
        raise HTTPException(status_code=503, detail="Caption model could not be loaded.")
    if not engine.ready:
        log.error("Caption model %s unavailable: %s", engine.name, engine.error)
        raise HTTPException(status_code=503, detail=f"Caption model '{engine.name}' is not available.")

    started = time.perf_counter()
    try:
        result = service.caption(image, beam_size=beam_size, max_len=max_len, model=model)
    except Exception:
        log.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed.")

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return CaptionResponse(mode="caption", inference_ms=elapsed_ms, **result)
