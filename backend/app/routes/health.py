"""Health / readiness endpoint."""
from fastapi import APIRouter, Request

from ..schemas import HealthResponse
from model.common.llm import default_llm
from ..services.caption_service import LABELS

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.caption_service
    prescription = getattr(request.app.state, "prescription_service", None)
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        device=str(service.device),
        vocab_size=service.vocab_size,
        checkpoint=str(service.checkpoint_path),
        caption_model=service.default_model,
        caption_models=service.status(),
        caption_model_labels=LABELS,
        modes={
            "caption": "ready" if service.ready else "unavailable",
            "prescription": prescription.status if prescription else "unavailable",
            "medicine_db": getattr(prescription, "medicine_db", "unavailable"),
            "product_label": getattr(getattr(request.app.state, "product_label_service", None), "status", "unavailable"),
            "plant": getattr(getattr(request.app.state, "plant_service", None), "status", "unavailable"),
        },
        ai={"providers": default_llm().describe()},   # provider order and models (never keys)
    )


# Alias without the /api prefix: some tools (and the reference app) poll GET /health.
root_router = APIRouter(tags=["health"])


@root_router.get("/health", response_model=HealthResponse, include_in_schema=False)
def health_alias(request: Request) -> HealthResponse:
    return health(request)
