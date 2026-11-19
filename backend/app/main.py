"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routes import caption, health, plant, prescription, product_label
from .services.caption_service import CaptionService
from .services.plant_service import PlantService
from .services.prescription_service import PrescriptionService
from .services.product_label_service import ProductLabelService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.caption_service = CaptionService()
    app.state.prescription_service = PrescriptionService()
    app.state.product_label_service = ProductLabelService()
    app.state.plant_service = PlantService()
    yield


app = FastAPI(
    title="Image Captioning API",
    version="1.0.0",
    description="Show, Attend and Tell captioning (pretrained ResNet-101 / COCO 2014 weights), prescription reader, food labels and plant identification (PlantNet).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(health.root_router)  # GET /health alias
app.include_router(caption.router)
app.include_router(prescription.router)
app.include_router(product_label.router)
app.include_router(plant.router)


# Serve the built React app if `frontend/dist` exists (production mode).
if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
