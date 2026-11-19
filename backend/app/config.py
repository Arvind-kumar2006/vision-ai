"""Central configuration. Every value can be overridden with an env var.
Loads configuration from .env.
It contains things like:
- File paths
- Maximum upload size
- Allowed image types
- CORS settings
"""
import os
from dataclasses import dataclass
from pathlib import Path

# backend/app/config.py -> parents[0]=app, [1]=backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Load backend/.env before Settings reads the environment. Optional: the API
# still works without python-dotenv, it just ignores the .env file.
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is optional
    pass


@dataclass(frozen=True)
class Settings:
    # Captioning settings (weights path, beam size, ...) live in the
    # `inference:` section of this YAML file; see app/services/caption_service.py.
    config_path: Path = Path(
        os.getenv("CONFIG_PATH", str(BACKEND_ROOT / "configs.yaml"))
    )

    # Optional: serve the built React app from FastAPI (frontend/dist).
    frontend_dist: Path = Path(
        os.getenv("FRONTEND_DIST", str(BACKEND_ROOT.parent / "frontend" / "dist"))
    )

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    allowed_types: tuple = ("image/jpeg", "image/png", "image/webp")

    # Prescriptions and food labels: OCR models (weights/ocr/, see scripts/setup_ocr_models.py) and the AI
    # (OPENROUTER_API_KEY / OPENROUTER_MODEL, backup GROQ_API_KEY / GROQ_MODEL) are read by
    # model/common/ocr.py and model/common/llm.py directly from the environment.
    cors_origins: tuple = tuple(
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if o.strip()
    )


settings = Settings()
