"""Shared FastAPI dependencies (image upload validation).
Common image validation.

read_image():
- Checks image type
- Checks file size
- Reads image safely
- Fixes EXIF orientation
- Converts image to RGB
- Creates SHA-256 hash
"""
import hashlib
import io

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings

# Large enough for PrescriptionReader OCR (it works at ~1200 px); captioning uses 256 px.
MAX_DECODE_SIDE = 2048


def read_image(file: UploadFile) -> Image.Image:
    """Validate an uploaded image and return an EXIF-corrected RGB PIL image."""
    if file.content_type not in settings.allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{file.content_type}'. "
                   f"Allowed: {', '.join(settings.allowed_types)}.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {settings.max_upload_mb} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        # JPEG only: decode directly at 1/2, 1/4 or 1/8 scale while keeping both sides
        # >= MAX_DECODE_SIDE. Huge camera photos then decode in a fraction of the time.
        image.draft("RGB", (MAX_DECODE_SIDE, MAX_DECODE_SIDE))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    # Key for the in-memory reader cache: the same photo uploaded again is never re-sent to Groq.
    image.info["upload_sha256"] = hashlib.sha256(data).hexdigest()

    return image
