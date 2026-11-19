# Backend image (Linux). OCR models are downloaded and checksum-verified at BUILD time by
# scripts/setup_ocr_models.py, never at request time. On Linux requirements.txt installs the headless OpenCV
# (no GUI libraries needed), which is also what rapidocr expects.
#   docker build -t vision-ai-backend backend/
#   docker run -p 8000:8000 --env-file backend/.env vision-ai-backend
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# onnxruntime needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ocr.txt ./
# CPU-only PyTorch wheels keep the image small; captioning runs on CPU in the container.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
 && pip install --no-deps -r requirements-ocr.txt

COPY . .
RUN python scripts/setup_ocr_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
