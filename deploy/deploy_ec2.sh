#!/usr/bin/env bash
# One-command EC2 setup for Vision AI.
# Run on a fresh Ubuntu 24.04 instance as the `ubuntu` user:
#
#   bash deploy/deploy_ec2.sh
#
# Optional environment variables:
#   APP_DIR      install path            (default /home/ubuntu/vision-ai)
#   REPO_URL     git clone URL           (default https://github.com/Arvind-kumar2006/vision-ai.git)
#
# The deployment serves BLIP only: its weights come from Hugging Face and are
# cached here, and the frontend is built with VITE_CAPTION_MODELS=blip. Locally,
# with no such variable, all three caption models remain selectable.
#
# After this finishes, edit $APP_DIR/backend/.env and add your API keys,
# then run: sudo systemctl restart vision-ai
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/vision-ai}"
REPO_URL="${REPO_URL:-https://github.com/Arvind-kumar2006/vision-ai.git}"

echo "==> [1/8] System packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip python3-dev nodejs npm git build-essential

echo "==> [2/8] Clone repository into $APP_DIR"
if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "    already cloned, pulling latest"
  git -C "$APP_DIR" pull --ff-only
fi

echo "==> [3/8] Python environment"
cd "$APP_DIR/backend"
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install --no-deps -r requirements-ocr.txt

echo "==> [4/8] OCR models (~250 MB, SHA-256 verified, downloaded once)"
venv/bin/python scripts/setup_ocr_models.py

echo "==> [5/8] Pre-cache BLIP (the deployed model, ~1 GB from Hugging Face, once)"
# Doing this here means no visitor pays the download on their first caption request.
venv/bin/python - <<'PY'
from transformers import BlipForConditionalGeneration, BlipProcessor
model_id = "Salesforce/blip-image-captioning-base"
BlipProcessor.from_pretrained(model_id)
BlipForConditionalGeneration.from_pretrained(model_id)
print("BLIP cached at ~/.cache/huggingface/hub")
PY

echo "==> [6/8] Build the React app with BLIP as the only selectable model"
cd "$APP_DIR/frontend"
npm install
VITE_CAPTION_MODELS=blip npm run build

echo "==> [7/8] Environment template (add your keys before restarting the service)"
cd "$APP_DIR/backend"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "    created .env from .env.example - EDIT IT and add OPENROUTER_API_KEY, GROQ_API_KEY, PLANTNET_API_KEY"
else
  echo "    .env already exists, leaving it untouched"
fi

echo "==> [8/8] Install and start the systemd service"
sudo cp "$APP_DIR/deploy/vision-ai.service" /etc/systemd/system/vision-ai.service
sudo sed -i "s|/home/ubuntu/vision-ai|$APP_DIR|g" /etc/systemd/system/vision-ai.service
sudo systemctl daemon-reload
sudo systemctl enable vision-ai
sudo systemctl restart vision-ai

echo "==> Health check (BLIP loads from the local cache at startup; allow up to 3 minutes)"
healthy=0
for i in $(seq 1 36); do
  if curl -fsS http://127.0.0.1:8000/api/health > /tmp/vision-ai-health.json 2>/dev/null; then
    healthy=1
    cat /tmp/vision-ai-health.json
    echo ""
    break
  fi
  if ! systemctl is-active --quiet vision-ai; then
    break
  fi
  echo "    waiting for the API ($i/36)..."
  sleep 5
done

if [[ "$healthy" -ne 1 ]]; then
  echo "ERROR: health check failed. Recent logs:" >&2
  journalctl -u vision-ai -n 50 --no-pager >&2
  exit 1
fi

echo ""
echo "DONE. Web UI: http://<EC2-public-IP>:8000   API docs: http://<EC2-public-IP>:8000/docs"
echo "If you just created .env, add your keys then: sudo systemctl restart vision-ai"
echo "Logs: journalctl -u vision-ai -f"
