# Stage 12: pack the Stage 11 API (app.py + best.pt) into an image.
# Every instruction below = one layer (one tray). Docker reuses unchanged trays from the bottom up.

# ---------- Layer 1: tiny Linux + Python 3.12 (same 3.12 as .venv) ----------
FROM python:3.12-slim

# no .pyc files, print logs immediately, no pip download cache in the image, no pip update nag
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# everything below lives in /app inside the box
WORKDIR /app

# ---------- Layer 2: packages (slow, changes rarely -> cached) ----------
# copy ONLY the shopping list first, so editing app.py does not re-download torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Layer 3: our code (changes often -> near the top) ----------
# exactly the files app.py imports, directly or through each other (attention_viz.py is not one)
COPY app.py translate.py model_training.py model.py encoder.py decoder.py multi_head.py attention.py embeddings.py data.py ./

# option A: the trained model travels inside the image (epoch 18, 5.4 MB)
COPY checkpoints/best.pt checkpoints/best.pt

# ---------- Settings ----------
# get_model_path() in app.py reads MODEL_PATH; PORT is only a default (Cloud Run sets its own)
ENV MODEL_PATH=/app/checkpoints/best.pt \
    PORT=8080

# a label for humans: "this box serves on 8080". It does not open anything by itself
EXPOSE 8080

# ---------- Health: Docker knocks on /health every 30s ----------
# slim has no curl, so Python does the knock; any error -> exit 1 -> counts as a failed knock
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)" || exit 1

# ---------- What runs when a container starts ----------
# shell form (a plain string) so ${PORT} is filled in at start time;
# exec = uvicorn replaces the shell, so "docker stop" reaches uvicorn and it shuts down cleanly
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}