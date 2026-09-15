# SynFC engine — container image for Cloud Run (or any Docker host).
# Built from the repo root so the image can include the sibling
# synfc_engine/ + Tools/ + Sub_Teams/ + data/ folders the engine needs
# at import time (see synfc_engine/engine.py's sys.path bootstrap).
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds
# that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080) and expects the container
# to bind 0.0.0.0 -- same convention server.py already uses for Render.
ENV PORT=8080
EXPOSE 8080

WORKDIR /app/synfc_engine
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
