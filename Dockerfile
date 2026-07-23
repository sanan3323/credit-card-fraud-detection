# Standard container image: serves the FastAPI app with uvicorn.
# Runs locally and on any generic container host (Fly.io, Render, ECS, App Runner).
# For AWS Lambda, see Dockerfile.lambda.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts/fraud_model_artifact.pkl ./artifacts/fraud_model_artifact.pkl

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
