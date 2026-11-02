# ==============================================================================
# Dockerfile — Production Multi-Tier Container Specification
# ==============================================================================
FROM python:3.11-slim

# Set environment variables for Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install system dependencies needed for compiling numerical wheels (LightGBM, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy dataset, application backend, and frontend
COPY data/ ./data/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example ./.env

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose standard container port
EXPOSE 8000

# Docker healthcheck probe using FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn ASGI production server using the port assigned by Render ($PORT)
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"

