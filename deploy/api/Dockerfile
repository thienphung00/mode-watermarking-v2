# API Service Dockerfile
# Multi-stage build for optimized production image
# CPU-only, horizontally scalable

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (CPU-only torch)
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch for API service
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.10-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash api
USER api
WORKDIR /home/api/app

# Copy virtual environment from builder
COPY --from=builder --chown=api:api /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=api:api service/ ./service/
COPY --chown=api:api src/ ./src/

# Set Python path
ENV PYTHONPATH="/home/api/app:$PYTHONPATH"

# Disable buffering for logs
ENV PYTHONUNBUFFERED=1

# Default environment variables
ENV ENVIRONMENT="production"
ENV INFERENCE_MODE="remote"
ENV ENABLE_DOCS="false"
ENV ENABLE_METRICS="true"
ENV LOG_LEVEL="INFO"
ENV LOG_FORMAT="json"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

