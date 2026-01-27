# GPU Worker Dockerfile
# Multi-stage build for optimized production image
# GPU-enabled, auto-scalable

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

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as runtime

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash worker
USER worker
WORKDIR /home/worker/app

# Copy virtual environment from builder
COPY --from=builder --chown=worker:worker /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=worker:worker service/ ./service/
COPY --chown=worker:worker src/ ./src/

# Set Python path
ENV PYTHONPATH="/home/worker/app:$PYTHONPATH"

# HuggingFace cache directory
ENV HF_HOME="/home/worker/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/home/worker/.cache/huggingface"

# Disable buffering for logs
ENV PYTHONUNBUFFERED=1

# Default environment variables
ENV MODEL_ID="runwayml/stable-diffusion-v1-5"
ENV DEVICE="cuda"
ENV USE_FP16="true"
ENV MAX_CONCURRENT_REQUESTS="4"
ENV MAX_QUEUE_SIZE="16"
ENV GPU_SEMAPHORE_SIZE="2"
ENV ENABLE_WARMUP="true"
ENV ENABLE_DOCS="false"
ENV LOG_LEVEL="INFO"
ENV LOG_FORMAT="json"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

# Expose port
EXPOSE 8080

# Run with uvicorn (single worker due to GPU)
CMD ["uvicorn", "service.worker.main:app", "--host", "0.0.0.0", "--port", "8080"]

