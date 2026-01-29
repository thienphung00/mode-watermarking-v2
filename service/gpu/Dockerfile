# GPU Worker Dockerfile
# Requires NVIDIA GPU runtime

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY service/gpu/requirements.txt /app/gpu_requirements.txt
RUN pip3 install --no-cache-dir -r /app/gpu_requirements.txt

# Copy source code
COPY src/ /app/src/
COPY service/gpu/ /app/service/gpu/

# Set Python path
ENV PYTHONPATH=/app

# Default environment
ENV GPU_HOST=0.0.0.0 \
    GPU_PORT=8001 \
    MODEL_ID=runwayml/stable-diffusion-v1-5 \
    DEVICE=cuda \
    STUB_MODE=false

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1

# Run the GPU worker
CMD ["python3", "-m", "service.gpu.main"]
