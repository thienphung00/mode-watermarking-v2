"""
GPU Worker for watermarking.

Internal endpoints (not user-facing):
- POST /infer/generate - Generate watermarked image
- POST /infer/reverse_ddim - DDIM inversion for detection
- GET /health - Health check
"""
