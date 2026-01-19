"""
Middleware for rate limiting and security.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    In production, use Redis or similar for distributed rate limiting.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier (IP address)
        
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True


# Global rate limiters
_detect_rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
_evaluate_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Applies rate limiting to /detect endpoint.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Apply rate limiting to /detect endpoint
        if request.url.path.startswith("/api/v1/detect"):
            if not _detect_rate_limiter.is_allowed(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        # Apply rate limiting to /evaluate endpoint (evaluation-only, stricter limits)
        if request.url.path.startswith("/api/v1/evaluate"):
            if not _evaluate_rate_limiter.is_allowed(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        response = await call_next(request)
        return response

