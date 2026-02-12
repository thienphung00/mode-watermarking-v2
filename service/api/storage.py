"""
Storage abstraction for images.

Supports:
- Local filesystem storage
- GCS-compatible storage (stub for now)
"""
from __future__ import annotations

import base64
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    async def save_image(
        self,
        image_data: bytes,
        filename: Optional[str] = None,
        content_type: str = "image/png",
    ) -> str:
        """
        Save image and return URL/path.
        
        Args:
            image_data: Raw image bytes
            filename: Optional filename (generated if not provided)
            content_type: MIME type
            
        Returns:
            URL or path to stored image
        """
        pass
    
    @abstractmethod
    async def get_image(self, path: str) -> Optional[bytes]:
        """
        Retrieve image data.
        
        Args:
            path: Path or identifier returned from save_image
            
        Returns:
            Image bytes or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_image(self, path: str) -> bool:
        """
        Delete stored image.
        
        Args:
            path: Path or identifier
            
        Returns:
            True if deleted
        """
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage.
    
    Stores images in a local directory with generated filenames.
    """
    
    def __init__(self, base_path: str = "./data/images"):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for image storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, extension: str = "png") -> str:
        """Generate unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"img_{timestamp}_{unique_id}.{extension}"
    
    async def save_image(
        self,
        image_data: bytes,
        filename: Optional[str] = None,
        content_type: str = "image/png",
    ) -> str:
        """
        Save image to local filesystem.
        
        Returns:
            Filename only (not full path) for URL construction.
        """
        if filename is None:
            ext = content_type.split("/")[-1]
            filename = self._generate_filename(ext)
        
        filepath = self.base_path / filename
        
        try:
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            logger.info(f"Saved image to {filepath}")
            # Return filename only - routes.py will construct the URL
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
    
    async def get_image(self, path: str) -> Optional[bytes]:
        """Retrieve image from local filesystem."""
        filepath = Path(path)
        
        if not filepath.exists():
            # Try relative to base_path
            filepath = self.base_path / path
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return None
    
    async def delete_image(self, path: str) -> bool:
        """Delete image from local filesystem."""
        filepath = Path(path)
        
        if not filepath.exists():
            filepath = self.base_path / path
        
        if not filepath.exists():
            return False
        
        try:
            filepath.unlink()
            logger.info(f"Deleted image: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            return False


class GCSStorage(StorageBackend):
    """
    Google Cloud Storage backend (stub).
    
    Writes images to local storage_path so they can be served at /images/<filename>.
    Returns filename only for URL construction; full GCS upload is stubbed.
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "watermark-images/",
        local_path: Optional[str] = None,
    ):
        """
        Initialize GCS storage.
        
        Args:
            bucket: GCS bucket name
            prefix: Object prefix/folder
            local_path: Local directory to write images so API can serve at /images/
        """
        self.bucket = bucket
        self.prefix = prefix
        self.local_path = Path(local_path) if local_path else None
        if self.local_path:
            self.local_path.mkdir(parents=True, exist_ok=True)
        logger.warning("GCSStorage is a stub - using mock implementation")
    
    def _generate_filename(self, extension: str = "png") -> str:
        """Generate unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"img_{timestamp}_{unique_id}.{extension}"
    
    async def save_image(
        self,
        image_data: bytes,
        filename: Optional[str] = None,
        content_type: str = "image/png",
    ) -> str:
        """Save image locally (so /images/ can serve it) and return filename only."""
        if filename is None:
            ext = content_type.split("/")[-1]
            filename = self._generate_filename(ext)
        # Ensure we return only the basename for URL construction
        filename = os.path.basename(filename)
        
        if self.local_path:
            filepath = self.local_path / filename
            try:
                with open(filepath, "wb") as f:
                    f.write(image_data)
                logger.info(f"Saved image to {filepath} (GCS stub)")
            except Exception as e:
                logger.error(f"Failed to save image locally: {e}")
                raise
        # Stub: would upload to gs://bucket/prefix/filename
        logger.info(f"[STUB] Would save to gs://{self.bucket}/{self.prefix}{filename}")
        return filename
    
    async def get_image(self, path: str) -> Optional[bytes]:
        """Retrieve image from GCS (stub)."""
        logger.warning(f"[STUB] GCS get_image not implemented: {path}")
        return None
    
    async def delete_image(self, path: str) -> bool:
        """Delete image from GCS (stub)."""
        logger.warning(f"[STUB] GCS delete_image not implemented: {path}")
        return False


def get_storage() -> StorageBackend:
    """
    Get storage backend based on configuration.
    
    Returns:
        Configured storage backend instance
    """
    from service.api.config import get_config
    
    config = get_config()
    
    if config.storage_backend == "gcs" and config.gcs_bucket:
        return GCSStorage(
            bucket=config.gcs_bucket,
            local_path=config.storage_path,
        )
    else:
        return LocalStorage(base_path=config.storage_path)
