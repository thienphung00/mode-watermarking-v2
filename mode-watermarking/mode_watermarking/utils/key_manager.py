"""
Key management system for watermarking.

This module handles watermark keys, model IDs, and key-to-model mappings
for secure watermarking operations.
"""

import secrets
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ModelInfo:
    """Information about a watermarking model."""
    model_id: str
    model_name: str
    version: str
    key_slices: List[int]  # Which key slices this model uses
    created_at: str
    description: Optional[str] = None
    is_active: bool = True


@dataclass
class KeyInfo:
    """Information about a watermark key."""
    key_id: str
    key_hash: str  # Hash of the actual key for verification
    created_at: str
    expires_at: Optional[str] = None
    is_active: bool = True
    usage_count: int = 0
    max_usage: Optional[int] = None


class WatermarkKeyManager:
    """
    Manages watermark keys and model mappings.
    
    Provides secure key generation, storage, and retrieval for watermarking
    operations across different models and releases.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize key manager.
        
        Args:
            storage_path: Path to store key information (default: in-memory)
        """
        self.storage_path = storage_path
        self.keys: Dict[str, bytes] = {}  # key_id -> actual_key
        self.key_info: Dict[str, KeyInfo] = {}  # key_id -> KeyInfo
        self.model_info: Dict[str, ModelInfo] = {}  # model_id -> ModelInfo
        self.model_key_mapping: Dict[str, str] = {}  # model_id -> key_id
        
        if storage_path:
            self._load_from_storage()
    
    def generate_key(self, key_id: Optional[str] = None, length: int = 32) -> str:
        """
        Generate a new watermark key.
        
        Args:
            key_id: Optional custom key ID
            length: Key length in bytes
            
        Returns:
            Generated key ID
        """
        if key_id is None:
            key_id = f"key_{secrets.token_hex(8)}"
        
        # Generate cryptographically secure random key
        key_bytes = secrets.token_bytes(length)
        
        # Store the key
        self.keys[key_id] = key_bytes
        
        # Create key info
        key_hash = hashlib.sha256(key_bytes).hexdigest()
        self.key_info[key_id] = KeyInfo(
            key_id=key_id,
            key_hash=key_hash,
            created_at=datetime.now().isoformat(),
            is_active=True,
            usage_count=0
        )
        
        if self.storage_path:
            self._save_to_storage()
        
        return key_id
    
    def get_key(self, key_id: str) -> bytes:
        """
        Retrieve a watermark key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Key bytes
            
        Raises:
            KeyError: If key not found
        """
        if key_id not in self.keys:
            raise KeyError(f"Key {key_id} not found")
        
        # Update usage count
        if key_id in self.key_info:
            self.key_info[key_id].usage_count += 1
        
        return self.keys[key_id]
    
    def register_model(self, model_id: str, model_name: str, version: str, 
                      key_id: str, key_slices: List[int], 
                      description: Optional[str] = None) -> None:
        """
        Register a model with its key mapping.
        
        Args:
            model_id: Unique model identifier
            model_name: Human-readable model name
            version: Model version
            key_id: Associated watermark key ID
            key_slices: Which key slices this model uses
            description: Optional model description
        """
        if key_id not in self.keys:
            raise KeyError(f"Key {key_id} not found")
        
        # Create model info
        self.model_info[model_id] = ModelInfo(
            model_id=model_id,
            model_name=model_name,
            version=version,
            key_slices=key_slices,
            created_at=datetime.now().isoformat(),
            description=description,
            is_active=True
        )
        
        # Map model to key
        self.model_key_mapping[model_id] = key_id
        
        if self.storage_path:
            self._save_to_storage()
    
    def get_model_key(self, model_id: str) -> bytes:
        """
        Get the watermark key for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Key bytes for the model
            
        Raises:
            KeyError: If model or key not found
        """
        if model_id not in self.model_key_mapping:
            raise KeyError(f"Model {model_id} not found")
        
        key_id = self.model_key_mapping[model_id]
        return self.get_key(key_id)
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information
            
        Raises:
            KeyError: If model not found
        """
        if model_id not in self.model_info:
            raise KeyError(f"Model {model_id} not found")
        
        return self.model_info[model_id]
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.model_info.values())
    
    def list_keys(self) -> List[KeyInfo]:
        """List all watermark keys."""
        return list(self.key_info.values())
    
    def deactivate_key(self, key_id: str) -> None:
        """Deactivate a watermark key."""
        if key_id in self.key_info:
            self.key_info[key_id].is_active = False
            if self.storage_path:
                self._save_to_storage()
    
    def deactivate_model(self, model_id: str) -> None:
        """Deactivate a model."""
        if model_id in self.model_info:
            self.model_info[model_id].is_active = False
            if self.storage_path:
                self._save_to_storage()
    
    def _save_to_storage(self) -> None:
        """Save key and model information to storage."""
        if not self.storage_path:
            return
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Save metadata (not the actual keys for security)
        metadata = {
            "key_info": {k: asdict(v) for k, v in self.key_info.items()},
            "model_info": {k: asdict(v) for k, v in self.model_info.items()},
            "model_key_mapping": self.model_key_mapping
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_from_storage(self) -> None:
        """Load key and model information from storage."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        with open(self.storage_path, 'r') as f:
            metadata = json.load(f)
        
        # Load metadata
        self.key_info = {
            k: KeyInfo(**v) for k, v in metadata.get("key_info", {}).items()
        }
        self.model_info = {
            k: ModelInfo(**v) for k, v in metadata.get("model_info", {}).items()
        }
        self.model_key_mapping = metadata.get("model_key_mapping", {})


def validate_watermark_key(key: Union[str, bytes]) -> bytes:
    """
    Validate and normalize a watermark key.
    
    Args:
        key: Key as string or bytes
        
    Returns:
        Normalized key bytes
        
    Raises:
        ValueError: If key is invalid
    """
    if isinstance(key, str):
        # Try to decode as hex first, then as UTF-8
        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            key_bytes = key.encode('utf-8')
    elif isinstance(key, bytes):
        key_bytes = key
    else:
        raise ValueError("Key must be string or bytes")
    
    if len(key_bytes) < 16:
        raise ValueError("Key must be at least 16 bytes")
    
    return key_bytes


def compute_key_hash(key: bytes) -> str:
    """
    Compute SHA256 hash of a key.
    
    Args:
        key: Key bytes
        
    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(key).hexdigest()