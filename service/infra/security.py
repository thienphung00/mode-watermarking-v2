"""
Security utilities for key generation and encryption.

Provides functions for:
- Generating secure watermark keys
- Encrypting/decrypting secret keys
- Key derivation
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


def generate_watermark_id() -> str:
    """
    Generate a unique watermark identifier.
    
    Returns:
        Watermark ID in format "wm_xxxxx" where xxxxx is a random hex string
    """
    random_bytes = secrets.token_bytes(8)
    hex_str = random_bytes.hex()[:10]
    return f"wm_{hex_str}"


def generate_master_key() -> str:
    """
    Generate a cryptographically secure master key.
    
    Returns:
        32-byte key encoded as hex string (64 characters)
    """
    key_bytes = secrets.token_bytes(32)
    return key_bytes.hex()


def derive_encryption_key(password: str, salt: Optional[bytes] = None) -> bytes:
    """
    Derive an encryption key from a password using PBKDF2.
    
    Args:
        password: Password string
        salt: Optional salt (if None, generates new salt)
    
    Returns:
        32-byte encryption key
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


class KeyEncryption:
    """
    Simple encryption wrapper for storing secret keys.
    
    Uses Fernet (symmetric encryption) for key storage.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize key encryption.
        
        Args:
            encryption_key: Optional encryption key (if None, generates from env or default)
        """
        if encryption_key is None:
            # In production, load from environment variable
            # For now, use a default (NOT SECURE - should be set via env var)
            default_password = "default_service_key_change_in_production"
            encryption_key = derive_encryption_key(default_password)
        
        self.cipher = Fernet(encryption_key)
    
    def encrypt_key(self, master_key: str) -> str:
        """
        Encrypt a master key for storage.
        
        Args:
            master_key: Master key string to encrypt
        
        Returns:
            Encrypted key as base64 string
        """
        return self.cipher.encrypt(master_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt a stored master key.
        
        Args:
            encrypted_key: Encrypted key string
        
        Returns:
            Decrypted master key string
        """
        return self.cipher.decrypt(encrypted_key.encode()).decode()

