"""
Pseudorandom Function (PRF) for SynthID-style watermark detection.

Implements cryptographically secure PRF using ChaCha20 for deterministic,
key-indexed seed generation. This replaces XOR-shift PRNG with a proper
cryptographic construction.

Design follows SynthID principles (Nature, 2024):
- Deterministic: same inputs → same outputs
- Key-indexed: different key_id → different seed sequence
- Cryptographically secure: computationally indistinguishable from random
- No metadata dependency: works with only master_key + key_id

Reference:
    Dathathri et al. "Scalable watermarking for identifying large language
    model outputs." Nature 634, 818-823 (2024).
    https://www.nature.com/articles/s41586-024-08025-4
"""
from __future__ import annotations

import hashlib
import struct
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Import PRFConfig from config module
from ..core.config import PRFConfig

# Try to import cryptography library for ChaCha20
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class PRFKeyDerivation:
    """
    Cryptographically secure PRF for SynthID-style watermark key generation.
    
    Generates deterministic seed sequences using ChaCha20 stream cipher.
    Each seed is derived as: seed_i = PRF(master_key, key_id, index=i)
    
    This replaces the XOR-shift PRNG with a proper cryptographic PRF that:
    - Requires no per-image metadata (zT_hash, sample_id, base_seed)
    - Uses only master_key + key_id for detection
    - Provides cryptographic security guarantees
    
    Example:
        >>> prf = PRFKeyDerivation(master_key="secret_master_key_32bytes_long!")
        >>> seeds = prf.generate_seeds(key_id="image_001", count=1000)
        >>> # Same key_id always produces same seeds
        >>> seeds2 = prf.generate_seeds(key_id="image_001", count=1000)
        >>> assert seeds == seeds2
    """
    
    def __init__(
        self,
        master_key: str,
        config: Optional[PRFConfig] = None,
    ):
        """
        Initialize PRF key derivation.
        
        Args:
            master_key: Secret master key (should be high-entropy, kept secret).
                       Will be hashed to derive actual cryptographic key.
            config: PRF configuration (default: ChaCha20 with 64-bit outputs)
        """
        self.config = config or PRFConfig(algorithm="chacha20", output_bits=64)
        self._master_key_raw = master_key
        
        # Derive 256-bit key from master_key using SHA-256
        self._derived_key = self._derive_key(master_key)
        
    def _derive_key(self, master_key: str) -> bytes:
        """
        Derive a 256-bit cryptographic key from the master key.
        
        Uses SHA-256 for key derivation. In production, consider using
        a proper KDF like HKDF or Argon2.
        
        Args:
            master_key: Raw master key string
            
        Returns:
            32-byte derived key suitable for ChaCha20/AES
        """
        # Hash the master key to get exactly 32 bytes
        return hashlib.sha256(master_key.encode("utf-8")).digest()
    
    def _derive_nonce(self, key_id: str) -> bytes:
        """
        Derive a nonce from the key_id.
        
        For ChaCha20: 16-byte nonce
        For AES-CTR: 16-byte nonce/IV
        
        Args:
            key_id: Unique identifier for this watermark key
            
        Returns:
            Nonce bytes (16 bytes)
        """
        # Hash key_id to get nonce - use first 16 bytes of SHA-256
        return hashlib.sha256(key_id.encode("utf-8")).digest()[:16]
    
    def generate_seed(self, key_id: str, index: int) -> int:
        """
        Generate a single PRF seed for the given key_id and index.
        
        Computes: seed = PRF(master_key, key_id, index)
        
        This is the core primitive: deterministic, key-indexed random value.
        
        Args:
            key_id: Unique identifier for this watermark (public, can be stored)
            index: Position index (0, 1, 2, ...) in the seed sequence
            
        Returns:
            64-bit (or 32-bit) pseudo-random integer
        """
        # Get the PRF output stream and extract the specific index
        nonce = self._derive_nonce(key_id)
        
        # Calculate byte offset for this index
        bytes_per_output = self.config.output_bits // 8
        byte_offset = index * bytes_per_output
        
        # Generate PRF output at this offset
        prf_bytes = self._prf_at_offset(nonce, byte_offset, bytes_per_output)
        
        # Convert to integer
        if self.config.output_bits == 64:
            return struct.unpack("<Q", prf_bytes)[0]  # Little-endian uint64
        else:
            return struct.unpack("<I", prf_bytes)[0]  # Little-endian uint32
    
    def generate_seeds(self, key_id: str, count: int) -> List[int]:
        """
        Generate multiple PRF seeds for the given key_id.
        
        More efficient than calling generate_seed() repeatedly as it
        generates all bytes in one stream operation.
        
        Args:
            key_id: Unique identifier for this watermark
            count: Number of seeds to generate
            
        Returns:
            List of pseudo-random integers
        """
        nonce = self._derive_nonce(key_id)
        bytes_per_output = self.config.output_bits // 8
        total_bytes = count * bytes_per_output
        
        # Generate full PRF stream
        prf_stream = self._prf_stream(nonce, total_bytes)
        
        # Parse into integers
        seeds = []
        fmt = "<Q" if self.config.output_bits == 64 else "<I"
        for i in range(count):
            offset = i * bytes_per_output
            seed = struct.unpack(fmt, prf_stream[offset:offset + bytes_per_output])[0]
            seeds.append(seed)
        
        return seeds
    
    def generate_seed_stream(self, key_id: str, count: int) -> Iterator[int]:
        """
        Generate a stream of PRF seeds (iterator interface).
        
        Useful for lazy evaluation when generating large G-fields.
        
        Args:
            key_id: Unique identifier for this watermark
            count: Number of seeds to generate
            
        Yields:
            Pseudo-random integers
        """
        # For simplicity, generate all at once and iterate
        # Could be optimized for very large counts
        seeds = self.generate_seeds(key_id, count)
        yield from seeds
    
    def _prf_stream(self, nonce: bytes, length: int) -> bytes:
        """
        Generate PRF output stream of given length.
        
        Args:
            nonce: 16-byte nonce derived from key_id
            length: Number of bytes to generate
            
        Returns:
            PRF output bytes
        """
        if self.config.algorithm == "chacha20":
            return self._chacha20_stream(nonce, length)
        else:
            return self._aes_ctr_stream(nonce, length)
    
    def _prf_at_offset(self, nonce: bytes, offset: int, length: int) -> bytes:
        """
        Generate PRF output at a specific byte offset.
        
        Args:
            nonce: 16-byte nonce
            offset: Byte offset into the stream
            length: Number of bytes to generate
            
        Returns:
            PRF output bytes at the given offset
        """
        # For simplicity, generate from start
        # Could be optimized with seek capability for large offsets
        total_needed = offset + length
        stream = self._prf_stream(nonce, total_needed)
        return stream[offset:offset + length]
    
    def _chacha20_stream(self, nonce: bytes, length: int) -> bytes:
        """
        Generate ChaCha20 keystream.
        
        ChaCha20 is a stream cipher that produces a deterministic
        pseudo-random byte stream from a key and nonce.
        
        Args:
            nonce: 16-byte nonce (ChaCha20 uses 16-byte nonce)
            length: Number of bytes to generate
            
        Returns:
            Keystream bytes
        """
        if HAS_CRYPTOGRAPHY:
            # Use cryptography library for proper ChaCha20
            cipher = Cipher(
                algorithms.ChaCha20(self._derived_key, nonce),
                mode=None,
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            # Encrypt zeros to get keystream
            return encryptor.update(b"\x00" * length)
        else:
            # Fallback: Use HMAC-based PRF (not as efficient but cryptographically sound)
            return self._hmac_prf_stream(nonce, length)
    
    def _aes_ctr_stream(self, nonce: bytes, length: int) -> bytes:
        """
        Generate AES-CTR keystream.
        
        AES in Counter mode produces a deterministic pseudo-random
        byte stream from a key and nonce/IV.
        
        Args:
            nonce: 16-byte nonce/IV
            length: Number of bytes to generate
            
        Returns:
            Keystream bytes
        """
        if HAS_CRYPTOGRAPHY:
            from cryptography.hazmat.primitives.ciphers import modes
            cipher = Cipher(
                algorithms.AES(self._derived_key),
                modes.CTR(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            return encryptor.update(b"\x00" * length)
        else:
            return self._hmac_prf_stream(nonce, length)
    
    def _hmac_prf_stream(self, nonce: bytes, length: int) -> bytes:
        """
        HMAC-based PRF fallback when cryptography library unavailable.
        
        Uses HMAC-SHA256 in counter mode as a PRF.
        
        Args:
            nonce: Nonce bytes
            length: Number of bytes to generate
            
        Returns:
            PRF output bytes
        """
        import hmac
        
        output = bytearray()
        counter = 0
        
        while len(output) < length:
            # HMAC(key, nonce || counter)
            msg = nonce + struct.pack("<Q", counter)
            h = hmac.new(self._derived_key, msg, hashlib.sha256)
            output.extend(h.digest())
            counter += 1
        
        return bytes(output[:length])
    
    def compute_key_id(self, identifier: str) -> str:
        """
        Compute a public key_id from an identifier.
        
        The key_id is safe to store publicly (e.g., in image metadata)
        and does not reveal the master_key.
        
        Args:
            identifier: Any string identifier (e.g., image filename, UUID)
            
        Returns:
            16-character hex string key_id
        """
        # Hash identifier with a domain separator
        h = hashlib.sha256()
        h.update(b"synthid_keyid_v1|")
        h.update(identifier.encode("utf-8"))
        return h.hexdigest()[:16]


def seeds_to_rademacher(seeds: List[int], bit_pos: int = 30) -> np.ndarray:
    """
    Convert PRF seeds to Rademacher (±1) values.
    
    Extracts a single bit from each seed and maps to ±1.
    This produces a mean-zero, unit-variance distribution.
    
    Args:
        seeds: List of 64-bit integers from PRF
        bit_pos: Which bit to extract (0-63). Default 30 is well-mixed.
        
    Returns:
        NumPy array of ±1 values (float32)
    """
    arr = np.array(seeds, dtype=np.uint64)
    bits = ((arr >> np.uint64(bit_pos)) & np.uint64(1)).astype(np.int32)
    return (2 * bits - 1).astype(np.float32)


def seeds_to_gaussian(seeds: List[int]) -> np.ndarray:
    """
    Convert PRF seeds to Gaussian values using Box-Muller transform.
    
    Produces approximately Gaussian-distributed values from uniform seeds.
    
    Args:
        seeds: List of 64-bit integers from PRF (must be even count)
        
    Returns:
        NumPy array of approximately Gaussian values (float32)
    """
    if len(seeds) % 2 != 0:
        seeds = seeds[:-1]  # Drop last if odd
    
    # Convert to uniform [0, 1)
    arr = np.array(seeds, dtype=np.float64) / (2**64)
    
    # Box-Muller transform
    u1 = arr[0::2]
    u2 = arr[1::2]
    
    # Avoid log(0)
    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    
    # Interleave results
    result = np.empty(len(seeds), dtype=np.float32)
    result[0::2] = z0
    result[1::2] = z1
    
    return result


# ============================================================================
# Convenience Functions
# ============================================================================


def create_prf(master_key: str, algorithm: str = "chacha20") -> PRFKeyDerivation:
    """
    Create a PRF instance with the given master key.
    
    Args:
        master_key: Secret master key
        algorithm: PRF algorithm ("chacha20" or "aes_ctr")
        
    Returns:
        PRFKeyDerivation instance
    """
    from ..core.config import PRFConfig
    config = PRFConfig(algorithm=algorithm, output_bits=64)
    return PRFKeyDerivation(master_key, config)


def generate_prf_seeds(
    master_key: str,
    key_id: str,
    count: int,
    algorithm: str = "chacha20",
) -> List[int]:
    """
    One-shot function to generate PRF seeds.
    
    Args:
        master_key: Secret master key
        key_id: Public key identifier
        count: Number of seeds to generate
        algorithm: PRF algorithm
        
    Returns:
        List of pseudo-random integers
    """
    prf = create_prf(master_key, algorithm)
    return prf.generate_seeds(key_id, count)

