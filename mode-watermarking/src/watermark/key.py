"""
Key derivation utilities for LCG-based watermarking.

Derives per-sample seeds and public key identifiers without exposing secrets.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class KeyDerivation:
	"""Derive deterministic seeds and key streams from a master key and metadata."""

	def _hmac_sha256(self, key: str, msg: str) -> int:
		"""Return 64-bit integer from HMAC-like SHA256 (concatenation-style)."""
		h = hashlib.sha256()
		h.update(key.encode("utf-8"))
		h.update(b"|")
		h.update(msg.encode("utf-8"))
		return int.from_bytes(h.digest()[:8], byteorder="big", signed=False)

	def compute_key_id(self, key_master: str, experiment_id: str, sample_id: str) -> str:
		"""Public identifier for key; safe to store in manifests."""
		digest = hashlib.sha256(
			(f"{experiment_id}|{sample_id}" + key_master[:8]).encode("utf-8")
		).hexdigest()
		return digest[:16]

	def derive_seed(
		self,
		key_master: str,
		sample_id: str,
		zT_hash: str,
		base_seed: int,
		experiment_id: str,
	) -> int:
		"""Derive initial 64-bit seed combining key and sample metadata."""
		mix = f"{experiment_id}|{sample_id}|{zT_hash}|{base_seed}"
		seed64 = self._hmac_sha256(key_master, mix)
		return seed64 & 0xFFFFFFFFFFFFFFFF

	def generate_key_stream(self, seed0: int, stream_len: int) -> Iterable[int]:
		"""Yield a simple key material stream (here, xorshift64* for speed)."""
		x = seed0 & 0xFFFFFFFFFFFFFFFF
		for _ in range(stream_len):
			x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
			x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
			x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
			y = (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
			yield y

