#!/usr/bin/env python3
"""
Train Bayesian likelihood models for g-values.

This script trains two likelihood models:
- P(g | watermarked)
- P(g | unwatermarked)

Models are simple Bernoulli/logistic models with per-position biases.
Learned parameters are saved to disk for use by the Bayesian detector.

Inputs:
- Precomputed inverted g-values (created by precompute_inverted_g_values.py)
- Corresponding labels: watermarked / unwatermarked
- Optional masks (from precomputation)

This script only loads precomputed g-values - no DDIM inversion, no image loading.
For Tier-2 training, use precompute_inverted_g_values.py first to precompute g-values
from images using DDIM inversion.

Usage:
    # Step 1: Precompute inverted g-values
    python scripts/precompute_inverted_g_values.py \
        --manifest path/to/train_manifest.jsonl \
        --output-dir path/to/precomputed_g_values \
        --config-path configs/experiments/seedbias.yaml \
        --master-key "your_secret_key" \
        --num-inversion-steps 20
    
    # Step 2: Train likelihood models
    python scripts/train_g_likelihoods.py \
        --g-manifest path/to/precomputed_g_values/g_manifest.jsonl \
        --output-dir outputs/likelihood_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GValueLikelihoodModel(nn.Module):
    """
    Simple likelihood model for g-values.
    
    Models P(g | class) as independent Bernoulli per position:
        P(g_i = 1 | class) = sigmoid(bias_i)
    
    For unwatermarked: bias_i ≈ 0 (P ≈ 0.5)
    For watermarked: bias_i can be learned (P can deviate from 0.5)
    
    Note: This model assumes a single global mask geometry. The model is only
    valid for detectors using the same mask pattern. G-values passed to forward()
    should already be masked (only valid positions).
    """
    
    def __init__(self, num_positions: int):
        """
        Initialize likelihood model.
        
        Args:
            num_positions: Number of g-value positions (N)
        """
        super().__init__()
        # Per-position bias parameters (logits)
        # Initialize near 0 (P ≈ 0.5 for unwatermarked)
        self.biases = nn.Parameter(torch.zeros(num_positions))
    
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood log P(g | class).
        
        Args:
            g: Binary g-values [B, N_eff] with values in {0, 1}
               Must already be masked (only valid positions)
        
        Returns:
            Log-likelihood [B]
        """
        B, N_eff = g.shape
        
        # Strict shape validation: g must match model size
        if N_eff != len(self.biases):
            raise ValueError(
                f"G-values length {N_eff} does not match model positions {len(self.biases)}"
            )
        
        # Get per-position probabilities
        probs = torch.sigmoid(self.biases)  # [N_eff]
        
        # Expand to batch
        probs = probs.unsqueeze(0).expand(B, -1)  # [B, N_eff]
        
        # Compute log-likelihood per position
        # log P(g_i | class) = g_i * log(p_i) + (1 - g_i) * log(1 - p_i)
        log_probs = g * torch.log(probs + 1e-10) + (1 - g) * torch.log(1 - probs + 1e-10)
        
        # Sum over positions (all positions are valid since g is already masked)
        log_likelihood = log_probs.sum(dim=1)  # [B]
        
        return log_likelihood
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get learned parameters as numpy arrays.
        
        Returns:
            Dictionary with 'biases' (logits) and 'probs' (probabilities)
        """
        with torch.no_grad():
            biases = self.biases.cpu().numpy()
            probs = torch.sigmoid(self.biases).cpu().numpy()
        
        return {
            "biases": biases.tolist(),
            "probs": probs.tolist(),
        }


class GValueNumpyDataset(Dataset):
    """
    Dataset for loading g-values from numpy arrays.
    
    Used for per-family training from Phase 1 exports.
    """
    
    def __init__(
        self,
        g_wm_path: Path,
        g_clean_path: Path,
        mask_path: Optional[Path] = None,
    ):
        """
        Initialize dataset from numpy arrays.
        
        Args:
            g_wm_path: Path to g_wm.npy [num_samples, N_eff]
            g_clean_path: Path to g_clean.npy [num_samples, N_eff]
            mask_path: Optional path to mask.npy [1, N_eff] or [num_samples, N_eff]
        """
        self.g_wm = np.load(g_wm_path)  # [num_samples, N_eff]
        self.g_clean = np.load(g_clean_path)  # [num_samples, N_eff]
        
        # Load mask if provided
        if mask_path is not None and mask_path.exists():
            mask = np.load(mask_path)
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]  # [N_eff]
            self.mask = mask
        else:
            self.mask = None
        
        # Combine into single dataset
        self.g_all = np.concatenate([self.g_wm, self.g_clean], axis=0)  # [2*num_samples, N_eff]
        self.labels = np.concatenate([
            np.ones(len(self.g_wm), dtype=np.int64),
            np.zeros(len(self.g_clean), dtype=np.int64),
        ])
    
    def __len__(self) -> int:
        return len(self.g_all)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Dictionary with:
                - 'g': g-values tensor [N_eff] (binary {0, 1})
                - 'label': binary label (1 for watermarked, 0 for unwatermarked)
                - 'mask': optional mask [N_eff] (if present)
        """
        g = torch.from_numpy(self.g_all[idx]).float()  # [N_eff]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        result = {
            "g": g,
            "label": label,
        }
        
        if self.mask is not None:
            mask = torch.from_numpy(self.mask).float()  # [N_eff]
            result["mask"] = mask
        
        return result


class GValueDataset(Dataset):
    """
    Dataset for loading precomputed g-values and labels.
    
    This dataset only loads precomputed g-values from disk.
    DDIM inversion should be performed separately using precompute_inverted_g_values.py.
    """
    
    def __init__(
        self,
        manifest_path: Path,
        g_key: str = "g_path",
        label_key: str = "label",
    ):
        """
        Initialize dataset.
        
        Args:
            manifest_path: Path to manifest.jsonl file
            g_key: Key in manifest for g-values path
            label_key: Key in manifest for label
        """
        self.manifest_path = manifest_path
        self.g_key = g_key
        self.label_key = label_key
        self.samples = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest.jsonl file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        samples = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Loads precomputed g-values from disk (no DDIM inversion).
        G-values should be precomputed using precompute_inverted_g_values.py.
        
        Returns:
            Dictionary with:
                - 'g': g-values tensor [N] (binary {0, 1}, binarized for Bernoulli likelihood)
                - 'label': binary label (1 for watermarked, 0 for unwatermarked)
                - 'mask': optional mask [N] (if present in manifest)
        """
        sample = self.samples[idx]
        
        # Load precomputed g-values from disk
        g_path_str = sample.get(self.g_key) or sample.get("g_path")
        if not g_path_str:
            raise ValueError(f"Sample {idx} missing '{self.g_key}' field")
        
        g_path = Path(g_path_str)
        if not g_path.is_absolute():
            g_path = self.manifest_path.parent / g_path
        
        if not g_path.exists():
            raise FileNotFoundError(f"G-values file not found: {g_path}")
        
        # Load g-values
        g_data = torch.load(g_path, map_location="cpu")
        
        # Extract g tensor
        if isinstance(g_data, dict):
            g = g_data.get("g", g_data.get("g_values"))
            if g is None:
                raise ValueError(f"G-values file {g_path} missing 'g' key")
        else:
            g = g_data
        
        # Ensure 1D
        if g.dim() == 0:
            raise ValueError(f"G-values must be 1D, got scalar")
        if g.dim() > 1:
            g = g.flatten()
        if g.dim() == 1:
            pass  # Already 1D, which is correct
        
        # Convert to float if needed
        if g.dtype in (torch.long, torch.int64):
            g = g.float()
        
        # Normalize to {0, 1} if needed (handle -1, 1 format)
        unique_vals = torch.unique(g)
        if set(unique_vals.cpu().tolist()).issubset({-1.0, 1.0}):
            g = (g + 1) / 2
        
        # Ensure binary {0, 1} format
        g = torch.clamp(torch.round(g), 0, 1)
        
        # CRITICAL: Binarize g-values for Bernoulli likelihood model
        # This ensures consistency with detection
        g = (g > 0).float()
        
        # Extract label
        label = self._extract_label(sample)
        
        # Extract mask if present (from precomputed g-values)
        mask = None
        if "mask_path" in sample:
            mask_path = sample["mask_path"]
            if isinstance(mask_path, str):
                mask_path = Path(mask_path)
                if not mask_path.is_absolute():
                    mask_path = self.manifest_path.parent / mask_path
                if mask_path.exists():
                    mask = torch.load(mask_path, map_location="cpu")
                    # Ensure mask is 1D and matches g shape
                    if mask.dim() > 1:
                        mask = mask.flatten()
                    if mask.dim() == 0:
                        raise ValueError(f"Mask must be 1D, got scalar")
                    # Ensure mask matches g length
                    if mask.shape[0] != g.shape[-1]:
                        raise ValueError(
                            f"Mask length {mask.shape[0]} does not match g length {g.shape[-1]}"
                        )
                    # Convert to binary {0, 1}
                    mask = (mask > 0.5).float()
        
        result = {
            "g": g,
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        if mask is not None:
            result["mask"] = mask
        
        return result
    
    def _extract_label(self, sample: Dict) -> int:
        """Extract binary label from sample."""
        label = sample.get(self.label_key) or sample.get("is_watermarked") or sample.get("watermarked")
        
        if label is None:
            return 0
        
        # Convert to int
        if isinstance(label, bool):
            return 1 if label else 0
        elif isinstance(label, str):
            label_lower = label.lower()
            if label_lower in ("true", "1", "watermarked", "yes"):
                return 1
            elif label_lower in ("false", "0", "unwatermarked", "no"):
                return 0
            else:
                return int(label)
        else:
            return int(label)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching g-values.
    
    Note: Masks are kept as-is for selection-based masking in training loop.
    Padding is applied for compatibility, but is not strictly necessary when
    using consistent global masks (masking is applied via tensor slicing).
    """
    # Find max length
    max_len = max(item["g"].shape[-1] for item in batch)
    
    g_list = []
    labels = []
    masks = []
    
    for item in batch:
        g = item["g"]
        if g.dim() == 1:
            g = g.unsqueeze(0)  # [1, N]
        
        # Pad to max_len
        if g.shape[-1] < max_len:
            padding = torch.zeros(g.shape[0], max_len - g.shape[-1], dtype=g.dtype)
            g = torch.cat([g, padding], dim=-1)
        
        g_list.append(g)
        labels.append(item["label"])
        
        if "mask" in item:
            mask = item["mask"]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[-1] < max_len:
                padding = torch.zeros(mask.shape[0], max_len - mask.shape[-1], dtype=mask.dtype)
                mask = torch.cat([mask, padding], dim=-1)
            masks.append(mask)
    
    # Stack
    g_batch = torch.cat(g_list, dim=0)  # [B, max_len]
    labels_batch = torch.stack(labels)  # [B]
    
    result = {
        "g": g_batch,
        "label": labels_batch,
    }
    
    if masks:
        mask_batch = torch.cat(masks, dim=0)  # [B, max_len]
        result["mask"] = mask_batch
    
    return result


def train_likelihood_models(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_positions: int,
    global_mask: Optional[torch.Tensor] = None,
    num_epochs: int = 10,
    lr: float = 0.01,
    device: str = "cpu",
) -> Tuple[GValueLikelihoodModel, GValueLikelihoodModel]:
    """
    Train two likelihood models: P(g | watermarked) and P(g | unwatermarked).
    
    Args:
        train_loader: Training data loader
        val_loader: Optional validation data loader
        num_positions: Number of effective g-value positions (N_eff)
        global_mask: Optional global boolean mask [N] for efficient tensor slicing.
                     If provided, must be consistent across all samples (validated in main()).
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Tuple of (watermarked_model, unwatermarked_model)
    """
    # Create models
    model_w = GValueLikelihoodModel(num_positions).to(device)
    model_u = GValueLikelihoodModel(num_positions).to(device)
    
    # Optimizers
    optimizer_w = torch.optim.Adam(model_w.parameters(), lr=lr)
    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=lr)
    
    # Pre-compute global mask indices for efficient tensor slicing
    if global_mask is not None:
        global_mask = global_mask.to(device)
        mask_indices = global_mask.bool()  # [N] boolean mask for slicing
    
    print(f"Training likelihood models...")
    print(f"  Effective positions (N_eff): {num_positions}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    
    for epoch in range(num_epochs):
        # Train watermarked model
        model_w.train()
        train_loss_w = 0.0
        train_count_w = 0
        
        for batch in train_loader:
            g = batch["g"].to(device)
            labels = batch["label"].to(device)
            
            # Only use watermarked samples
            watermarked_mask = (labels == 1)
            if watermarked_mask.sum() == 0:
                continue
            
            g_w = g[watermarked_mask]
            
            # Apply global mask via tensor slicing (efficient, no Python loops)
            if global_mask is not None:
                g_w = g_w[:, mask_indices]  # [B, N_eff]
            
            # Negative log-likelihood loss (g_w is already masked)
            log_likelihood = model_w(g_w)
            loss = -log_likelihood.mean()
            
            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()
            
            train_loss_w += loss.item()
            train_count_w += 1
        
        # Train unwatermarked model
        model_u.train()
        train_loss_u = 0.0
        train_count_u = 0
        
        for batch in train_loader:
            g = batch["g"].to(device)
            labels = batch["label"].to(device)
            
            # Only use unwatermarked samples
            unwatermarked_mask = (labels == 0)
            if unwatermarked_mask.sum() == 0:
                continue
            
            g_u = g[unwatermarked_mask]
            
            # Apply global mask via tensor slicing (efficient, no Python loops)
            if global_mask is not None:
                g_u = g_u[:, mask_indices]  # [B, N_eff]
            
            # Negative log-likelihood loss (g_u is already masked)
            log_likelihood = model_u(g_u)
            loss = -log_likelihood.mean()
            
            optimizer_u.zero_grad()
            loss.backward()
            optimizer_u.step()
            
            train_loss_u += loss.item()
            train_count_u += 1
        
        # Validation
        val_loss_w = 0.0
        val_loss_u = 0.0
        val_count = 0
        
        if val_loader is not None:
            model_w.eval()
            model_u.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    g = batch["g"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Watermarked
                    watermarked_mask = (labels == 1)
                    if watermarked_mask.sum() > 0:
                        g_w = g[watermarked_mask]
                        # Apply global mask via tensor slicing
                        if global_mask is not None:
                            g_w = g_w[:, mask_indices]  # [B, N_eff]
                        log_likelihood = model_w(g_w)
                        val_loss_w += (-log_likelihood.mean()).item()
                    
                    # Unwatermarked
                    unwatermarked_mask = (labels == 0)
                    if unwatermarked_mask.sum() > 0:
                        g_u = g[unwatermarked_mask]
                        # Apply global mask via tensor slicing
                        if global_mask is not None:
                            g_u = g_u[:, mask_indices]  # [B, N_eff]
                        log_likelihood = model_u(g_u)
                        val_loss_u += (-log_likelihood.mean()).item()
                    
                    val_count += 1
        
        # Log
        avg_train_loss_w = train_loss_w / max(train_count_w, 1)
        avg_train_loss_u = train_loss_u / max(train_count_u, 1)
        
        if val_count > 0:
            avg_val_loss_w = val_loss_w / val_count
            avg_val_loss_u = val_loss_u / val_count
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_w={avg_train_loss_w:.4f}, train_u={avg_train_loss_u:.4f}, "
                f"val_w={avg_val_loss_w:.4f}, val_u={avg_val_loss_u:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_w={avg_train_loss_w:.4f}, train_u={avg_train_loss_u:.4f}"
            )
    
    return model_w, model_u


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian likelihood models for g-values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--g-manifest",
        type=str,
        default=None,
        help="Path to g-values manifest.jsonl (created by precompute_inverted_g_values.py). "
             "Required if --g-wm and --g-clean are not provided.",
    )
    
    parser.add_argument(
        "--g-wm",
        type=str,
        default=None,
        help="Path to g_wm.npy (from Phase 1 export). Required if --g-manifest is not provided.",
    )
    
    parser.add_argument(
        "--g-clean",
        type=str,
        default=None,
        help="Path to g_clean.npy (from Phase 1 export). Required if --g-manifest is not provided.",
    )
    
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to mask.npy (from Phase 1 export, optional)",
    )
    
    parser.add_argument(
        "--val-g-manifest",
        type=str,
        default=None,
        help="Path to validation g-values manifest.jsonl (optional)",
    )
    
    parser.add_argument(
        "--val-g-wm",
        type=str,
        default=None,
        help="Path to validation g_wm.npy (optional)",
    )
    
    parser.add_argument(
        "--val-g-clean",
        type=str,
        default=None,
        help="Path to validation g_clean.npy (optional)",
    )
    
    parser.add_argument(
        "--val-mask",
        type=str,
        default=None,
        help="Path to validation mask.npy (optional)",
    )
    
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to metadata.json (created by precompute_inverted_g_values.py). "
             "If not provided, will look for metadata.json in the same directory as --g-manifest.",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for likelihood model JSON (default: {output-dir}/likelihood_params.json or {output-dir}/{family_id}.json)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/likelihood_models",
        help="Directory to save trained models",
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("Training Bayesian Likelihood Models")
    print("=" * 60)
    
    # Determine input mode
    use_numpy = args.g_wm is not None and args.g_clean is not None
    use_manifest = args.g_manifest is not None
    
    if not use_numpy and not use_manifest:
        raise ValueError("Must provide either --g-manifest or both --g-wm and --g-clean")
    
    if use_numpy and use_manifest:
        raise ValueError("Cannot provide both --g-manifest and --g-wm/--g-clean. Choose one mode.")
    
    training_metadata = None
    
    # Create datasets
    if use_numpy:
        # Load from numpy arrays (Phase 1 export)
        g_wm_path = Path(args.g_wm)
        g_clean_path = Path(args.g_clean)
        mask_path = Path(args.mask) if args.mask else None
        
        if not g_wm_path.exists():
            raise FileNotFoundError(f"g_wm.npy not found: {g_wm_path}")
        if not g_clean_path.exists():
            raise FileNotFoundError(f"g_clean.npy not found: {g_clean_path}")
        
        train_dataset = GValueNumpyDataset(g_wm_path, g_clean_path, mask_path)
        print(f"Training samples: {len(train_dataset)}")
        
        # Try to load metadata from meta.json in same directory
        meta_path = g_wm_path.parent / "meta.json"
        if meta_path.exists():
            print(f"\nLoading metadata from {meta_path}")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            training_metadata = {
                "latent_type": "zT",
                "num_inversion_steps": None,  # Not stored in meta
                "g_field_config_hash": None,
                "g_field_config": meta.get("signature", {}).get("g_field", {}),
                # CRITICAL: Extract key fingerprint from meta.json
                "key_fingerprint": meta.get("key_fingerprint"),
                "key_id": meta.get("key_id"),
                "prf_algorithm": meta.get("prf_algorithm"),
            }
            print(f"  Family ID: {meta.get('family_id', 'unknown')}")
            print(f"  N_eff: {meta.get('N_eff', 'unknown')}")
            if meta.get("key_fingerprint"):
                print(f"  Key fingerprint: {meta.get('key_fingerprint', 'unknown')[:16]}...")
        
        # Determine num_positions from g shape
        first_sample = train_dataset[0]
        g_shape = first_sample["g"].shape[-1]
        num_positions = g_shape
        
        print(f"Effective masked positions (N_eff): {num_positions}")
        
        # Validate g-value consistency
        print("Validating g-value consistency...")
        for i in range(min(100, len(train_dataset))):
            sample = train_dataset[i]
            if sample["g"].shape[-1] != g_shape:
                raise ValueError(
                    f"Sample {i} has g shape {sample['g'].shape[-1]}, "
                    f"but expected {g_shape}. G-value shapes must be consistent."
                )
        print("✓ G-value consistency validated")
        
        # Validation dataset
        val_dataset = None
        if args.val_g_wm and args.val_g_clean:
            val_g_wm_path = Path(args.val_g_wm)
            val_g_clean_path = Path(args.val_g_clean)
            val_mask_path = Path(args.val_mask) if args.val_mask else None
            
            val_dataset = GValueNumpyDataset(val_g_wm_path, val_g_clean_path, val_mask_path)
            print(f"Validation samples: {len(val_dataset)}")
            
            val_first = val_dataset[0]
            val_g_shape = val_first["g"].shape[-1]
            if val_g_shape != num_positions:
                raise ValueError(
                    f"Validation set has {val_g_shape} g-value positions, "
                    f"but training set has {num_positions}. Shapes must match."
                )
    
    else:
        # Load from manifest (original mode)
        train_manifest_path = Path(args.g_manifest)
        
        if args.metadata_path:
            metadata_path = Path(args.metadata_path)
        else:
            metadata_path = train_manifest_path.parent / "metadata.json"
        
        if metadata_path.exists():
            print(f"\nLoading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                training_metadata = json.load(f)
            print(f"  Latent type: {training_metadata.get('latent_type', 'unknown')}")
            print(f"  Inversion steps: {training_metadata.get('num_inversion_steps', 'unknown')}")
            print(f"  G-field config hash: {training_metadata.get('g_field_config_hash', 'unknown')}")
        else:
            print(f"\n⚠️  Warning: Metadata file not found at {metadata_path}")
            print("  Train-detect symmetry checks may fail.")
        
        train_dataset = GValueDataset(train_manifest_path)
        print(f"Training samples: {len(train_dataset)}")
        
        first_sample = train_dataset[0]
        g_shape = first_sample["g"].shape[-1]
        num_positions = g_shape
        
        print(f"Effective masked positions (N_eff): {num_positions}")
        print("  (G-values are already masked from precomputation)")
        
        print("Validating g-value consistency...")
        for i in range(min(100, len(train_dataset))):
            sample = train_dataset[i]
            if sample["g"].shape[-1] != g_shape:
                raise ValueError(
                    f"Sample {i} has g shape {sample['g'].shape[-1]}, "
                    f"but expected {g_shape}. G-value shapes must be consistent."
                )
        print("✓ G-value consistency validated")
        
        val_dataset = None
        if args.val_g_manifest:
            val_dataset = GValueDataset(Path(args.val_g_manifest))
            print(f"Validation samples: {len(val_dataset)}")
            
            val_first = val_dataset[0]
            val_g_shape = val_first["g"].shape[-1]
            if val_g_shape != num_positions:
                raise ValueError(
                    f"Validation set has {val_g_shape} g-value positions, "
                    f"but training set has {num_positions}. Shapes must match."
                )
    
    # No global mask needed: g-values are already masked (reduced size)
    global_mask = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
    
    # Train models
    model_w, model_u = train_likelihood_models(
        train_loader,
        val_loader,
        num_positions,
        global_mask=global_mask,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )
    
    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Try to infer family_id from metadata
        family_id = None
        if use_numpy and training_metadata:
            meta_path = Path(args.g_wm).parent / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                family_id = meta.get("family_id")
        
        if family_id:
            output_path = output_dir / f"{family_id}.json"
        else:
            output_path = output_dir / "likelihood_params.json"
    
    # Save as JSON (simple format)
    params_w = model_w.get_parameters()
    params_u = model_u.get_parameters()
    
    output_data = {
        "num_positions": num_positions,
        "watermarked": params_w,
        "unwatermarked": params_u,
    }
    
    # CRITICAL: Extract and store key fingerprint from training data
    # Try to get key info from metadata or args
    key_fingerprint = None
    key_id = None
    prf_algorithm = None
    
    # Check if key info is in metadata (from Phase-1 exports)
    if training_metadata is not None:
        key_fingerprint = training_metadata.get("key_fingerprint")
        key_id = training_metadata.get("key_id")
        prf_algorithm = training_metadata.get("prf_algorithm")
    
    # If not in metadata, try to get from args (if provided)
    if key_fingerprint is None and hasattr(args, 'master_key') and args.master_key:
        from src.core.config import compute_key_fingerprint, PRFConfig
        if hasattr(args, 'key_id') and args.key_id:
            key_id = args.key_id
        if hasattr(args, 'prf_algorithm') and args.prf_algorithm:
            prf_config = PRFConfig(algorithm=args.prf_algorithm)
        else:
            prf_config = PRFConfig()  # Default: chacha20
        key_fingerprint = compute_key_fingerprint(args.master_key, key_id or "default_key_001", prf_config)
        prf_algorithm = prf_config.algorithm
    
    # Store key fingerprint in output
    output_data["key_fingerprint"] = key_fingerprint
    output_data["key_id"] = key_id
    output_data["prf_algorithm"] = prf_algorithm
    
    # Save training metadata for train-detect symmetry verification
    if training_metadata is not None:
        output_data["training_metadata"] = {
            "latent_type": training_metadata.get("latent_type", "unknown"),
            "num_inversion_steps": training_metadata.get("num_inversion_steps"),
            "g_field_config_hash": training_metadata.get("g_field_config_hash"),
            "g_field_config": training_metadata.get("g_field_config"),
        }
    else:
        output_data["training_metadata"] = {
            "latent_type": "unknown",
            "num_inversion_steps": None,
            "g_field_config_hash": None,
            "g_field_config": None,
        }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved likelihood parameters to: {output_path}")
    
    # Also save as torch checkpoint
    checkpoint_path = output_dir / "likelihood_models.pt"
    torch.save({
        "watermarked": model_w.state_dict(),
        "unwatermarked": model_u.state_dict(),
        "num_positions": num_positions,
    }, checkpoint_path)
    
    print(f"✓ Saved model checkpoint to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

