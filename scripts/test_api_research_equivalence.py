#!/usr/bin/env python3
"""
Equivalence test comparing API detection logic vs research script logic.

This script validates numerical equivalence between:
1. Research detection logic (detect_bayesian_test.py - reference/ground truth)
2. API DetectionService logic (production code)

Both paths operate on identical precomputed g-values and produce:
- posterior: float
- log_odds: float  
- is_watermarked: bool

They should match within tolerance.

CRITICAL: This test does NOT duplicate any detection logic.
- Research path: Uses exact function from detect_bayesian_test.py
- API path: Calls DetectionService.detect_from_g_values() (production code)
- Test layer: Only loads data, calls both paths, and compares results

Usage:
    python scripts/test_api_research_equivalence.py \
        --g-path outputs/test_exp/precomputed_g_values/0000.pt \
        --likelihood-params outputs/likelihood_models_exp/likelihood_params.json \
        --key-id "test_key_001" \
        [--mask-path outputs/test_exp/masks/0000.pt] \
        [--tolerance 1e-6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import exact reference function from research script
# This is the ground truth - do NOT modify or reimplement
from scripts.detect_bayesian_test import detect_watermark_bayesian_from_g_values

# Import production DetectionService
from service.detection import DetectionService
from service.app.dependencies import get_detection_service
from src.models.detectors import BayesianDetector


def load_g_values(g_path: Path) -> torch.Tensor:
    """
    Load g-values tensor from disk.
    
    Args:
        g_path: Path to g-values file (.pt format)
    
    Returns:
        G-values tensor [N] or [1, N]
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not g_path.exists():
        raise FileNotFoundError(f"G-values file not found: {g_path}")
    
    g_data = torch.load(g_path, map_location="cpu")
    
    if isinstance(g_data, dict):
        g = g_data.get("g", g_data.get("g_values"))
        if g is None:
            raise ValueError(f"G-values file {g_path} missing 'g' or 'g_values' key")
    else:
        g = g_data
    
    # Ensure at least 1D
    if g.dim() == 0:
        raise ValueError(f"G-values must be at least 1D, got scalar")
    if g.dim() > 1:
        g = g.flatten()
    
    return g


def load_mask(mask_path: Optional[Path]) -> Optional[torch.Tensor]:
    """
    Load mask tensor from disk if provided.
    
    Args:
        mask_path: Optional path to mask file (.pt format)
    
    Returns:
        Mask tensor [N] or None
    """
    if mask_path is None or not mask_path.exists():
        return None
    
    mask = torch.load(mask_path, map_location="cpu")
    
    if mask.dim() > 1:
        mask = mask.flatten()
    
    # Binarize mask (ensure {0, 1} values)
    mask = (mask > 0.5).float()
    
    return mask


def run_research_detection(
    g: torch.Tensor,
    mask: Optional[torch.Tensor],
    likelihood_params_path: Path,
) -> Dict[str, Any]:
    """
    Run detection using research script logic (reference/ground truth).
    
    This uses the exact function from detect_bayesian_test.py.
    No modifications or reimplementations are allowed.
    
    Args:
        g: G-values tensor [N] or [1, N]
        mask: Optional mask tensor [N] or [1, N]
        likelihood_params_path: Path to likelihood parameters JSON
    
    Returns:
        Dictionary with: posterior, log_odds, is_watermarked
    """
    # Load detector using exact same logic as research script
    detector = BayesianDetector(
        likelihood_params_path=str(likelihood_params_path),
    )
    
    # Call exact reference function from detect_bayesian_test.py
    # This function handles all normalization, binarization, and detection
    result = detect_watermark_bayesian_from_g_values(
        g=g,
        mask=mask,
        detector=detector,
    )
    
    return {
        "posterior": result["posterior"],
        "log_odds": result["log_odds"],
        "is_watermarked": result["is_watermarked"],
    }


def run_api_detection(
    g: torch.Tensor,
    mask: Optional[torch.Tensor],
    likelihood_params_path: Path,
    mask_path: Optional[Path] = None,
    key_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run detection using API DetectionService (production code).
    
    This calls the actual production DetectionService.detect_from_g_values() method.
    No reimplementations or logic duplication.
    
    Uses override mechanism to bypass Authority lookup for testing/demo purposes.
    
    Args:
        g: G-values tensor [N] or [1, N] (raw, will be processed by DetectionService)
        mask: Optional mask tensor [N] or [1, N]
        likelihood_params_path: Path to likelihood parameters JSON (for override mode)
        mask_path: Optional path to mask file (for override mode)
        key_id: Optional key ID (only used if override not provided, for production mode)
    
    Returns:
        Dictionary with: posterior, log_odds, is_watermarked
    """
    # Get production DetectionService instance
    detection_service = get_detection_service()
    
    # Use override mechanism to bypass Authority lookup
    # This allows testing without registered keys
    detection_config_override = {
        "likelihood_params_path": str(likelihood_params_path),
        "mask_path": str(mask_path) if mask_path else None,
        "threshold": 0.5,
        "prior_watermarked": 0.5,
    }
    
    # Call production method that handles all normalization, binarization, and detection
    # This is the same code path used by the API endpoints, but with override config
    result = detection_service.detect_from_g_values(
        g=g,
        mask=mask,
        key_id=key_id,
        detection_config_override=detection_config_override,
        g_field_config_override={},  # Optional unless required
    )
    
    return {
        "posterior": result["posterior"],
        "log_odds": result["log_odds"],
        "is_watermarked": result["is_watermarked"],
    }


def compare_results(
    research_result: Dict[str, Any],
    api_result: Dict[str, Any],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare research and API detection results with strict numerical validation.
    
    Args:
        research_result: Results from research script path
        api_result: Results from API path
        tolerance: Numerical tolerance for comparison
    
    Returns:
        Dictionary with comparison results and pass/fail status
    """
    posterior_diff = abs(research_result["posterior"] - api_result["posterior"])
    log_odds_diff = abs(research_result["log_odds"] - api_result["log_odds"])
    detected_match = research_result["is_watermarked"] == api_result["is_watermarked"]
    
    posterior_match = posterior_diff < tolerance
    log_odds_match = log_odds_diff < tolerance
    
    all_match = posterior_match and log_odds_match and detected_match
    
    return {
        "posterior_diff": posterior_diff,
        "log_odds_diff": log_odds_diff,
        "detected_match": detected_match,
        "posterior_match": posterior_match,
        "log_odds_match": log_odds_match,
        "all_match": all_match,
    }


def test_equivalence(
    g_path: Path,
    likelihood_params_path: Path,
    mask_path: Optional[Path] = None,
    tolerance: float = 1e-6,
    key_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run equivalence test comparing research script vs API detection logic.
    
    Args:
        g_path: Path to precomputed g-values file
        likelihood_params_path: Path to likelihood parameters JSON
        mask_path: Optional path to mask file
        tolerance: Numerical tolerance for comparison
        key_id: Optional key ID (not used in override mode)
    
    Returns:
        Dictionary with test results
    """
    print("=" * 80)
    print("API vs Research Script Equivalence Test")
    print("=" * 80)
    
    # Load inputs (no processing, just loading)
    print(f"\nLoading inputs:")
    print(f"  G-values: {g_path}")
    g = load_g_values(g_path)
    print(f"    shape: {g.shape}, dtype: {g.dtype}")
    print(f"    min: {g.min().item():.6f}, max: {g.max().item():.6f}, mean: {g.mean().item():.6f}")
    
    print(f"  Likelihood params: {likelihood_params_path}")
    
    mask = None
    if mask_path:
        print(f"  Mask: {mask_path}")
        mask = load_mask(mask_path)
        if mask is not None:
            print(f"    shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"    sum: {mask.sum().item():.0f} (valid positions)")
    
    if key_id:
        print(f"  Key ID: {key_id} (not used in override mode)")
    print(f"  Tolerance: {tolerance:.2e}")
    
    # Validate tensor shapes
    print(f"\nValidating inputs:")
    if g.dim() not in (1, 2):
        raise ValueError(f"G-values must be 1D or 2D, got {g.dim()}D with shape {g.shape}")
    if mask is not None and mask.dim() not in (1, 2):
        raise ValueError(f"Mask must be 1D or 2D, got {mask.dim()}D with shape {mask.shape}")
    if mask is not None:
        g_flat_len = g.shape[0] if g.dim() == 1 else g.shape[1]
        mask_flat_len = mask.shape[0] if mask.dim() == 1 else mask.shape[1]
        if g_flat_len != mask_flat_len:
            raise ValueError(
                f"G-values and mask length mismatch: g={g_flat_len}, mask={mask_flat_len}"
            )
    print("  ✓ Input validation passed")
    
    # Run research detection (reference)
    print("\n" + "=" * 80)
    print("Research Script Detection (Reference/Ground Truth)")
    print("=" * 80)
    research_result = run_research_detection(g, mask, likelihood_params_path)
    print(f"  posterior:    {research_result['posterior']:.8f}")
    print(f"  log_odds:     {research_result['log_odds']:.8f}")
    print(f"  is_watermarked: {research_result['is_watermarked']}")
    
    # Run API detection (production)
    print("\n" + "=" * 80)
    print("API Detection (Production Code)")
    print("=" * 80)
    api_result = run_api_detection(
        g=g,
        mask=mask,
        likelihood_params_path=likelihood_params_path,
        mask_path=mask_path,
        key_id=None,  # Use override mode, bypass Authority
    )
    print(f"  posterior:    {api_result['posterior']:.8f}")
    print(f"  log_odds:     {api_result['log_odds']:.8f}")
    print(f"  is_watermarked: {api_result['is_watermarked']}")
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    comparison = compare_results(research_result, api_result, tolerance)
    
    print(f"  posterior difference:  {comparison['posterior_diff']:.2e}")
    print(f"    tolerance:           {tolerance:.2e}")
    print(f"    match:               {comparison['posterior_match']}")
    
    print(f"  log_odds difference:   {comparison['log_odds_diff']:.2e}")
    print(f"    tolerance:           {tolerance:.2e}")
    print(f"    match:               {comparison['log_odds_match']}")
    
    print(f"  detected flag match:   {comparison['detected_match']}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if comparison["all_match"]:
        print("✓ PASS: All results match within tolerance!")
        print("=" * 80)
    else:
        print("✗ FAIL: Results do not match!")
        print("=" * 80)
        if not comparison["posterior_match"]:
            print(f"  - Posterior mismatch: {comparison['posterior_diff']:.2e} > {tolerance:.2e}")
        if not comparison["log_odds_match"]:
            print(f"  - Log-odds mismatch: {comparison['log_odds_diff']:.2e} > {tolerance:.2e}")
        if not comparison["detected_match"]:
            print(f"  - Detection decision mismatch")
    
    return {
        "research": research_result,
        "api": api_result,
        "comparison": comparison,
        "all_match": comparison["all_match"],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test equivalence between API and research script detection logic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--g-path",
        type=str,
        required=True,
        help="Path to precomputed g-values file (.pt format)",
    )
    
    parser.add_argument(
        "--likelihood-params",
        type=str,
        required=True,
        help="Path to likelihood parameters JSON file",
    )
    
    parser.add_argument(
        "--key-id",
        type=str,
        default=None,
        help="Key ID for API detection (optional, not used in override mode)",
    )
    
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Optional path to mask file (.pt format)",
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for comparison",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    g_path = Path(args.g_path).resolve()
    likelihood_params_path = Path(args.likelihood_params).resolve()
    mask_path = Path(args.mask_path).resolve() if args.mask_path else None
    
    # Validate paths
    if not g_path.exists():
        print(f"Error: G-values file not found: {g_path}", file=sys.stderr)
        return 1
    
    if not likelihood_params_path.exists():
        print(f"Error: Likelihood params file not found: {likelihood_params_path}", file=sys.stderr)
        return 1
    
    try:
        result = test_equivalence(
            g_path=g_path,
            likelihood_params_path=likelihood_params_path,
            mask_path=mask_path,
            tolerance=args.tolerance,
            key_id=args.key_id,
        )
        
        return 0 if result["all_match"] else 1
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
