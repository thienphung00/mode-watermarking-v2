# Watermark Strength Optimization Guide

This document explains how to configure watermark parameters to achieve stronger detection results when running `generate_training_images.py`.

## Detection Theory Overview

The detection system uses the **S-statistic**:

```
S = (1 / √n) * Σ G_observed[i] * G_expected[i]
```

Where:
- **n** = number of effective positions (N_eff)
- **G_observed** = watermark signal extracted from image latent
- **G_expected** = expected watermark pattern from PRF

**Statistical Properties:**
- Under H0 (no watermark): `S ~ N(0, 1)`
- Under H1 (watermarked): `S ~ N(√n * ρ, 1)` where ρ is correlation

**Key Insight:** To increase detection strength, we need to:
1. **Increase correlation ρ** (stronger signal injection)
2. **Increase N_eff** (more effective positions)

## Parameter Tuning Strategies

### 1. Increase Injection Strength (`lambda_strength`)

**Parameter:** `seed_bias.lambda_strength`

**Current Default:** `0.075`

**Stronger Config:** `0.08 - 0.10`

**Impact:**
- Directly increases correlation ρ between observed and expected signals
- Higher λ → stronger watermark signal in z_T
- Formula: `z_T = sqrt(1 - λ²) * ε + λ * G`

**Trade-offs:**
- ✅ Higher S-statistic values
- ✅ Better separation between watermarked/unwatermarked
- ⚠️ Values >0.15 may cause visible artifacts
- ⚠️ May slightly reduce image quality (PSNR/SSIM)

**Recommendation:** Start with `0.09`, monitor quality metrics.

---

### 2. Widen Frequency Band (Increase N_eff)

**Parameters:** 
- `g_field.low_freq_cutoff` / `seed_bias.low_freq_cutoff`
- `g_field.high_freq_cutoff` / `seed_bias.high_freq_cutoff`

**Current Default:** `low: 0.05, high: 0.4`

**Stronger Config:** `low: 0.03, high: 0.5`

**Impact:**
- Wider band → more frequency bins carry watermark → higher N_eff
- N_eff directly multiplies the S-statistic mean: `E[S] = √N_eff * ρ`
- More positions = stronger statistical signal

**Trade-offs:**
- ✅ Significantly increases N_eff (can double or triple it)
- ✅ Better detection without increasing visible artifacts
- ⚠️ May include more noise in high frequencies
- ⚠️ Very low frequencies (<0.03) may cause artifacts

**Recommendation:** 
- Decrease `low_freq_cutoff` to `0.03` (but not lower)
- Increase `high_freq_cutoff` to `0.5` (can go up to `0.6` if needed)
- **CRITICAL:** Ensure `g_field` and `seed_bias` cutoffs match exactly

---

### 3. Increase Mask Density

**Parameters:**
- `mask.strength`
- `mask.bandwidth_fraction`

**Current Default:** `strength: 0.9, bandwidth_fraction: 0.20`

**Stronger Config:** `strength: 0.95, bandwidth_fraction: 0.25-0.30`

**Impact:**
- Higher mask strength → more positions actively carry watermark
- Larger bandwidth → smoother transitions, more positions included
- Both increase N_eff

**Trade-offs:**
- ✅ More effective positions without quality loss
- ✅ Minimal impact on image quality (mask is applied to G-field, not image)
- ⚠️ Very high strength (>0.98) may reduce robustness

**Recommendation:**
- Increase `strength` to `0.95`
- Increase `bandwidth_fraction` to `0.25-0.30`

---

### 4. Enable Unit Variance Normalization

**Parameter:** `g_field.normalize_unit_variance`

**Current Default:** `true` ✓

**Impact:**
- Ensures G-field has unit variance → stronger signal scaling
- Critical for consistent detection performance

**Recommendation:** Always keep enabled (already set correctly).

---

### 5. Increase Inference Steps

**Parameter:** `diffusion.inference_timesteps`

**Current Default:** `25`

**Stronger Config:** `30-50`

**Impact:**
- More steps → better signal preservation through denoising
- Watermark signal may be better preserved with more refinement steps

**Trade-offs:**
- ✅ Better signal preservation
- ⚠️ 2x slower generation (25→50 steps)
- ⚠️ Diminishing returns beyond 50 steps

**Recommendation:** Use `50` steps if generation time allows, otherwise `30-40` is a good compromise.

---

## Expected Detection Improvements

With the "strong" configuration (`seedbias_strong.yaml`):

| Metric | Default Config | Strong Config | Improvement |
|--------|---------------|---------------|-------------|
| **λ (lambda_strength)** | 0.075 | 0.09 | +20% signal strength |
| **Frequency Band** | 0.05-0.4 | 0.03-0.5 | ~2-3x more positions |
| **N_eff (estimated)** | ~2,000-3,000 | ~5,000-7,000 | ~2-3x increase |
| **Expected S-statistic** | ~3-5 | ~6-10 | ~2x increase |
| **TPR @ 1% FPR** | ~70-85% | ~90-98% | +15-20% |

*Note: Actual results depend on image content, detector training, and other factors.*

---

## Usage Example

```bash
# Generate training images with strong watermark config
python scripts/generate_training_images.py \
    --mode both \
    --config configs/experiments/seedbias_strong.yaml \
    --unwatermarked-config configs/experiments/unwatermarked.yaml \
    --prompts-file data/coco/prompts_train.txt \
    --output-dir outputs/train_strong \
    --batch-size 32 \
    --num-images 1000 \
    --key-id batch_001_strong \
    --num-inference-steps 50
```

---

## Monitoring and Validation

After generating with stronger config, verify:

1. **Quality Metrics** (should remain acceptable):
   - PSNR > 35 dB (ideally > 40 dB)
   - SSIM > 0.95 (ideally > 0.98)

2. **Detection Metrics** (should improve):
   - Higher S-statistic values for watermarked images
   - Better TPR at target FPR (e.g., 1%)
   - Larger separation between watermarked/unwatermarked distributions

3. **Visual Inspection**:
   - No visible artifacts in generated images
   - Images maintain high fidelity

---

## Troubleshooting

**Problem:** Detection still weak after increasing parameters

**Solutions:**
1. Verify frequency cutoffs match between `g_field` and `seed_bias`
2. Check that `normalize_unit_variance: true` is set
3. Ensure using same `key_id` for generation and detection
4. Generate more training data (larger dataset helps)
5. Train detector for more epochs

**Problem:** Visible artifacts appear

**Solutions:**
1. Reduce `lambda_strength` to 0.08 or lower
2. Increase `low_freq_cutoff` to 0.05 (remove very low frequencies)
3. Check that mask isn't too aggressive

**Problem:** Detection works but quality degrades

**Solutions:**
1. Use moderate `lambda_strength` (0.08-0.09)
2. Focus on increasing N_eff (frequency band, mask) rather than λ
3. Increase inference steps to preserve quality

---

## Mathematical Relationship

The expected S-statistic under H1 (watermarked) is:

```
E[S | H1] = √N_eff * ρ
```

Where:
- **N_eff** = number of effective positions (affected by frequency band width and mask)
- **ρ** = correlation (affected by `lambda_strength`)

To maximize detection:
- **Maximize N_eff**: Widen frequency band, increase mask density
- **Maximize ρ**: Increase `lambda_strength` (within quality constraints)

The TPR at threshold τ is:

```
TPR = P(S > τ | H1) = 1 - Φ(τ - √N_eff * ρ)
```

Where Φ is the standard normal CDF. Higher N_eff and ρ both increase TPR.

---

## References

- Detection uses SynthID-style S-statistic (Dathathri et al., Nature 2024)
- Seed bias strategy: `z_T = sqrt(1 - λ²) * ε + λ * G`
- See `src/detection/statistics.py` for implementation details

