# Comprehensive Codebase Summary: Mode Watermarking

This document describes the current, active architecture of the mode-watermarking codebase. It reflects the post-refactor system and intentionally omits unused or deprecated workflows.

---

## Overview

Mode Watermarking is a PRF-based watermarking system for Stable Diffusion images.
The system embeds a watermark by biasing the initial latent z_T in a deterministic, variance-preserving way and detects the watermark using a SynthID-style statistical test.

### Core Properties

- Seed-bias watermarking (latent initialization) is the primary and supported embedding method
- Unified PRF system for both generation and detection
- Metadata-free detection (requires only image + key_id)
- Statistical detection with theoretical guarantees
- Hybrid detection pipeline for speed–accuracy trade-offs

---

## Project Structure (Active Components Only)

```
mode-watermarking-restructured/
├── src/
│   ├── algorithms/
│   │   ├── g_field.py
│   │   ├── masks.py
│   │   └── scheduling.py
│   ├── core/
│   │   ├── config.py
│   │   ├── interfaces.py
│   │   ├── context.py
│   │   └── metadata_schema.py
│   ├── engine/
│   │   ├── pipeline.py
│   │   ├── strategy_factory.py
│   │   ├── strategies/
│   │   │   └── seed_bias.py
│   │   ├── sampling_utils.py
│   │   └── trainer.py
│   ├── detection/
│   │   ├── inversion.py
│   │   ├── observe.py
│   │   ├── prf.py
│   │   ├── gfield.py
│   │   ├── statistics.py
│   │   ├── calibration.py
│   │   ├── pipeline.py
│   │   └── survival.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   └── transforms.py
│   └── models/
│       ├── detectors.py
│       └── layers.py
│   └── evaluate_detector_v2.py
├── configs/
├── tests/
├── docs/
└── README.md
```

---

## Core Architecture

### Watermark Embedding (Seed Bias Strategy)

Seed Bias Watermarking injects a watermark by modifying the initial latent z_T before denoising begins.

1. Sample ε ~ N(0, I)
2. Generate watermark field G using PRF(master_key, key_id)
3. Apply frequency bandpass filtering to G
4. Normalize G to zero mean and unit variance
5. Mix:
   ```
   z_T = √(1 − λ²) · ε + λ · G
   ```
6. Run standard diffusion denoising (no hooks)

**Key properties:**
- Deterministic per key_id
- Variance preserving
- No UNet hooks
- No scheduler modification
- No metadata dependency

This is the default and recommended embedding path.

### Unified PRF System

All watermark-related randomness is derived from a single cryptographically secure PRF.

**PRF Definition**
```
seed_i = PRF(master_key, key_id, index=i)
```

- **Implemented in**: `detection/prf.py`
- **Algorithms**: ChaCha20 or AES-CTR
- **Used identically by**:
  - Generation
  - Detection
  - Detector training

There are no parallel key systems and no metadata-derived seeds.

### Detection Architecture

Detection is self-contained and requires only:
- (image, key_id, master_key)

#### Detection Pipeline

1. **Latent recovery**
   - Fast: VAE encode → z_0
   - Accurate: DDIM inversion → z_T

2. **Expected watermark**
   - Generated via the same PRF used in embedding

3. **Observed signal extraction**
   - z_0 → whitened extraction
   - z_T → normalized extraction

4. **Statistical test**
   ```
   S = (1 / √n) · Σ G_observed[i] · G_expected[i]
   ```
   - Under H₀: S ~ N(0, 1)

5. **Decision**
   - Thresholded by calibrated or theoretical bounds

#### Hybrid Detection (Seed Bias)

Implemented in `detection/pipeline.py`.

| Stage | Method | Cost | Purpose |
|-------|--------|------|---------|
| Stage 1 | VAE encode + whitened extraction | Fast | Early decision |
| Stage 2 | Full DDIM inversion | Slow | Exact recovery |

Stage 2 is only executed if Stage 1 is ambiguous.

---

## Active Modules

### `detection/prf.py`

Cryptographic PRF used system-wide.

### `engine/strategies/seed_bias.py`

Implements latent initialization watermarking.

### `detection/statistics.py`

SynthID-style S-statistic computation.

### `detection/observe.py`

Context-aware latent signal extraction.

### `algorithms/g_field.py`

Deterministic watermark field generation from PRF seeds.

---

## Legacy / Compatibility Modules

These modules are not part of the active pipeline and must not be used for new development.

### `algorithms/keys.py` (LEGACY)

- Old XOR-shift / LCG-based key derivation
- Metadata-dependent
- Kept only for `initialize_zT_hash()` compatibility
- Not used for watermark generation or detection

### `detection/recover.py` (LEGACY)

- Bridges to deprecated recovery logic
- Not used by the current detection pipeline

These modules appear only for backward compatibility and documentation completeness.

---

## Removed from Documentation

The following are intentionally not documented because they are unused or unreachable:

- Metadata-dependent watermark pipelines
- XOR-shift or LCG-based key streams
- Parallel key derivation systems
- Experimental recovery paths
- Unmaintained scripts or examples

If it is not reachable from the current pipeline, it does not appear here.

---

## Configuration Model

Watermarking is enabled exclusively via `seed_bias`:

```yaml
watermark:
  mode: watermarked
  algorithm_params:
    seed_bias:
      lambda_strength: 0.05
      domain: frequency
      low_freq_cutoff: 0.05
      high_freq_cutoff: 0.4
      detection_mode: hybrid
      num_inference_steps: 50
  key_settings:
    key_master: SECRET
    key_id: example_key
```

No watermarking occurs unless `seed_bias` is explicitly configured.

---

## Testing Scope

Tests validate:

- PRF determinism
- Variance preservation of z_T
- Detection null distribution
- Hybrid detector correctness
- End-to-end generation → detection consistency

Tests use real diffusion pipelines, not mocks.

---

## Summary

**Current system characteristics:**

✅ Seed-bias watermarking only  
✅ Unified cryptographic PRF  
✅ Metadata-free detection  
✅ Hybrid detection pipeline  
✅ Statistical guarantees  

❌ No legacy key systems  
❌ No metadata-derived randomness  
❌ No undocumented strategies  

If a component is not documented here, it is not part of the active system.
