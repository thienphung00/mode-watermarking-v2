# Watermark Key Dependency Verification Report

## Executive Summary

This report provides a static code analysis of the watermark pipeline to determine whether it is truly conditioned on `master_key`, or whether architectural shortcuts allow key-independent leakage.

---

## 1. MASTER KEY FLOW VERIFICATION

### 1.1 Entry Point: `scripts/run_ablation_g_export.py`

**File**: `scripts/run_ablation_g_export.py`

**Flow**:
1. **Line 547-550**: `master_key` parsed from CLI argument `--master-key` (required)
   ```python
   parser.add_argument("--master-key", type=str, required=True, help="Master key for PRF")
   ```

2. **Line 624**: `master_key` passed to `export_family_g_dataset()` function
   ```python
   master_key=args.master_key,
   ```

3. **Line 333**: `export_family_g_dataset()` receives `master_key: str` parameter

4. **Line 418**: `master_key` passed to `compute_g_values_for_image()` function
   ```python
   master_key=master_key,
   ```

5. **Line 170**: `compute_g_values_for_image()` receives `master_key: str` parameter

6. **Line 300**: `master_key` passed to `compute_g_values()` function
   ```python
   g, mask = compute_g_values(
       latent_T,
       computation_key,
       master_key,  # ← Passed here
       return_mask=True,
       g_field_config=g_field_config,
       latent_type="zT",
   )
   ```

**Key Observation**: `master_key` flows directly from CLI → function calls → `compute_g_values()` without transformation.

---

### 1.2 Entry Point: `scripts/run_ablation_detection.py`

**File**: `scripts/run_ablation_detection.py`

**Flow**:
1. **Line 533-536**: `master_key` parsed from CLI argument `--master-key` (required)
   ```python
   parser.add_argument("--master-key", type=str, required=True, help="Master key for PRF")
   ```

2. **Line 611**: `master_key` passed to `run_detection_for_config()` function
   ```python
   master_key=args.master_key,
   ```

3. **Line 327**: `run_detection_for_config()` receives `master_key: str` parameter

4. **Line 407, 436**: `master_key` passed to `compute_detection_statistics()` function
   ```python
   master_key=master_key,
   ```

5. **Line 132**: `compute_detection_statistics()` receives `master_key: str` parameter

6. **Line 262**: `master_key` passed to `compute_g_values()` function
   ```python
   g, mask = compute_g_values(
       latent_T,
       computation_key,
       master_key,  # ← Passed here
       return_mask=True,
       g_field_config=g_field_config,
       latent_type="zT",
   )
   ```

**Key Observation**: Same direct flow pattern as g-export script.

---

### 1.3 Core Function: `src/detection/g_values.py::compute_g_values()`

**File**: `src/detection/g_values.py`

**Flow**:
1. **Line 190**: Function signature receives `master_key: str`
   ```python
   def compute_g_values(
       x0: torch.Tensor,
       key: str,
       master_key: str,  # ← Received here
       ...
   )
   ```

2. **Line 316-320**: `master_key` used to create `PRFKeyDerivation` instance
   ```python
   if prf_config is None:
       prf_config = PRFConfig(algorithm="chacha20", output_bits=64)
   
   prf = PRFKeyDerivation(master_key, prf_config)  # ← master_key used here
   ```

3. **Line 325**: PRF generates seeds using `key_id` (not master_key directly, but master_key is in PRF state)
   ```python
   prf_seeds = prf.generate_seeds(key, num_elements)  # key is key_id
   ```

**Key Observation**: `master_key` is stored in PRF instance state, then used indirectly via `generate_seeds()`.

---

### 1.4 PRF Key Derivation: `src/detection/prf.py::PRFKeyDerivation`

**File**: `src/detection/prf.py`

**Flow**:
1. **Line 59-76**: `__init__()` receives `master_key: str` and stores it
   ```python
   def __init__(self, master_key: str, config: Optional[PRFConfig] = None):
       self.config = config or PRFConfig(algorithm="chacha20", output_bits=64)
       self._master_key_raw = master_key  # ← Stored as raw string
       
       # Derive 256-bit key from master_key using SHA-256
       self._derived_key = self._derive_key(master_key)  # ← Hashed to 32 bytes
   ```

2. **Line 78-92**: `_derive_key()` hashes master_key to 32 bytes
   ```python
   def _derive_key(self, master_key: str) -> bytes:
       return hashlib.sha256(master_key.encode("utf-8")).digest()  # ← SHA-256 hash
   ```

3. **Line 94-108**: `_derive_nonce()` derives nonce from `key_id` (NOT master_key)
   ```python
   def _derive_nonce(self, key_id: str) -> bytes:
       return hashlib.sha256(key_id.encode("utf-8")).digest()[:16]  # ← Only key_id
   ```

4. **Line 110-139**: `generate_seed()` uses both `_derived_key` (from master_key) and nonce (from key_id)
   ```python
   def generate_seed(self, key_id: str, index: int) -> int:
       nonce = self._derive_nonce(key_id)  # ← key_id → nonce
       prf_bytes = self._prf_at_offset(nonce, byte_offset, bytes_per_output)
       # Uses self._derived_key (from master_key) internally
   ```

5. **Line 224-250**: `_chacha20_stream()` uses `self._derived_key` (from master_key) and nonce (from key_id)
   ```python
   def _chacha20_stream(self, nonce: bytes, length: int) -> bytes:
       cipher = Cipher(
           algorithms.ChaCha20(self._derived_key, nonce),  # ← master_key via _derived_key
           ...
       )
   ```

**Key Observation**: 
- `master_key` is hashed to `_derived_key` (SHA-256, deterministic)
- `key_id` is hashed to `nonce` (SHA-256, deterministic)
- Both are used in ChaCha20: `ChaCha20(derived_key, nonce)`
- **The PRF output depends on BOTH master_key and key_id**

---

### 1.5 G-Field Generation: `src/algorithms/g_field.py::GFieldGenerator`

**File**: `src/algorithms/g_field.py`

**Flow**:
1. **Line 280-358**: `generate_g_field()` receives PRF seeds (already derived from master_key + key_id)
   ```python
   def generate_g_field(self, shape, seeds, ...):
       # seeds are PRF outputs (depend on master_key + key_id)
       values_array = np.array(values, dtype=np.uint64)  # ← PRF seeds
       g_values = self._map_to_gvalues(values_array)  # ← Maps seeds to ±1
   ```

2. **Line 399-413**: `_map_to_gvalues()` maps PRF seeds to binary values
   ```python
   def _map_to_gvalues(self, values: np.ndarray) -> np.ndarray:
       if self.mapping_mode == "binary":
           bits = ((values >> np.uint64(self.bit_pos)) & np.uint64(1))
           g = (2 * bits - 1).astype(np.float32)  # ← ±1 values
   ```

**Key Observation**: G-field is deterministic function of PRF seeds, which depend on master_key + key_id.

---

### 1.6 Summary: Master Key Flow

**Complete Flow**:
```
CLI arg (--master-key)
  ↓
export_family_g_dataset(master_key)
  ↓
compute_g_values_for_image(master_key)
  ↓
compute_g_values(x0, key_id, master_key)
  ↓
PRFKeyDerivation(master_key) → _derived_key = SHA256(master_key)
  ↓
prf.generate_seeds(key_id) → uses _derived_key + nonce(key_id)
  ↓
GFieldGenerator.generate_g_field(seeds) → G-field
  ↓
compute_g_values() compares latent vs G-field → g-values
```

**Transformation Points**:
1. **SHA-256 hash** (line 92 in `prf.py`): `master_key` → 32-byte `_derived_key` (deterministic)
2. **ChaCha20** (line 241 in `prf.py`): `_derived_key` + `nonce(key_id)` → PRF stream
3. **Bit extraction** (line 411 in `g_field.py`): PRF seeds → binary ±1 values
4. **Frequency filtering** (line 349 in `g_field.py`): G-field → filtered G-field (key-independent geometry)
5. **Sign comparison** (line 354-358 in `g_values.py`): latent vs G-field → binary g-values

**Dead Parameters**: None identified. `master_key` is used at every step.

**Unused Variables**: None identified.

**Key Stops Influencing Computation**: Never. `master_key` influences:
- PRF key derivation (SHA-256 hash)
- PRF seed generation (ChaCha20 with derived_key)
- G-field generation (from PRF seeds)
- g-value computation (sign comparison with G-field)

---

## 2. G-FIELD / G-VALUE KEY DEPENDENCY VERIFICATION

### 2.1 Function: `compute_g_values()`

**File**: `src/detection/g_values.py::compute_g_values()`

**Inputs**:
- `x0`: Observed latent tensor [B, 4, 64, 64]
- `key`: Key identifier (key_id) for PRF
- `master_key`: Master key for PRF
- `g_field_config`: G-field configuration dict
- `prf_config`: Optional PRF configuration

**Direct Dependency on master_key**: **YES**

**Evidence**:
- Line 320: `prf = PRFKeyDerivation(master_key, prf_config)` - master_key used to create PRF
- Line 325: `prf_seeds = prf.generate_seeds(key, num_elements)` - PRF uses master_key internally
- Line 329: `g_gen = GFieldGenerator(**g_field_config)` - G-field generator created
- Line 331: `G_expected_np, mask_np = g_gen.generate_g_field(shape, prf_seeds, return_mask=True)` - G-field generated from PRF seeds
- Line 354-358: Sign comparison between latent and G-field produces g-values

**How Key is Mixed**:
1. **SHA-256 hash**: `master_key` → 32-byte `_derived_key` (cryptographically secure, deterministic)
2. **ChaCha20 stream cipher**: `ChaCha20(_derived_key, nonce(key_id))` → PRF stream
3. **Bit extraction**: PRF stream → binary ±1 values (bit position 30)
4. **Frequency filtering**: Binary values → filtered G-field (geometry is key-independent, but values are key-dependent)
5. **Normalization**: Zero-mean, unit-variance normalization (preserves key dependency)

**Is Mixing Deterministic**: **YES** - Same master_key + key_id → same PRF seeds → same G-field → same g-values

**Is Mixing Cryptographically Strong**: **YES** - Uses SHA-256 (key derivation) and ChaCha20 (PRF), both cryptographically secure

**Is Seed Global or Per-Sample**: **Per-sample** - Each `key_id` produces different PRF seeds, but same `key_id` always produces same seeds

---

### 2.2 Mask Computation

**File**: `src/algorithms/g_field.py::GFieldGenerator._get_frequency_mask()`

**Key Dependency**: **NO** - Masks are key-independent

**Evidence**:
- Line 437-471: `_get_frequency_mask()` computes mask based on:
  - `H, W` (latent dimensions)
  - `frequency_mode` (lowpass/highpass/bandpass)
  - `low_freq_cutoff`, `high_freq_cutoff` (frequency cutoffs)
- **No use of master_key or key_id**
- Mask geometry is purely structural (frequency domain filtering)

**Conclusion**: Masks identify valid watermark positions based on frequency domain geometry. They are deterministic given the g_field_config but do NOT depend on master_key.

---

### 2.3 G-Value Caching

**File**: `scripts/run_ablation_g_export.py`

**Are g-values cached?**: **YES** (implicitly via latent cache)

**Evidence**:
- Line 204-259: Latent cache logic caches `latent_T` (inverted latents)
- Line 296-304: g-values are computed from cached latents
- **g-values themselves are NOT cached separately** - they are recomputed each time

**Cache Filenames Encode master_key?**: **NO**

**Evidence**:
- Line 212-216: Cache key computed via `compute_cache_key()`:
  ```python
  cache_key = compute_cache_key(
      image_id=image_id,
      config=config,
      num_inversion_steps=num_inversion_steps,
  )
  ```
- `compute_cache_key()` does NOT include master_key (see Section 3)

**Can g-values from one key be reused by another key?**: **POTENTIALLY YES** (if latents are cached and g-values are recomputed with wrong key)

**Evidence**:
- Latents are cached key-independently (cache key doesn't include master_key)
- If detection uses wrong master_key but same cached latent, g-values will be computed with wrong key
- However, g-values are recomputed each time (not cached), so they will reflect the master_key used in `compute_g_values()`

**Critical Finding**: While g-values are recomputed (not cached), the **latents** that feed into g-value computation ARE cached key-independently. This means:
- Phase-1: Generates images with key A, inverts to latents, caches latents
- Phase-3: Loads cached latents, computes g-values with key B (wrong key)
- Result: g-values computed with wrong key, but from latents generated with correct key

---

## 3. CACHE CONTAMINATION AND REUSE CHECK

### 3.1 Cache Key Computation

**File**: `src/core/config.py::compute_cache_key()`

**Function**: Lines 584-678

**Cache Key Components**:
```python
components = []

# 1. Geometry signature hash
if isinstance(config.watermark, WatermarkedConfig):
    geometry_sig = extract_detector_geometry_signature(config.watermark)
    normalized_sig = normalize_geometry_signature(geometry_sig)
    sig_json = json.dumps(normalized_sig, sort_keys=True, separators=(',', ':'))
    sig_hash = hashlib.md5(sig_json.encode()).hexdigest()[:8]
    components.append(f"geom{sig_hash}")

# 2. Full config hash
config_dict = config.model_dump(mode='json')
config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]
components.append(f"cfg{config_hash}")

# 3. Prompt hash (if provided)
if prompt is not None:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    components.append(f"prompt{prompt_hash}")

# 4. Seed (if provided)
if seed is not None:
    components.append(f"seed{seed}")

# 5. Inversion parameters
if num_inversion_steps is not None:
    components.append(f"invsteps{num_inversion_steps}")

# 6. Model ID hash
model_hash = hashlib.md5(config.diffusion.model_id.encode()).hexdigest()[:8]
components.append(f"model{model_hash}")

# 7. Image generation parameters
components.append(f"guidance{config.diffusion.guidance_scale}")
components.append(f"infsteps{config.diffusion.inference_timesteps}")

# 8. Code version hash (if available)
if code_version_hash is not None:
    components.append(f"code{code_version_hash[:8]}")

# 9. Image ID
components.append(f"img{image_id}")
```

**Does Cache Key Include master_key?**: **NO**

**Evidence**:
- Cache key includes: geometry signature, config hash, prompt, seed, inversion steps, model_id, guidance scale, inference steps, code version, image_id
- **master_key is NOT included in any component**
- **key_id is NOT included in cache key** (only image_id, which is sample identifier, not key identifier)

**Critical Finding**: Cache keys are key-agnostic. Same cache key will be generated for:
- Same config (geometry signature)
- Same image_id
- Same inversion steps
- **Regardless of master_key or key_id**

---

### 3.2 Cache Directory Patterns

**File**: `scripts/run_ablation_g_export.py`

**Cache Directory Structure**:
- Line 367-368: `config_cache_dir = cache_dir / config_name` - Cache directory based on config name
- Line 371: `latent_cache_dir = config_cache_dir / "latents"` - Latents cached in subdirectory

**Does Cache Directory Include master_key?**: **NO**

**Evidence**:
- Line 355: `config_cache_dir = cache_dir / config_name` - Only config name
- Line 356: `latent_cache_dir = config_cache_dir / "latents"` - Subdirectory for latents
- **No master_key or key_id in path**

**Can Changing master_key Reuse Same Cached Artifacts?**: **YES**

**Evidence**:
- Cache directory: `cache_dir / config_name / latents /`
- Cache filename: `{cache_key}.pt` where `cache_key` does NOT include master_key
- If Phase-1 runs with key A, generates latents, caches them
- If Phase-3 runs with key B, uses same cache directory, finds cached latents, loads them
- **Result**: Latents generated with key A are used for detection with key B

---

### 3.3 Phase-1 Cache Creation

**File**: `scripts/run_ablation_g_export.py`

**Does Phase-1 Only Create One Cache Per Family?**: **YES**

**Evidence**:
- Line 611-612: Only first config per family is processed:
  ```python
  # Pick first config as representative
  config_path = args.configs_dir / config_paths[0]
  ```
- Line 618-629: `export_family_g_dataset()` called only for first config
- Line 374-399: Cache check - if `manifest_path.exists()`, uses cached dataset
- **Other configs in family are NOT processed in Phase-1**

**Does Detection Reuse Phase-1 Cache Regardless of Key?**: **YES**

**Evidence**:
- `run_ablation_detection.py` line 359-367: Checks for cached manifest
- If manifest exists, loads cached images and latents
- **No check for master_key compatibility**
- Line 403-415: Uses cached latents to compute g-values with **current** master_key (which may be different from Phase-1 master_key)

**Critical Finding**: 
1. Phase-1 creates cache for first config per family (one cache per family)
2. Phase-3 reuses this cache for ALL configs in family
3. Cache does NOT encode master_key
4. Detection uses cached latents but computes g-values with potentially different master_key

---

### 3.4 Latent Cache Metadata

**File**: `scripts/run_ablation_g_export.py`

**Latent Cache Format**:
- Line 278-287: Cached latents stored with metadata:
  ```python
  cache_data = {
      "latent": latent_T,
      "metadata": {
          "image_id": image_id,
          "num_inversion_steps": num_inversion_steps,
          "model_id": config.diffusion.model_id,
          "cache_key": cache_key,
      }
  }
  ```

**Does Metadata Include master_key?**: **NO**

**Evidence**:
- Metadata includes: image_id, num_inversion_steps, model_id, cache_key
- **master_key is NOT stored in metadata**
- **key_id is NOT stored in metadata**

**Can Cached Artifacts Store Embedded Key Information?**: **NO** (not in current implementation)

**Evidence**:
- Latent cache stores only: latent tensor + metadata (image_id, steps, model_id, cache_key)
- No watermark key information stored
- Images are stored as PNG files (no metadata embedding visible in code)

---

## 4. LIKELIHOOD TRAINING KEY AWARENESS

### 4.1 Training Script: `scripts/train_g_likelihoods.py`

**File**: `scripts/train_g_likelihoods.py`

**Does Training Receive master_key?**: **NO**

**Evidence**:
- No `--master-key` argument in argument parser
- Training loads precomputed g-values from numpy arrays (Phase-1 exports)
- Line 149-166: `GValueNumpyDataset` loads `g_wm.npy` and `g_clean.npy`
- **No master_key parameter anywhere in training pipeline**

**Were g-values Generated Using Exactly One Key?**: **NOT VERIFIABLE FROM CODE**

**Evidence**:
- Phase-1 (`run_ablation_g_export.py`) uses single master_key per run (line 624)
- But Phase-1 could be run multiple times with different keys
- Training loads g-values from files, doesn't know which key was used

**Is master_key Stored in Trained Likelihood Artifacts?**: **NO**

**Evidence**:
- Line 902-929: Likelihood model saved as JSON:
  ```python
  output_data = {
      "num_positions": num_positions,
      "watermarked": params_w,
      "unwatermarked": params_u,
  }
  ```
- Line 912-919: Training metadata stored:
  ```python
  output_data["training_metadata"] = {
      "latent_type": training_metadata.get("latent_type", "unknown"),
      "num_inversion_steps": training_metadata.get("num_inversion_steps"),
      "g_field_config_hash": training_metadata.get("g_field_config_hash"),
      "g_field_config": training_metadata.get("g_field_config"),
  }
  ```
- **No master_key in output**

**Does Likelihood Model Receive master_key During Inference?**: **NO**

**Evidence**:
- `BayesianDetector.score()` (line 327-449 in `detectors.py`) receives only:
  - `g`: Binary g-values
  - `mask`: Optional mask
- **No master_key parameter**
- Model computes likelihoods from g-values only

**Is Likelihood Purely Modeling g-Distributions Independent of Key?**: **YES**

**Evidence**:
- Model learns: `P(g_i = 1 | watermarked)` and `P(g_i = 1 | unwatermarked)` per position
- These are position-wise Bernoulli probabilities
- **No key information in model**
- Model is trained on g-values computed with one key, but doesn't know which key

**Can Likelihood Model Theoretically Distinguish Keys?**: **NO**

**Evidence**:
- Model learns position-wise probabilities: `P(g_i | class)`
- If g-values from key A and key B produce different position-wise patterns, model could learn to distinguish them
- But model has no explicit key information
- **Model can only distinguish if g-value patterns differ between keys (which they should, if keys are different)**

**Critical Finding**: 
- Likelihood model is key-agnostic (doesn't receive key)
- Model learns g-value distributions from training data
- If training data uses key A, model learns patterns specific to key A
- If detection uses key B, model may still work if g-value patterns are similar (structural similarity) or fail if patterns differ (key-dependent)

---

### 4.2 Training Data Source

**File**: `scripts/train_g_likelihoods.py`

**Training Data**:
- Line 728-738: Loads `g_wm.npy` and `g_clean.npy` from Phase-1 exports
- Phase-1 exports g-values computed with specific master_key
- **Training doesn't know which key was used**

**Does Training Mix Keys?**: **NOT VERIFIABLE FROM CODE**

**Evidence**:
- Training loads single set of g-value files per family
- Files are from single Phase-1 run (one master_key)
- But nothing prevents loading files from multiple Phase-1 runs with different keys
- **No validation that all training data uses same key**

---

## 5. WATERMARK INJECTION KEY DEPENDENCY

### 5.1 Latent Injection Strategy

**File**: `src/engine/strategy_factory.py::create_watermark_strategy()`

**Flow**:
1. Line 95: `prf = PRFKeyDerivation(config.key_settings.key_master, prf_config)` - master_key from config
2. Line 229: `prf_seeds = prf.generate_seeds(key_id, total_elements)` - PRF seeds depend on master_key + key_id
3. Line 232-236: `g_schedule = strategy._g_field_generator.generate_schedule(..., seeds=prf_seeds)` - G-fields from PRF seeds

**Does master_key Influence Injection?**: **YES**

**Evidence**:
- G-fields are generated from PRF seeds
- PRF seeds depend on master_key (via `_derived_key`) and key_id (via `nonce`)
- G-fields are injected into noise predictions (via `WatermarkHook`)
- **Injection is key-dependent**

---

### 5.2 Seed Bias Strategy

**File**: `src/engine/strategies/seed_bias.py::SeedBiasStrategy`

**Flow**:
1. Line 69: `self.master_key = master_key` - Stored in strategy
2. Line 86: `self.prf = PRFKeyDerivation(master_key, prf_config)` - PRF created with master_key
3. Line 150: `prf_seeds = self.prf.generate_seeds(prf_key_id, num_elements)` - PRF seeds depend on master_key + key_id
4. Line 157-160: `G_np = self.g_field_generator.generate_g_field(shape, seeds=prf_seeds)` - G-field from PRF seeds
5. Line 169: `z_T = sqrt_term * epsilon + lambda_val * G` - Initial latent biased with G-field

**Does master_key Influence Injection?**: **YES**

**Evidence**:
- Initial latent `z_T` is biased with G-field
- G-field depends on PRF seeds
- PRF seeds depend on master_key + key_id
- **Injection is key-dependent**

---

### 5.3 Key-Invariant Structural Elements

**Are There Key-Invariant Elements?**: **YES** (masks and frequency geometry)

**Evidence**:
- Masks are key-independent (frequency domain geometry)
- Frequency filtering (lowpass/highpass/bandpass) is key-independent
- Normalization (zero-mean, unit-variance) is key-independent
- **But G-field VALUES are key-dependent** (from PRF seeds)

**Conclusion**: Structural elements (masks, frequency geometry) are key-invariant, but the actual watermark signal (G-field values) is key-dependent.

---

## 6. NEGATIVE CONTROL CAPABILITY (KEY A vs KEY B)

### 6.1 Architecture Support for Negative Control

**Question**: Can architecture support (watermarked with key A) → (detected with key B)?

**Answer**: **PARTIALLY YES** (due to cache reuse, but g-values will be wrong)

**Evidence**:
1. **Generation with key A**:
   - Images generated with key A
   - Latents inverted and cached (cache key doesn't include master_key)
   - g-values computed with key A (for training)

2. **Detection with key B**:
   - Cached latents loaded (same cache key)
   - g-values recomputed with key B (wrong key)
   - Likelihood model (trained on key A g-values) used for detection

3. **Result**:
   - Latents are from key A generation (correct)
   - g-values computed with key B (wrong)
   - Likelihood model expects key A patterns (trained on key A)
   - **Mismatch between g-values (key B) and model expectations (key A)**

**Critical Finding**: Architecture allows negative control, but:
- Cached latents are from correct key (key A)
- g-values are computed with wrong key (key B)
- Likelihood model expects correct key patterns (key A)
- **Detection will fail if keys are truly different** (g-value patterns won't match)

---

### 6.2 Cache Reuse Across Keys

**Do Caches Force Reuse Across Keys?**: **YES** (cache keys don't include master_key)

**Evidence**:
- Cache key: `geom{hash}_cfg{hash}_..._img{image_id}`
- **No master_key or key_id in cache key**
- Same cache key generated for same config + image_id, regardless of key
- **Cached latents can be reused across different keys**

---

### 6.3 G-Export Artifacts Key-Separability

**Are g-Export Artifacts Key-Separable?**: **NO** (not explicitly)

**Evidence**:
- Phase-1 exports: `g_wm.npy`, `g_clean.npy`, `mask.npy`, `meta.json`
- `meta.json` (line 480-487) includes: family_id, config_used, N_eff, num_samples, signature
- **No master_key or key_id in metadata**
- **g-values in .npy files are computed with specific key, but key is not stored**

**Can Detection Operate on g-Values Generated with Different Key?**: **YES** (technically possible, but wrong)

**Evidence**:
- Detection loads likelihood model (trained on key A g-values)
- Detection computes g-values with key B
- Model scores g-values (key B) using patterns learned from key A
- **Result depends on similarity of g-value patterns between keys**

---

### 6.4 Implicit Key Coupling

**Does Pipeline Implicitly Couple All Runs to Single Effective Key?**: **YES** (via cache reuse)

**Evidence**:
1. Phase-1: Generates with key A, caches latents
2. Phase-2: Trains on g-values from key A
3. Phase-3: Uses cached latents (from key A), computes g-values with key B, uses model (trained on key A)
4. **Effective key is key A** (latents and model), but detection uses key B (g-values)

**Critical Finding**: Pipeline has implicit key coupling via:
- Cached latents (from generation key)
- Trained likelihood model (from training key)
- But detection can use different key for g-value computation

---

## 7. PERFORMANCE SHORTCUT / FAST EXECUTION VERIFICATION

### 7.1 What Is Cached vs Recomputed

**Cached**:
1. **Images** (line 129-130, 152-153 in `run_ablation_g_export.py`): Saved to disk, reused if manifest exists
2. **Latents** (line 204-290): Inverted latents cached as `.pt` files
3. **Manifests** (line 374-399): Dataset manifests cached as JSON

**Recomputed**:
1. **g-values** (line 296-304): Always recomputed from latents (not cached)
2. **Masks** (line 296-304): Always recomputed (but should be identical for all samples)

**Evidence**:
- Line 204-259: Latent cache logic - if cached latent exists, loads it; otherwise inverts and caches
- Line 296-304: g-values always computed via `compute_g_values()` (not cached)
- **g-values are recomputed each time, but from cached latents**

---

### 7.2 Why Phase-1 and Detection Execute Quickly

**Phase-1 Fast Execution**:
1. **Image generation**: Only done once per family (first config)
2. **Latent inversion**: Cached after first computation
3. **g-value computation**: Fast (just sign comparison)

**Detection Fast Execution**:
1. **No image generation**: Uses cached images from Phase-1
2. **No latent inversion**: Uses cached latents from Phase-1
3. **g-value computation**: Fast (sign comparison)
4. **Likelihood scoring**: Fast (pre-trained model, just matrix ops)

**Evidence**:
- Phase-1 line 611-612: Only processes first config per family
- Detection line 359-367: Reuses cached manifest (images + latents)
- **Most expensive operations (generation, inversion) are cached**

---

### 7.3 Is Reuse Key-Agnostic?

**Latent Cache Reuse**: **YES** (key-agnostic)

**Evidence**:
- Cache key doesn't include master_key
- Same cache key for same config + image_id, regardless of key
- **Cached latents can be used with any key**

**g-Value Computation**: **NO** (key-dependent)

**Evidence**:
- g-values are recomputed each time
- Computation uses current master_key (from CLI arg)
- **g-values reflect the key used in computation**

**Likelihood Model**: **KEY-AGNOSTIC** (doesn't receive key)

**Evidence**:
- Model learns g-value patterns from training data
- Model doesn't know which key was used
- **Model is key-agnostic but pattern-specific**

**Critical Finding**: 
- **Latents are key-agnostic** (cached, can be reused)
- **g-values are key-dependent** (recomputed with current key)
- **Likelihood model is key-agnostic** (doesn't receive key, but learns key-specific patterns)

---

### 7.4 Potential Contamination

**Can Reuse Contaminate Experimental Validity?**: **YES**

**Evidence**:
1. Phase-1 generates with key A, caches latents
2. Detection uses cached latents (from key A) but computes g-values with key B
3. Likelihood model trained on key A patterns
4. **Mismatch**: Latents (key A) + g-values (key B) + model (key A patterns)

**Result**: Detection may work if:
- g-value patterns are similar between keys (structural similarity)
- Likelihood model generalizes across keys (unlikely if keys are truly different)
- Or detection fails (expected if keys are different)

---

## 8. FINAL VERDICT

### 8.1 Is Watermark Truly Key-Conditioned?

**Answer**: **YES, BUT WITH ARCHITECTURAL RISKS**

**Evidence**:
1. **G-field generation is key-dependent**:
   - PRF seeds depend on master_key (SHA-256 hash) + key_id (nonce)
   - G-fields are deterministic function of PRF seeds
   - **Different master_key → different G-fields → different g-values**

2. **Watermark injection is key-dependent**:
   - Latent injection: G-fields injected into noise predictions (key-dependent)
   - Seed bias: Initial latent biased with G-field (key-dependent)
   - **Different master_key → different watermark signal**

3. **g-value computation is key-dependent**:
   - `compute_g_values()` uses master_key to generate expected G-field
   - Sign comparison between latent and G-field produces g-values
   - **Different master_key → different expected G-field → different g-values**

**BUT**:
- **Cache keys don't include master_key** → cached latents can be reused across keys
- **Likelihood models don't receive master_key** → models are key-agnostic but pattern-specific
- **Phase-1 creates one cache per family** → other configs reuse first config's cache

---

### 8.2 Concrete Evidence

**Key-Dependent Components**:
1. ✅ PRF seed generation (`prf.py::PRFKeyDerivation.generate_seeds()`)
2. ✅ G-field generation (`g_field.py::GFieldGenerator.generate_g_field()`)
3. ✅ g-value computation (`g_values.py::compute_g_values()`)
4. ✅ Watermark injection (`strategy_factory.py`, `seed_bias.py`)

**Key-Independent Components**:
1. ❌ Cache keys (`config.py::compute_cache_key()` - no master_key)
2. ❌ Cache directories (based on config name, not key)
3. ❌ Likelihood models (`detectors.py::BayesianDetector` - no master_key parameter)
4. ❌ Masks (frequency geometry, key-independent)

---

### 8.3 Architectural Risks

**Confirmed Risks**:
1. **Cache contamination**: Cached latents can be reused across different master_keys
   - **Location**: `run_ablation_g_export.py` line 212-216, `compute_cache_key()` line 584-678
   - **Impact**: Detection with wrong key uses latents from correct key generation

2. **Likelihood model key-agnosticism**: Models don't receive master_key
   - **Location**: `detectors.py::BayesianDetector.score()` line 327-449
   - **Impact**: Models learn key-specific patterns but don't know which key

3. **Phase-1 single cache per family**: Only first config per family creates cache
   - **Location**: `run_ablation_g_export.py` line 611-612
   - **Impact**: Other configs reuse first config's cache (may have different key_id but same master_key)

**Suspected Risks**:
1. **g-value pattern similarity**: If g-value patterns are similar across keys (due to structural elements), likelihood model may generalize incorrectly
2. **Cache metadata missing key**: Cached latents don't store key information, making it impossible to verify key compatibility

---

### 8.4 Failure Modes Explaining Empirical Observations

**Observation 1**: "Detection still performs similarly even when using the wrong master_key"

**Explanation**:
- Cached latents are from correct key generation (structural watermark present)
- g-values computed with wrong key (wrong expected G-field)
- But likelihood model (trained on correct key patterns) may still detect structural similarities
- **OR**: If keys are similar (e.g., same key_id, different master_key), g-value patterns may be similar

**Observation 2**: "Cache directories are reused across configs and possibly across keys"

**Explanation**:
- Cache keys don't include master_key → same cache key for same config + image_id
- Cache directories based on config name, not key
- **Confirmed**: Cache reuse across keys is possible

**Observation 3**: "Phase-1 only creates one cache per family"

**Explanation**:
- Line 611-612 in `run_ablation_g_export.py`: Only first config per family processed
- **Confirmed**: Phase-1 intentionally creates one cache per family

**Observation 4**: "g-export and detection run much faster than full image generation"

**Explanation**:
- Images and latents are cached (no generation/inversion needed)
- Only g-value computation and likelihood scoring (both fast)
- **Confirmed**: Cache reuse explains fast execution

**Observation 5**: "Some families show non-random AUC even when the key is incorrect"

**Explanation**:
- Likelihood model learns position-wise probabilities from training data
- If g-value patterns have structural similarities (masks, frequency geometry), model may detect these
- **OR**: If training and detection use different keys but similar patterns, model may generalize

---

### 8.5 Conclusion

**Is the watermark truly key-conditioned?**: **YES** (g-values and injection are key-dependent)

**What concrete evidence supports this?**:
- PRF seed generation uses master_key (SHA-256 hash → ChaCha20)
- G-field generation depends on PRF seeds
- g-value computation compares latents against key-dependent G-field
- Watermark injection uses key-dependent G-fields

**What architectural risks are confirmed?**:
1. ✅ Cache keys don't include master_key → cache reuse across keys
2. ✅ Likelihood models don't receive master_key → key-agnostic but pattern-specific
3. ✅ Phase-1 creates one cache per family → cache reuse across configs

**Which failure modes explain empirical observations?**:
1. Cache contamination: Wrong key detection uses correct key latents
2. Likelihood model generalization: Model may detect structural similarities even with wrong key
3. g-value pattern similarity: If keys produce similar patterns, detection may work incorrectly

**Final Answer**: The watermark **IS** key-conditioned at the computation level, but **architectural shortcuts** (cache reuse, key-agnostic models) allow key-independent leakage that can explain empirical observations of detection working with wrong keys.

