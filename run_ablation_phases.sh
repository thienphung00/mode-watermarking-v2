#!/bin/bash
# Commands to run the ablation pipeline phases 0-3
# 
# Prerequisites:
# - Config YAML files in experiments/watermark_ablation/configs/
# - Prompts file (one prompt per line) for Phase 1
# - Master key for watermarking

# ============================================================================
# Phase 0: Group ablation configs into detector families
# ============================================================================
# This phase extracts geometry signatures and groups configs by detector geometry.
# Output: experiments/watermark_ablation/families/family_XXX/{signature.json, configs.json}

python scripts/generate_ablation_configs.py \
    --configs-dir experiments/watermark_ablation/configs \
    --output-dir experiments/watermark_ablation/families

# ============================================================================
# Phase 1: Export g-values per family for likelihood training
# ============================================================================
# This phase generates images, inverts to latents, and computes g-values.
# Output: experiments/g_datasets/family_XXX/{g_wm.npy, g_clean.npy, mask.npy, meta.json}

python scripts/run_ablation_g_export.py \
    --families-dir experiments/watermark_ablation/families \
    --cache-dir experiments/watermark_ablation/cache \
    --output-dir experiments/g_datasets \
    --prompts-file data/prompts.txt \
    --num-samples 500 \
    --master-key "your_secret_master_key_here" \
    --device cuda \
    --num-inversion-steps 25 \
    --configs-dir experiments/watermark_ablation/configs

# ============================================================================
# Phase 2: Train likelihood models per family
# ============================================================================
# This phase trains Bayesian likelihood models P(g | watermarked) and P(g | unwatermarked).
# Output: experiments/likelihood_models/family_XXX.json
#
# Note: Run this once per family. You can loop over families or run manually per family.

# Example for a single family (replace family_XXX with actual family ID):
FAMILY_ID="family_001"
python scripts/train_g_likelihoods.py \
    --g-wm experiments/g_datasets/${FAMILY_ID}/g_wm.npy \
    --g-clean experiments/g_datasets/${FAMILY_ID}/g_clean.npy \
    --mask experiments/g_datasets/${FAMILY_ID}/mask.npy \
    --output-dir experiments/likelihood_models \
    --output experiments/likelihood_models/${FAMILY_ID}.json \
    --num-epochs 10 \
    --batch-size 32 \
    --lr 0.01 \
    --device cuda

# Or loop over all families:
# for family_dir in experiments/g_datasets/family_*; do
#     family_id=$(basename "$family_dir")
#     python scripts/train_g_likelihoods.py \
#         --g-wm "${family_dir}/g_wm.npy" \
#         --g-clean "${family_dir}/g_clean.npy" \
#         --mask "${family_dir}/mask.npy" \
#         --output-dir experiments/likelihood_models \
#         --output "experiments/likelihood_models/${family_id}.json" \
#         --num-epochs 10 \
#         --batch-size 32 \
#         --lr 0.01 \
#         --device cuda
# done

# ============================================================================
# Phase 3: Run detection ablation using trained likelihood models
# ============================================================================
# This phase runs detection on all configs using family-specific likelihood models.
# Output: experiments/watermark_ablation/results/{config_name}.json

python scripts/run_ablation_detection.py \
    --families-dir experiments/watermark_ablation/families \
    --likelihood-dir experiments/likelihood_models \
    --cache-dir experiments/watermark_ablation/cache \
    --results-dir experiments/watermark_ablation/results \
    --master-key "your_secret_master_key_here" \
    --device cuda \
    --num-inversion-steps 25 \
    --configs-dir experiments/watermark_ablation/configs

