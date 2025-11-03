#!/bin/bash
# End-to-end pipeline: Generate dataset → Train → Evaluate

set -e  # Exit on error

# Default values
PROMPTS_FILE=""
OUTPUT_DIR="data/generated"
CONFIG_DIR="configs"
NUM_SAMPLES=""
DETECTOR="unet"
DEVICE="cuda"
SKIP_GENERATE=false
SKIP_TRAIN=false
SKIP_EVAL=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --detector)
            DETECTOR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-generate)
            SKIP_GENERATE=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Runs full pipeline: dataset generation → training → evaluation"
            echo ""
            echo "Options:"
            echo "  --prompts-file PATH     Path to prompts file (required unless --skip-generate)"
            echo "  --output-dir PATH       Output directory (default: data/generated)"
            echo "  --config-dir PATH       Config directory (default: configs)"
            echo "  --num-samples N         Number of samples to generate (optional)"
            echo "  --detector TYPE         Detector type: 'unet' or 'bayesian' (default: unet)"
            echo "  --device DEVICE         Device: 'cuda' or 'cpu' (default: cuda)"
            echo "  --skip-generate         Skip dataset generation step"
            echo "  --skip-train           Skip training step"
            echo "  --skip-eval            Skip evaluation step"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --prompts-file data/coco/prompts_train.txt --num-samples 100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Full Pipeline Execution"
echo "=========================================="
echo ""

# Step 1: Generate Dataset
if [[ "$SKIP_GENERATE" == false ]]; then
    if [[ -z "$PROMPTS_FILE" ]]; then
        echo "Error: --prompts-file is required for dataset generation"
        exit 1
    fi
    
    echo "Step 1/3: Generating dataset..."
    echo "----------------------------------------"
    
    CMD_GEN="python scripts/generate_dataset.py --mode both --prompts-file $PROMPTS_FILE --output-dir $OUTPUT_DIR --config-dir $CONFIG_DIR --device $DEVICE"
    
    if [[ -n "$NUM_SAMPLES" ]]; then
        CMD_GEN="$CMD_GEN --num-samples $NUM_SAMPLES"
    fi
    
    $CMD_GEN
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Dataset generation failed"
        exit 1
    fi
    
    echo ""
    echo "Dataset generation complete!"
    echo ""
else
    echo "Skipping dataset generation (--skip-generate)"
    echo ""
fi

# Step 2: Train Detector
if [[ "$SKIP_TRAIN" == false ]]; then
    echo "Step 2/3: Training detector..."
    echo "----------------------------------------"
    
    TRAIN_CONFIG="$CONFIG_DIR/train_config.yaml"
    
    if [[ ! -f "$TRAIN_CONFIG" ]]; then
        echo "Error: Train config not found: $TRAIN_CONFIG"
        exit 1
    fi
    
    ./scripts/run_train.sh \
        --detector "$DETECTOR" \
        --config "$TRAIN_CONFIG" \
        --device "$DEVICE" \
        --config-dir "$CONFIG_DIR"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Training failed"
        exit 1
    fi
    
    echo ""
    echo "Training complete!"
    echo ""
else
    echo "Skipping training (--skip-train)"
    echo ""
fi

# Step 3: Evaluate
if [[ "$SKIP_EVAL" == false ]]; then
    echo "Step 3/3: Running evaluation..."
    echo "----------------------------------------"
    
    TEST_MANIFEST="data/splits/test.json"
    
    if [[ ! -f "$TEST_MANIFEST" ]]; then
        echo "Warning: Test manifest not found: $TEST_MANIFEST"
        echo "Skipping evaluation (create data/splits/test.json to enable)"
    else
        EVAL_CONFIG="$CONFIG_DIR/eval_config.yaml"
        
        ./scripts/run_eval.sh \
            --test-manifest "$TEST_MANIFEST" \
            --eval-config "$EVAL_CONFIG" \
            --detector-type "$DETECTOR" \
            --device "$DEVICE" \
            --config-dir "$CONFIG_DIR"
        
        if [[ $? -ne 0 ]]; then
            echo "Error: Evaluation failed"
            exit 1
        fi
        
        echo ""
        echo "Evaluation complete!"
        echo ""
    fi
else
    echo "Skipping evaluation (--skip-eval)"
    echo ""
fi

echo "=========================================="
echo "Full Pipeline Complete!"
echo "=========================================="

