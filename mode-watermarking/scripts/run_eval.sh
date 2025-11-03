#!/bin/bash
# Evaluation pipeline automation script

set -e  # Exit on error

# Default values
TEST_MANIFEST=""
EVAL_CONFIG="configs/eval_config.yaml"
DETECTOR_TYPE=""
CHECKPOINT=""
OUTPUT_DIR=""
DEVICE="cuda"
CONFIG_DIR=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-manifest)
            TEST_MANIFEST="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        --detector-type)
            DETECTOR_TYPE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test-manifest PATH    Path to test manifest file (required)"
            echo "  --eval-config PATH      Path to eval_config.yaml (default: configs/eval_config.yaml)"
            echo "  --detector-type TYPE    Detector type: 'unet' or 'bayesian' (optional, overrides config)"
            echo "  --checkpoint PATH       Path to detector checkpoint (optional, overrides config)"
            echo "  --output-dir PATH       Output directory (optional)"
            echo "  --device DEVICE         Device: 'cuda' or 'cpu' (default: cuda)"
            echo "  --config-dir PATH       Directory containing configs (optional)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --test-manifest data/splits/test.json --eval-config configs/eval_config.yaml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TEST_MANIFEST" ]]; then
    echo "Error: --test-manifest is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate test manifest exists
if [[ ! -f "$TEST_MANIFEST" ]]; then
    echo "Error: Test manifest not found: $TEST_MANIFEST"
    exit 1
fi

# Validate eval config exists
if [[ ! -f "$EVAL_CONFIG" ]]; then
    echo "Error: Eval config not found: $EVAL_CONFIG"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Build command
CMD="python -m src.cli.eval --test-manifest $TEST_MANIFEST --eval-config $EVAL_CONFIG --device $DEVICE"

if [[ -n "$DETECTOR_TYPE" ]]; then
    CMD="$CMD --detector-type $DETECTOR_TYPE"
fi

if [[ -n "$CHECKPOINT" ]]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [[ -n "$CONFIG_DIR" ]]; then
    CMD="$CMD --config-dir $CONFIG_DIR"
fi

# Print configuration
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Test manifest:  $TEST_MANIFEST"
echo "Eval config:    $EVAL_CONFIG"
echo "Device:         $DEVICE"
if [[ -n "$DETECTOR_TYPE" ]]; then
    echo "Detector type:  $DETECTOR_TYPE"
fi
if [[ -n "$CHECKPOINT" ]]; then
    echo "Checkpoint:     $CHECKPOINT"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output dir:     $OUTPUT_DIR"
fi
echo "=========================================="
echo ""

# Execute evaluation
echo "Starting evaluation..."
echo ""

$CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "Evaluation completed successfully!"
else
    echo ""
    echo "Evaluation failed with exit code $?"
    exit 1
fi

