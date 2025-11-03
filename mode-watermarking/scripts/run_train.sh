#!/bin/bash
# Training pipeline automation script
# Simplifies experiment execution for detector training

set -e  # Exit on error

# Default values
DETECTOR="unet"
CONFIG="configs/train_config.yaml"
RESUME_FROM=""
OUTPUT_DIR=""
DEVICE="cuda"
CONFIG_DIR=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --detector)
            DETECTOR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
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
            echo "  --detector TYPE         Detector type: 'unet' or 'bayesian' (default: unet)"
            echo "  --config PATH           Path to train_config.yaml (default: configs/train_config.yaml)"
            echo "  --resume-from PATH      Path to checkpoint to resume from (optional)"
            echo "  --output-dir PATH       Override output directory from config (optional)"
            echo "  --device DEVICE         Device: 'cuda' or 'cpu' (default: cuda)"
            echo "  --config-dir PATH       Directory containing configs (optional)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --detector unet --config configs/train_config.yaml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate detector type
if [[ "$DETECTOR" != "unet" && "$DETECTOR" != "bayesian" ]]; then
    echo "Error: Detector must be 'unet' or 'bayesian'"
    exit 1
fi

# Validate config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Build command
CMD="python -m src.cli.train --detector $DETECTOR --config $CONFIG --device $DEVICE"

if [[ -n "$RESUME_FROM" ]]; then
    CMD="$CMD --resume-from $RESUME_FROM"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [[ -n "$CONFIG_DIR" ]]; then
    CMD="$CMD --config-dir $CONFIG_DIR"
fi

# Print configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Detector:      $DETECTOR"
echo "Config:        $CONFIG"
echo "Device:        $DEVICE"
if [[ -n "$RESUME_FROM" ]]; then
    echo "Resume from:   $RESUME_FROM"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output dir:    $OUTPUT_DIR"
fi
echo "=========================================="
echo ""

# Execute training
echo "Starting training..."
echo ""

$CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

