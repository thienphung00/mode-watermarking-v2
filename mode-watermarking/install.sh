#!/bin/bash
# Mode Watermarking Installation Script
# Compatible with pip, conda, and macOS

set -e

echo "🚀 Mode Watermarking Installation Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_status "Detected Windows"
    else
        OS="unknown"
        print_warning "Unknown operating system: $OSTYPE"
    fi
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    else
        print_error "Python not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    print_status "Found Python $PYTHON_VERSION"
    
    # Check Python version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 3.9 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    print_status "Found pip: $PIP_CMD"
}

# Install with pip
install_with_pip() {
    print_status "Installing with pip..."
    
    # Upgrade pip first
    $PIP_CMD install --upgrade pip
    
    # Install the package
    if [ "$1" = "dev" ]; then
        print_status "Installing in development mode with dev dependencies..."
        $PIP_CMD install -e ".[dev]"
    elif [ "$1" = "api" ]; then
        print_status "Installing with API dependencies..."
        $PIP_CMD install -e ".[api]"
    elif [ "$1" = "all" ]; then
        print_status "Installing with all optional dependencies..."
        $PIP_CMD install -e ".[all]"
    else
        print_status "Installing in development mode..."
        $PIP_CMD install -e .
    fi
    
    print_success "Installation completed with pip!"
}

# Install with conda
install_with_conda() {
    print_status "Installing with conda..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "conda not found. Please install Anaconda or Miniconda."
        exit 1
    fi
    
    # Create conda environment
    print_status "Creating conda environment 'mode-watermarking'..."
    conda create -n mode-watermarking python=3.11 -y
    
    # Activate environment
    print_status "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mode-watermarking
    
    # Install with pip in conda environment
    install_with_pip "$1"
    
    print_success "Installation completed with conda!"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    $PYTHON_CMD -c "
import mode_watermarking
print('✅ Mode watermarking imported successfully!')
print(f'Version: {mode_watermarking.__version__}')

# Test core functionality
from mode_watermarking import WatermarkConfig, WatermarkKeyManager
config = WatermarkConfig()
key_manager = WatermarkKeyManager()
print('✅ Core functionality working!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed!"
    else
        print_error "Installation test failed!"
        exit 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --pip          Install using pip (default)"
    echo "  --conda        Install using conda"
    echo "  --dev          Install with development dependencies"
    echo "  --api          Install with API dependencies"
    echo "  --all          Install with all optional dependencies"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Install with pip"
    echo "  $0 --conda            # Install with conda"
    echo "  $0 --pip --dev        # Install with pip and dev dependencies"
    echo "  $0 --conda --api      # Install with conda and API dependencies"
}

# Main installation function
main() {
    local INSTALL_METHOD="pip"
    local INSTALL_TYPE=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pip)
                INSTALL_METHOD="pip"
                shift
                ;;
            --conda)
                INSTALL_METHOD="conda"
                shift
                ;;
            --dev)
                INSTALL_TYPE="dev"
                shift
                ;;
            --api)
                INSTALL_TYPE="api"
                shift
                ;;
            --all)
                INSTALL_TYPE="all"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Detect system
    detect_os
    check_python
    check_pip
    
    print_status "Installation method: $INSTALL_METHOD"
    if [ -n "$INSTALL_TYPE" ]; then
        print_status "Installation type: $INSTALL_TYPE"
    fi
    
    # Install based on method
    if [ "$INSTALL_METHOD" = "conda" ]; then
        install_with_conda "$INSTALL_TYPE"
    else
        install_with_pip "$INSTALL_TYPE"
    fi
    
    # Test installation
    test_installation
    
    echo ""
    print_success "🎉 Mode Watermarking installed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run examples: python -c 'from mode_watermarking.examples import run_all_examples; run_all_examples()'"
    echo "2. Start API server: watermark-api --help"
    echo "3. Check documentation: README.md"
    echo ""
    
    if [ "$INSTALL_METHOD" = "conda" ]; then
        echo "To activate the conda environment:"
        echo "  conda activate mode-watermarking"
    fi
}

# Run main function
main "$@"
