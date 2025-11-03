#!/bin/bash
# Setup script for mode-watermarking

echo "Setting up mode-watermarking environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Setup complete! Activate with: source venv/bin/activate"

