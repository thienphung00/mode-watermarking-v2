#!/bin/bash
# Smoke test for watermarking service
# Tests: key registration, image generation, and detection

set -e

API_URL="${API_URL:-http://localhost:8000}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if service is available
log "Checking service health..."
HEALTH=$(curl -s "${API_URL}/health" || echo '{"status":"error"}')
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

if [ "$STATUS" == "error" ]; then
    error "Service not available at ${API_URL}"
fi

log "Service status: $STATUS"
if [ "$VERBOSE" == "true" ]; then
    echo "$HEALTH" | python3 -m json.tool
fi

# Step 1: Register a key
log "Step 1: Registering new watermark key..."
KEY_RESPONSE=$(curl -s -X POST "${API_URL}/keys/register" \
    -H "Content-Type: application/json" \
    -d '{}')

KEY_ID=$(echo "$KEY_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('key_id',''))" 2>/dev/null)

if [ -z "$KEY_ID" ]; then
    error "Failed to register key: $KEY_RESPONSE"
fi

log "Registered key: $KEY_ID"
if [ "$VERBOSE" == "true" ]; then
    echo "$KEY_RESPONSE" | python3 -m json.tool
fi

# Step 2: Generate an image
log "Step 2: Generating watermarked image..."
GENERATE_RESPONSE=$(curl -s -X POST "${API_URL}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"key_id\": \"$KEY_ID\", \"prompt\": \"a beautiful mountain landscape\", \"seed\": 42}")

IMAGE_URL=$(echo "$GENERATE_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('image_url',''))" 2>/dev/null)
SEED_USED=$(echo "$GENERATE_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('seed_used',''))" 2>/dev/null)

if [ -z "$IMAGE_URL" ]; then
    error "Failed to generate image: $GENERATE_RESPONSE"
fi

log "Generated image: $IMAGE_URL (seed: $SEED_USED)"
if [ "$VERBOSE" == "true" ]; then
    echo "$GENERATE_RESPONSE" | python3 -m json.tool
fi

# Step 3: Detect watermark (using a dummy image for now)
log "Step 3: Detecting watermark..."

# Create a simple test image (1x1 PNG)
TEST_IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

DETECT_RESPONSE=$(curl -s -X POST "${API_URL}/detect" \
    -F "key_id=$KEY_ID" \
    -F "image_base64=$TEST_IMAGE_B64")

DETECTED=$(echo "$DETECT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('detected',''))" 2>/dev/null)
CONFIDENCE=$(echo "$DETECT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('confidence',''))" 2>/dev/null)

if [ -z "$DETECTED" ]; then
    warn "Detection response unexpected: $DETECT_RESPONSE"
else
    log "Detection result: detected=$DETECTED, confidence=$CONFIDENCE"
fi

if [ "$VERBOSE" == "true" ]; then
    echo "$DETECT_RESPONSE" | python3 -m json.tool
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}SMOKE TEST PASSED${NC}"
echo "=========================================="
echo "Key ID:    $KEY_ID"
echo "Image URL: $IMAGE_URL"
echo "Detected:  $DETECTED"
echo "Confidence: $CONFIDENCE"
echo "=========================================="
