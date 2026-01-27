#!/bin/bash
# Deployment script for watermarking service
#
# Usage:
#   ./deploy.sh build         # Build Docker images
#   ./deploy.sh push          # Push to registry
#   ./deploy.sh deploy        # Deploy to Kubernetes
#   ./deploy.sh rollback      # Rollback to previous version
#   ./deploy.sh status        # Check deployment status

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
CLUSTER_NAME="${CLUSTER_NAME:-watermarking-cluster}"
REGISTRY="${REGISTRY:-gcr.io/${PROJECT_ID}}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build Docker images
build() {
    log_info "Building Docker images with tag: ${TAG}"
    
    # Build API image
    log_info "Building API image..."
    docker build \
        -f deploy/api/Dockerfile \
        -t ${REGISTRY}/watermarking-api:${TAG} \
        -t ${REGISTRY}/watermarking-api:latest \
        .
    
    # Build Worker image
    log_info "Building Worker image..."
    docker build \
        -f deploy/worker/Dockerfile \
        -t ${REGISTRY}/watermarking-worker:${TAG} \
        -t ${REGISTRY}/watermarking-worker:latest \
        .
    
    log_info "Build complete"
}

# Push images to registry
push() {
    log_info "Pushing images to ${REGISTRY}"
    
    # Push API image
    docker push ${REGISTRY}/watermarking-api:${TAG}
    docker push ${REGISTRY}/watermarking-api:latest
    
    # Push Worker image
    docker push ${REGISTRY}/watermarking-worker:${TAG}
    docker push ${REGISTRY}/watermarking-worker:latest
    
    log_info "Push complete"
}

# Deploy to Kubernetes
deploy() {
    log_info "Deploying to Kubernetes cluster: ${CLUSTER_NAME}"
    
    # Get cluster credentials
    gcloud container clusters get-credentials ${CLUSTER_NAME} \
        --region ${REGION} \
        --project ${PROJECT_ID}
    
    # Update image tags in kustomization
    cd deploy/k8s
    kustomize edit set image \
        watermarking-api=${REGISTRY}/watermarking-api:${TAG} \
        watermarking-worker=${REGISTRY}/watermarking-worker:${TAG}
    cd ../..
    
    # Apply manifests
    kubectl apply -k deploy/k8s/
    
    # Wait for rollout
    log_info "Waiting for API deployment..."
    kubectl rollout status deployment/api -n watermarking --timeout=300s
    
    log_info "Waiting for Worker deployment..."
    kubectl rollout status deployment/gpu-worker -n watermarking --timeout=600s
    
    log_info "Deployment complete"
}

# Rollback to previous version
rollback() {
    log_info "Rolling back deployments"
    
    kubectl rollout undo deployment/api -n watermarking
    kubectl rollout undo deployment/gpu-worker -n watermarking
    
    log_info "Rollback initiated. Waiting for completion..."
    kubectl rollout status deployment/api -n watermarking --timeout=300s
    kubectl rollout status deployment/gpu-worker -n watermarking --timeout=600s
    
    log_info "Rollback complete"
}

# Check deployment status
status() {
    log_info "Checking deployment status"
    
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n watermarking -o wide
    
    echo ""
    echo "=== Services ==="
    kubectl get services -n watermarking
    
    echo ""
    echo "=== Deployments ==="
    kubectl get deployments -n watermarking
    
    echo ""
    echo "=== HPA ==="
    kubectl get hpa -n watermarking
    
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n watermarking
}

# Main
case "$1" in
    build)
        build
        ;;
    push)
        push
        ;;
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    all)
        build
        push
        deploy
        status
        ;;
    *)
        echo "Usage: $0 {build|push|deploy|rollback|status|all}"
        echo ""
        echo "Commands:"
        echo "  build     Build Docker images"
        echo "  push      Push images to registry"
        echo "  deploy    Deploy to Kubernetes"
        echo "  rollback  Rollback to previous version"
        echo "  status    Check deployment status"
        echo "  all       Build, push, and deploy"
        exit 1
        ;;
esac

