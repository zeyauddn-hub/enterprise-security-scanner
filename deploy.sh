#!/bin/bash

# 🏆 ENTERPRISE SECURITY SCANNER - DEPLOYMENT SCRIPT 🏆
# Production deployment with security validation and monitoring

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="enterprise-security-scanner"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGISTRY="${REGISTRY:-ghcr.io}"
NAMESPACE="${NAMESPACE:-security-scanner}"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    local required_tools=("docker" "kubectl" "helm" "curl" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "❌ Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "❌ Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "✅ All prerequisites met"
}

# Security validation
validate_security() {
    log "🔐 Running security validation..."
    
    # Scan Docker image for vulnerabilities
    if command -v trivy &> /dev/null; then
        info "Running Trivy security scan..."
        trivy image --severity HIGH,CRITICAL "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
    else
        warning "Trivy not installed, skipping image security scan"
    fi
    
    # Validate Kubernetes manifests
    if command -v kubesec &> /dev/null; then
        info "Validating Kubernetes security with kubesec..."
        find k8s/ -name "*.yaml" -exec kubesec scan {} \;
    else
        warning "Kubesec not installed, skipping Kubernetes security validation"
    fi
    
    # Check for secrets in code
    if command -v gitleaks &> /dev/null; then
        info "Checking for secrets with gitleaks..."
        gitleaks detect --source . --verbose
    else
        warning "Gitleaks not installed, skipping secret detection"
    fi
    
    log "✅ Security validation completed"
}

# Build and push images
build_and_push() {
    log "🏗️ Building and pushing Docker images..."
    
    # Build main application
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="${VERSION}" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        -t "${REGISTRY}/${PROJECT_NAME}:${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}:latest" \
        .
    
    # Push images
    docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}:latest"
    
    log "✅ Images built and pushed successfully"
}

# Setup infrastructure
setup_infrastructure() {
    log "🏗️ Setting up infrastructure..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        info "Created namespace: $NAMESPACE"
    fi
    
    # Apply RBAC
    kubectl apply -f k8s/rbac/ -n "$NAMESPACE"
    
    # Setup secrets
    setup_secrets
    
    # Deploy monitoring stack
    deploy_monitoring
    
    log "✅ Infrastructure setup completed"
}

# Setup secrets
setup_secrets() {
    log "🔐 Setting up secrets..."
    
    # Generate encryption key if not exists
    if ! kubectl get secret encryption-key -n "$NAMESPACE" &> /dev/null; then
        ENCRYPTION_KEY=$(openssl rand -base64 32)
        kubectl create secret generic encryption-key \
            --from-literal=key="$ENCRYPTION_KEY" \
            -n "$NAMESPACE"
        info "Created encryption key secret"
    fi
    
    # Generate JWT secret if not exists
    if ! kubectl get secret jwt-secret -n "$NAMESPACE" &> /dev/null; then
        JWT_SECRET=$(openssl rand -base64 64)
        kubectl create secret generic jwt-secret \
            --from-literal=secret="$JWT_SECRET" \
            -n "$NAMESPACE"
        info "Created JWT secret"
    fi
    
    # Database credentials
    if ! kubectl get secret postgres-credentials -n "$NAMESPACE" &> /dev/null; then
        DB_PASSWORD=$(openssl rand -base64 32)
        kubectl create secret generic postgres-credentials \
            --from-literal=username=scanner \
            --from-literal=password="$DB_PASSWORD" \
            --from-literal=database=enterprise_scanner \
            -n "$NAMESPACE"
        info "Created database credentials"
    fi
    
    log "✅ Secrets configured"
}

# Deploy monitoring
deploy_monitoring() {
    log "📊 Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    if ! helm list -n monitoring | grep prometheus &> /dev/null; then
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false
        info "Deployed Prometheus monitoring"
    fi
    
    # Deploy Grafana dashboards
    kubectl apply -f monitoring/grafana/ -n monitoring
    
    log "✅ Monitoring stack deployed"
}

# Deploy application
deploy_application() {
    log "🚀 Deploying application..."
    
    # Update image tags in manifests
    find k8s/ -name "*.yaml" -exec sed -i "s|IMAGE_TAG|${VERSION}|g" {} \;
    
    # Apply ConfigMaps and Secrets first
    kubectl apply -f k8s/configmaps/ -n "$NAMESPACE"
    
    # Deploy PostgreSQL
    kubectl apply -f k8s/postgres/ -n "$NAMESPACE"
    
    # Wait for PostgreSQL to be ready
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    
    # Deploy Redis
    kubectl apply -f k8s/redis/ -n "$NAMESPACE"
    
    # Wait for Redis to be ready
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    
    # Deploy main application
    kubectl apply -f k8s/app/ -n "$NAMESPACE"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/enterprise-scanner -n "$NAMESPACE" --timeout=600s
    
    # Deploy services and ingress
    kubectl apply -f k8s/services/ -n "$NAMESPACE"
    kubectl apply -f k8s/ingress/ -n "$NAMESPACE"
    
    log "✅ Application deployed successfully"
}

# Run health checks
health_checks() {
    log "🏥 Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=enterprise-scanner -n "$NAMESPACE" --timeout=300s
    
    # Get service URL
    if kubectl get service enterprise-scanner-lb -n "$NAMESPACE" &> /dev/null; then
        SERVICE_URL=$(kubectl get service enterprise-scanner-lb -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        if [ -z "$SERVICE_URL" ]; then
            SERVICE_URL=$(kubectl get service enterprise-scanner-lb -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        fi
    else
        # Port forward for testing
        kubectl port-forward service/enterprise-scanner 8080:3000 -n "$NAMESPACE" &
        SERVICE_URL="localhost:8080"
        sleep 5
    fi
    
    # Health endpoint check
    info "Checking health endpoint..."
    if curl -f "http://${SERVICE_URL}/health" &> /dev/null; then
        log "✅ Health check passed"
    else
        error "❌ Health check failed"
        exit 1
    fi
    
    # Metrics endpoint check
    info "Checking metrics endpoint..."
    if curl -f "http://${SERVICE_URL}/metrics" &> /dev/null; then
        log "✅ Metrics endpoint accessible"
    else
        warning "⚠️ Metrics endpoint not accessible"
    fi
    
    # API endpoint check
    info "Checking API endpoints..."
    if curl -f "http://${SERVICE_URL}/api/health" &> /dev/null; then
        log "✅ API endpoints accessible"
    else
        warning "⚠️ API endpoints not accessible"
    fi
    
    log "✅ Health checks completed"
}

# Setup monitoring alerts
setup_alerts() {
    log "🚨 Setting up monitoring alerts..."
    
    # Deploy Prometheus alerts
    kubectl apply -f monitoring/alerts/ -n monitoring
    
    # Configure alertmanager
    kubectl apply -f monitoring/alertmanager/ -n monitoring
    
    log "✅ Monitoring alerts configured"
}

# Backup current deployment
backup_deployment() {
    log "💾 Backing up current deployment..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup Kubernetes manifests
    kubectl get all -n "$NAMESPACE" -o yaml > "${BACKUP_DIR}/k8s-resources.yaml"
    
    # Backup database
    DB_POD=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n "$NAMESPACE" "$DB_POD" -- pg_dump -U scanner enterprise_scanner > "${BACKUP_DIR}/database.sql"
    
    # Backup secrets (encrypted)
    kubectl get secrets -n "$NAMESPACE" -o yaml > "${BACKUP_DIR}/secrets.yaml"
    
    log "✅ Backup completed: $BACKUP_DIR"
}

# Rollback function
rollback() {
    local version="${1:-previous}"
    warning "🔄 Rolling back to version: $version"
    
    kubectl rollout undo deployment/enterprise-scanner -n "$NAMESPACE"
    kubectl rollout status deployment/enterprise-scanner -n "$NAMESPACE" --timeout=300s
    
    log "✅ Rollback completed"
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    log "✅ Cleanup completed"
}

# Main deployment function
main() {
    log "🏆 Starting Enterprise Security Scanner Deployment"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Namespace: $NAMESPACE"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    validate_security
    
    if [ "$1" = "build" ] || [ "$1" = "all" ]; then
        build_and_push
    fi
    
    if [ "$1" = "deploy" ] || [ "$1" = "all" ]; then
        backup_deployment
        setup_infrastructure
        deploy_application
        health_checks
        setup_alerts
    fi
    
    if [ "$1" = "rollback" ]; then
        rollback "${2:-}"
        return
    fi
    
    # Display access information
    log "🎉 Deployment completed successfully!"
    log "📊 Grafana Dashboard: http://grafana.monitoring.svc.cluster.local:3000"
    log "📈 Prometheus Metrics: http://prometheus.monitoring.svc.cluster.local:9090"
    log "🛡️ Security Scanner: http://${SERVICE_URL:-<pending>}"
    log "📚 API Documentation: http://${SERVICE_URL:-<pending>}/docs"
}

# Script usage
usage() {
    echo "Usage: $0 {build|deploy|all|rollback|help}"
    echo ""
    echo "Commands:"
    echo "  build     - Build and push Docker images"
    echo "  deploy    - Deploy to Kubernetes"
    echo "  all       - Build and deploy"
    echo "  rollback  - Rollback to previous version"
    echo "  help      - Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  VERSION     - Image version (default: latest)"
    echo "  ENVIRONMENT - Deployment environment (default: production)"
    echo "  NAMESPACE   - Kubernetes namespace (default: security-scanner)"
    echo "  REGISTRY    - Container registry (default: ghcr.io)"
}

# Check arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    build|deploy|all|rollback)
        main "$@"
        ;;
    help)
        usage
        ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac