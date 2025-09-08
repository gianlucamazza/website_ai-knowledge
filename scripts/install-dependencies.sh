#!/bin/bash

# Robust dependency installation script with timeout handling and retries
# This script implements the tiered dependency strategy for CI/CD

set -euo pipefail

# Configuration
MAX_RETRIES=3
TIMEOUT_DURATION=300  # 5 minutes
LOG_FILE="${LOG_FILE:-/tmp/dependency-install.log}"
DEPENDENCY_TIER="${DEPENDENCY_TIER:-ci}"  # ci, dev, full
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    if [[ ${#TEMP_FILES[@]} -gt 0 ]]; then
        log "Cleaning up temporary files..."
        rm -f "${TEMP_FILES[@]}"
    fi
}

# Signal handlers
trap cleanup EXIT INT TERM

# Initialize temporary files array
TEMP_FILES=()

# Function to run command with timeout and retries
run_with_retry() {
    local cmd="$1"
    local description="$2"
    local max_attempts="${3:-$MAX_RETRIES}"
    local timeout="${4:-$TIMEOUT_DURATION}"
    
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Attempt $attempt/$max_attempts: $description"
        
        if timeout "$timeout" bash -c "$cmd" >> "$LOG_FILE" 2>&1; then
            log_success "$description completed successfully"
            return 0
        else
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                log_error "$description timed out after ${timeout}s"
            else
                log_error "$description failed with exit code $exit_code"
            fi
            
            if [[ $attempt -lt $max_attempts ]]; then
                local delay=$((attempt * 10))
                log_warning "Waiting ${delay}s before retry..."
                sleep $delay
            fi
        fi
        
        ((attempt++))
    done
    
    log_error "$description failed after $max_attempts attempts"
    return 1
}

# Function to check disk space
check_disk_space() {
    local required_mb="${1:-1000}"
    local available_mb=$(df /tmp | tail -1 | awk '{print int($4/1024)}')
    
    if [[ $available_mb -lt $required_mb ]]; then
        log_error "Insufficient disk space. Required: ${required_mb}MB, Available: ${available_mb}MB"
        return 1
    fi
    
    log "Disk space check passed. Available: ${available_mb}MB"
    return 0
}

# Function to detect and optimize for CI environment
optimize_for_ci() {
    if [[ "${CI:-}" == "true" ]]; then
        log "CI environment detected. Applying optimizations..."
        
        # Set pip optimizations
        export PIP_NO_CACHE_DIR=1
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        export PIP_QUIET=1
        
        # Set npm optimizations
        export NPM_CONFIG_PROGRESS=false
        export NPM_CONFIG_LOGLEVEL=error
        export CI=true
        
        # Reduce parallelism for stability
        export MAKEFLAGS="-j2"
        
        log_success "CI optimizations applied"
    fi
}

# Function to select appropriate requirements file
select_requirements_file() {
    local base_dir="$1"
    local tier="$2"
    
    case "$tier" in
        "ci")
            if [[ -f "$base_dir/requirements-ci.txt" ]]; then
                echo "$base_dir/requirements-ci.txt"
                return 0
            fi
            ;;
        "dev")
            if [[ -f "$base_dir/requirements-dev.txt" ]]; then
                echo "$base_dir/requirements-dev.txt"
                return 0
            fi
            ;;
        "full")
            echo "$base_dir/requirements.txt"
            return 0
            ;;
    esac
    
    # Fallback to main requirements
    echo "$base_dir/requirements.txt"
}

# Function to install Python dependencies
install_python_dependencies() {
    local pipelines_dir="$WORKSPACE_ROOT/pipelines"
    
    if [[ ! -d "$pipelines_dir" ]]; then
        log_warning "Python pipelines directory not found. Skipping Python dependencies."
        return 0
    fi
    
    log "Installing Python dependencies (tier: $DEPENDENCY_TIER)..."
    
    # Upgrade pip first
    run_with_retry \
        "python -m pip install --upgrade pip setuptools wheel" \
        "Upgrading pip and build tools" \
        2 \
        120
    
    # Select appropriate requirements file
    local requirements_file
    requirements_file=$(select_requirements_file "$pipelines_dir" "$DEPENDENCY_TIER")
    
    if [[ ! -f "$requirements_file" ]]; then
        log_error "Requirements file not found: $requirements_file"
        return 1
    fi
    
    log "Using requirements file: $requirements_file"
    
    # Install main requirements
    run_with_retry \
        "cd '$pipelines_dir' && pip install -r '$(basename "$requirements_file")'" \
        "Installing Python requirements" \
        3 \
        600
    
    # Install additional dev requirements if not using CI tier
    if [[ "$DEPENDENCY_TIER" == "dev" || "$DEPENDENCY_TIER" == "full" ]]; then
        local dev_requirements="$pipelines_dir/requirements-dev.txt"
        if [[ -f "$dev_requirements" ]]; then
            run_with_retry \
                "cd '$pipelines_dir' && pip install -r requirements-dev.txt" \
                "Installing Python dev requirements" \
                3 \
                600
        fi
    fi
    
    log_success "Python dependencies installed successfully"
}

# Function to install Node.js dependencies
install_nodejs_dependencies() {
    local site_dir="$WORKSPACE_ROOT/apps/site"
    
    if [[ ! -d "$site_dir" ]]; then
        log_warning "Site directory not found. Skipping Node.js dependencies."
        return 0
    fi
    
    if [[ ! -f "$site_dir/package.json" ]]; then
        log_warning "package.json not found. Skipping Node.js dependencies."
        return 0
    fi
    
    log "Installing Node.js dependencies..."
    
    # Clear npm cache if in CI
    if [[ "${CI:-}" == "true" ]]; then
        run_with_retry \
            "npm cache clean --force" \
            "Clearing npm cache" \
            1 \
            60
    fi
    
    # Install dependencies using npm ci in CI, npm install locally
    local install_cmd="npm install"
    if [[ "${CI:-}" == "true" && -f "$site_dir/package-lock.json" ]]; then
        install_cmd="npm ci"
    fi
    
    run_with_retry \
        "cd '$site_dir' && $install_cmd" \
        "Installing Node.js dependencies" \
        3 \
        600
    
    log_success "Node.js dependencies installed successfully"
}

# Function to verify installations
verify_installations() {
    log "Verifying installations..."
    
    # Verify Python packages
    if command -v python3 > /dev/null; then
        if python3 -c "import pipelines" 2>/dev/null; then
            log_success "Python pipelines module accessible"
        else
            log_warning "Python pipelines module not accessible"
        fi
    fi
    
    # Verify Node.js packages
    if [[ -d "$WORKSPACE_ROOT/apps/site/node_modules" ]]; then
        local node_modules_size=$(du -sm "$WORKSPACE_ROOT/apps/site/node_modules" | cut -f1)
        log_success "Node.js modules installed (${node_modules_size}MB)"
    else
        log_warning "Node.js modules directory not found"
    fi
    
    # Check for common executables
    local tools=("pytest" "black" "isort" "mypy" "astro")
    for tool in "${tools[@]}"; do
        if command -v "$tool" > /dev/null; then
            log_success "$tool is available"
        else
            log_warning "$tool is not available"
        fi
    done
}

# Function to generate installation report
generate_installation_report() {
    local report_file="$WORKSPACE_ROOT/dependency-installation-report.md"
    
    cat > "$report_file" << EOF
# Dependency Installation Report

**Generated:** $(date)
**Tier:** $DEPENDENCY_TIER
**Workspace:** $WORKSPACE_ROOT

## Summary

- **Python Dependencies:** $(pip list 2>/dev/null | wc -l) packages installed
- **Node.js Dependencies:** $(find "$WORKSPACE_ROOT/apps/site/node_modules" -maxdepth 1 -type d 2>/dev/null | wc -l) packages installed
- **Installation Log:** $LOG_FILE

## Environment

- **Python Version:** $(python3 --version 2>/dev/null || echo "Not available")
- **Node.js Version:** $(node --version 2>/dev/null || echo "Not available")
- **NPM Version:** $(npm --version 2>/dev/null || echo "Not available")
- **Pip Version:** $(pip --version 2>/dev/null || echo "Not available")

## Verification Results

$(verify_installations 2>&1 | grep -E "(✅|⚠️|❌)" || echo "No verification results")

## Installation Time

- **Start:** $(head -1 "$LOG_FILE" 2>/dev/null | grep -o '\[.*\]' | tr -d '[]' || echo "Unknown")
- **End:** $(tail -1 "$LOG_FILE" 2>/dev/null | grep -o '\[.*\]' | tr -d '[]' || echo "Unknown")

---
*Generated by install-dependencies.sh*
EOF

    log_success "Installation report generated: $report_file"
}

# Main function
main() {
    local start_time=$(date +%s)
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tier)
                DEPENDENCY_TIER="$2"
                shift 2
                ;;
            --workspace)
                WORKSPACE_ROOT="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT_DURATION="$2"
                shift 2
                ;;
            --log)
                LOG_FILE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --tier TIER        Dependency tier: ci, dev, full (default: ci)"
                echo "  --workspace DIR    Workspace root directory (default: current)"
                echo "  --timeout SECONDS  Command timeout in seconds (default: 300)"
                echo "  --log FILE         Log file path (default: /tmp/dependency-install.log)"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Initialize
    log "Starting dependency installation (tier: $DEPENDENCY_TIER)"
    log "Workspace: $WORKSPACE_ROOT"
    log "Log file: $LOG_FILE"
    
    # Pre-flight checks
    check_disk_space 500 || exit 1
    optimize_for_ci
    
    # Install dependencies
    local success=true
    
    if ! install_python_dependencies; then
        log_error "Python dependency installation failed"
        success=false
    fi
    
    if ! install_nodejs_dependencies; then
        log_error "Node.js dependency installation failed"
        success=false
    fi
    
    # Verification and reporting
    verify_installations
    generate_installation_report
    
    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    if [[ "$success" == "true" ]]; then
        log_success "All dependencies installed successfully in ${elapsed_min}m${elapsed_sec}s"
        exit 0
    else
        log_error "Some dependency installations failed. Check log for details."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"