#!/bin/bash

# Improved dependency installation script with better error handling and tiered approach
# Addresses CI failures and dependency conflicts

set -euo pipefail

# Configuration
MAX_RETRIES=2  # Reduced retries for faster feedback
TIMEOUT_DURATION=300
LOG_FILE="${LOG_FILE:-/tmp/dependency-install.log}"
DEPENDENCY_TIER="${DEPENDENCY_TIER:-ci}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Enhanced logging functions
log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}${msg}${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1"
    echo -e "${GREEN}${msg}${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1"
    echo -e "${YELLOW}${msg}${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1"
    echo -e "${RED}${msg}${NC}" | tee -a "$LOG_FILE"
}

# Improved error handling function
run_with_enhanced_retry() {
    local cmd="$1"
    local description="$2"
    local max_attempts="${3:-$MAX_RETRIES}"
    local timeout="${4:-$TIMEOUT_DURATION}"
    
    local attempt=1
    local temp_log="/tmp/cmd_output_$$.log"
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Attempt $attempt/$max_attempts: $description"
        
        # Run command with both file and stdout logging for CI visibility
        if [[ "$VERBOSE" == "true" || "${CI:-}" == "true" ]]; then
            # In CI or verbose mode, show output in real-time
            if timeout "$timeout" bash -c "$cmd" 2>&1 | tee -a "$LOG_FILE" "$temp_log"; then
                log_success "$description completed successfully"
                rm -f "$temp_log"
                return 0
            fi
        else
            # In quiet mode, capture output but show errors
            if timeout "$timeout" bash -c "$cmd" > "$temp_log" 2>&1; then
                cat "$temp_log" >> "$LOG_FILE"
                log_success "$description completed successfully"
                rm -f "$temp_log"
                return 0
            fi
        fi
        
        local exit_code=$?
        
        # Always show errors immediately for debugging
        echo "=== ERROR OUTPUT ===" | tee -a "$LOG_FILE"
        tail -50 "$temp_log" | tee -a "$LOG_FILE"
        echo "=================" | tee -a "$LOG_FILE"
        
        if [[ $exit_code -eq 124 ]]; then
            log_error "$description timed out after ${timeout}s"
        else
            log_error "$description failed with exit code $exit_code"
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            local delay=$((attempt * 5))  # Reduced delay
            log_warning "Waiting ${delay}s before retry..."
            sleep $delay
        fi
        
        ((attempt++))
    done
    
    rm -f "$temp_log"
    log_error "$description failed after $max_attempts attempts"
    return 1
}

# Tiered installation strategy
install_python_dependencies_tiered() {
    local pipelines_dir="$WORKSPACE_ROOT/pipelines"
    
    if [[ ! -d "$pipelines_dir" ]]; then
        log_warning "Python pipelines directory not found. Skipping Python dependencies."
        return 0
    fi
    
    log "Installing Python dependencies using tiered approach (tier: $DEPENDENCY_TIER)..."
    
    # Always upgrade pip first with minimal output
    run_with_enhanced_retry \
        "python -m pip install --upgrade pip setuptools wheel --quiet --no-warn-script-location" \
        "Upgrading pip and build tools" \
        1 \
        60
    
    cd "$pipelines_dir"
    
    case "$DEPENDENCY_TIER" in
        "ci")
            # CI tier: minimal dependencies for testing
            if [[ -f "requirements-core.txt" ]]; then
                run_with_enhanced_retry \
                    "pip install -r requirements-core.txt --quiet --no-warn-script-location" \
                    "Installing core dependencies" \
                    2 \
                    300
            fi
            
            if [[ -f "requirements-ci.txt" ]]; then
                run_with_enhanced_retry \
                    "pip install -r requirements-ci.txt --quiet --no-warn-script-location" \
                    "Installing CI testing dependencies" \
                    2 \
                    300
            else
                # Fallback to existing files
                run_with_enhanced_retry \
                    "pip install -r requirements.txt --quiet --no-warn-script-location" \
                    "Installing main requirements" \
                    2 \
                    300
                    
                # Install minimal dev dependencies
                run_with_enhanced_retry \
                    "pip install pytest==7.4.3 pytest-asyncio==0.21.1 pytest-mock==3.12.0 pytest-cov==4.1.0 coverage[toml]==7.3.2 black==23.11.0 isort==5.12.0 safety==2.3.5 --quiet --no-warn-script-location" \
                    "Installing essential dev dependencies" \
                    2 \
                    300
            fi
            ;;
            
        "dev")
            # Dev tier: core + dev dependencies, excluding AI/ML
            if [[ -f "requirements-core.txt" ]]; then
                run_with_enhanced_retry \
                    "pip install -r requirements-core.txt --quiet" \
                    "Installing core dependencies" \
                    2 \
                    300
                    
                run_with_enhanced_retry \
                    "pip install -r requirements-dev.txt --quiet" \
                    "Installing dev dependencies" \
                    2 \
                    400
            else
                # Fallback approach
                run_with_enhanced_retry \
                    "pip install -r requirements.txt --quiet" \
                    "Installing main requirements" \
                    2 \
                    600
                    
                run_with_enhanced_retry \
                    "pip install -r requirements-dev.txt --quiet" \
                    "Installing dev requirements" \
                    2 \
                    400
            fi
            ;;
            
        "full")
            # Full tier: everything including AI/ML
            run_with_enhanced_retry \
                "pip install -r requirements.txt --quiet" \
                "Installing all requirements" \
                3 \
                900
                
            if [[ -f "requirements-ai.txt" ]]; then
                run_with_enhanced_retry \
                    "pip install -r requirements-ai.txt --quiet" \
                    "Installing AI/ML dependencies" \
                    2 \
                    600
            fi
            
            run_with_enhanced_retry \
                "pip install -r requirements-dev.txt --quiet" \
                "Installing dev requirements" \
                2 \
                400
            ;;
    esac
    
    log_success "Python dependencies installed successfully (tier: $DEPENDENCY_TIER)"
}

# Enhanced dependency verification
verify_installations_enhanced() {
    log "Performing enhanced dependency verification..."
    
    local verification_failed=false
    
    # Check critical packages by import
    local critical_packages=("pytest" "coverage")
    
    if [[ "$DEPENDENCY_TIER" != "ci" ]]; then
        critical_packages+=("black" "isort")
    fi
    
    for package in "${critical_packages[@]}"; do
        if python -c "import ${package}" 2>/dev/null; then
            log_success "✓ ${package} importable"
        else
            log_error "✗ ${package} not importable"
            verification_failed=true
        fi
    done
    
    # Check package versions for conflicts
    log "Checking for version conflicts..."
    if python -c "
import pkg_resources
import sys

conflicts = []
try:
    pkg_resources.require(['safety==2.3.5'])
    print('✓ Safety version compatible')
except Exception as e:
    conflicts.append(f'Safety: {e}')

if conflicts:
    print('❌ Version conflicts detected:')
    for conflict in conflicts:
        print(f'  - {conflict}')
    sys.exit(1)
else:
    print('✓ No critical version conflicts detected')
" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Version compatibility check passed"
    else
        log_error "Version conflicts detected"
        verification_failed=true
    fi
    
    if [[ "$verification_failed" == "true" ]]; then
        return 1
    fi
    
    return 0
}

# Main function with enhanced error handling
main() {
    local start_time=$(date +%s)
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tier)
                DEPENDENCY_TIER="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --workspace)
                WORKSPACE_ROOT="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT_DURATION="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --tier TIER        Dependency tier: ci, dev, full (default: ci)"
                echo "  --verbose          Show detailed output"
                echo "  --workspace DIR    Workspace root directory"
                echo "  --timeout SECONDS  Command timeout in seconds"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Initialize logging
    echo "Starting dependency installation ($(date))" > "$LOG_FILE"
    log "Starting dependency installation (tier: $DEPENDENCY_TIER)"
    log "Workspace: $WORKSPACE_ROOT"
    log "Verbose mode: $VERBOSE"
    
    # Set CI optimizations
    if [[ "${CI:-}" == "true" ]]; then
        log "CI environment detected - enabling optimizations"
        export PIP_NO_CACHE_DIR=1
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        export PIP_QUIET=1
    fi
    
    # Install Python dependencies with enhanced error handling
    if ! install_python_dependencies_tiered; then
        log_error "Python dependency installation failed"
        
        # Show recent log for debugging
        echo "=== RECENT LOG OUTPUT ===" 
        tail -100 "$LOG_FILE"
        echo "========================="
        
        exit 1
    fi
    
    # Enhanced verification
    if ! verify_installations_enhanced; then
        log_error "Dependency verification failed"
        exit 1
    fi
    
    # Success
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    log_success "All dependencies installed and verified successfully in ${elapsed}s"
    log "Installation log: $LOG_FILE"
    
    exit 0
}

# Execute main with all arguments
main "$@"