#!/bin/bash

# Act Installation Script
# Installs Act (GitHub Actions local runner) on macOS, Linux, and Windows (WSL)

set -e

echo "🚀 Installing Act - GitHub Actions Local Runner"
echo "=============================================="

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "📍 Detected OS: $OS"

# Install Act based on OS
case $OS in
    "macos")
        echo "🍺 Installing Act via Homebrew..."
        if command -v brew >/dev/null 2>&1; then
            brew install act
        else
            echo "❌ Homebrew not found. Installing via script..."
            curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
        fi
        ;;
    "linux")
        echo "🐧 Installing Act via script..."
        curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
        ;;
    "windows")
        echo "🪟 Installing Act via Chocolatey or Scoop..."
        if command -v choco >/dev/null 2>&1; then
            choco install act-cli
        elif command -v scoop >/dev/null 2>&1; then
            scoop install act
        else
            echo "❌ Please install Chocolatey or Scoop first, then run:"
            echo "   choco install act-cli"
            echo "   or"
            echo "   scoop install act"
            exit 1
        fi
        ;;
    *)
        echo "❌ Unsupported OS. Please install Act manually:"
        echo "   https://github.com/nektos/act#installation"
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "🔍 Verifying installation..."
if command -v act >/dev/null 2>&1; then
    echo "✅ Act installed successfully!"
    echo "📋 Version: $(act --version)"
else
    echo "❌ Act installation failed!"
    exit 1
fi

# Check Docker
echo ""
echo "🐳 Checking Docker installation..."
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker is installed"
    echo "📋 Version: $(docker --version)"
    
    # Test Docker connectivity
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon is running"
    else
        echo "⚠️  Docker daemon is not running. Please start Docker."
    fi
else
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Pull required Docker images
echo ""
echo "📥 Pulling required Docker images..."
docker pull catthehacker/ubuntu:act-latest
docker pull postgres:15

echo ""
echo "✅ Act setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Copy .secrets.example to .secrets and fill in values"
echo "2. Run 'make act-test' to test local workflows"
echo "3. Use 'act -l' to list available workflows"
echo ""
echo "🔧 Common commands:"
echo "   act -l                           # List workflows"
echo "   act pull_request                 # Run PR workflows"
echo "   act push                         # Run push workflows"
echo "   act -j python-tests             # Run specific job"
echo "   act --dry-run                   # Dry run (no execution)"