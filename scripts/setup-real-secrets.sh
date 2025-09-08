#!/bin/bash

# Setup Real Secrets for Act Testing
# ==================================
# This script helps you securely configure real API keys and services for local testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECRETS_FILE="$PROJECT_ROOT/.secrets"
SECRETS_TEMPLATE="$PROJECT_ROOT/.secrets.example"

echo "🔐 Setting up Real Secrets for Act Testing"
echo "=========================================="

# Check if running in a secure environment
if [ "${CI:-false}" = "true" ] && [ "${ACT:-false}" != "true" ]; then
    echo "❌ This script should not be run in CI environments!"
    echo "   It's designed for local development with act only."
    exit 1
fi

# Warn about security implications
echo ""
echo "⚠️  SECURITY NOTICE"
echo "=================="
echo "This script will help you configure REAL API keys and credentials"
echo "for production-like local testing with act (GitHub Actions runner)."
echo ""
echo "IMPORTANT SECURITY CONSIDERATIONS:"
echo "- Only use this for LOCAL DEVELOPMENT on secure machines"
echo "- Never commit the .secrets file to version control"
echo "- Use dedicated test/development API keys when possible"
echo "- Rotate any keys that might have been compromised"
echo "- Consider using separate accounts for testing"
echo ""

# Confirm the user wants to proceed
read -p "Do you want to continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "👋 Setup cancelled."
    exit 0
fi

# Check if .secrets already exists
if [ -f "$SECRETS_FILE" ]; then
    echo ""
    echo "📄 Existing .secrets file found."
    echo "   Current file: $SECRETS_FILE"
    echo ""
    read -p "Do you want to backup and replace it? (y/N): " replace_confirm
    if [[ "$replace_confirm" =~ ^[Yy]$ ]]; then
        backup_file="$SECRETS_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$SECRETS_FILE" "$backup_file"
        echo "✅ Backup created: $backup_file"
    else
        echo "📝 Continuing with existing file..."
    fi
fi

# Copy template if user wants to replace or file doesn't exist
if [ ! -f "$SECRETS_FILE" ] || [[ "$replace_confirm" =~ ^[Yy]$ ]]; then
    cp "$SECRETS_TEMPLATE" "$SECRETS_FILE"
    echo "📋 Created $SECRETS_FILE from template"
fi

# Make sure .secrets is in .gitignore
GITIGNORE_FILE="$PROJECT_ROOT/.gitignore"
if [ -f "$GITIGNORE_FILE" ]; then
    if ! grep -q "^\.secrets$" "$GITIGNORE_FILE"; then
        echo "" >> "$GITIGNORE_FILE"
        echo "# Act secrets (never commit real secrets!)" >> "$GITIGNORE_FILE"
        echo ".secrets" >> "$GITIGNORE_FILE"
        echo "✅ Added .secrets to .gitignore"
    fi
else
    echo ".secrets" > "$GITIGNORE_FILE"
    echo "✅ Created .gitignore with .secrets"
fi

# Interactive configuration helper
echo ""
echo "🔧 Configuration Helper"
echo "======================"
echo ""
echo "The .secrets file has been created with placeholder values."
echo "You'll need to replace the placeholder values with real credentials."
echo ""
echo "Here are the most critical ones to configure:"
echo ""

# Array of critical secrets to configure
declare -A critical_secrets=(
    ["DATABASE_URL"]="Database connection string (Supabase/PostgreSQL)"
    ["OPENAI_API_KEY"]="OpenAI API key for AI operations"
    ["ANTHROPIC_API_KEY"]="Anthropic API key for Claude operations"
    ["GITHUB_TOKEN"]="GitHub Personal Access Token"
    ["CODECOV_TOKEN"]="Codecov token for coverage reporting"
)

echo "Critical Secrets to Configure:"
echo "=============================="
for key in "${!critical_secrets[@]}"; do
    echo "- $key: ${critical_secrets[$key]}"
done

echo ""
echo "📝 Configuration Options:"
echo ""
echo "1. 🖊️  Edit manually: $SECRETS_FILE"
echo "2. 🔍 View template:   $SECRETS_TEMPLATE"
echo "3. 🚀 Test with act:   make act-test"
echo ""

# Offer to open the secrets file in editor
if command -v code >/dev/null 2>&1; then
    read -p "Open .secrets in VS Code for editing? (y/N): " open_vscode
    if [[ "$open_vscode" =~ ^[Yy]$ ]]; then
        code "$SECRETS_FILE"
    fi
elif command -v nano >/dev/null 2>&1; then
    read -p "Open .secrets in nano for editing? (y/N): " open_nano
    if [[ "$open_nano" =~ ^[Yy]$ ]]; then
        nano "$SECRETS_FILE"
    fi
fi

# Validate secrets file
echo ""
echo "🔍 Validating secrets configuration..."

# Check for placeholder values that need to be replaced
placeholder_count=0
while IFS= read -r line; do
    if [[ "$line" =~ \[YOUR_.*\] ]] && [[ ! "$line" =~ ^# ]]; then
        if [ $placeholder_count -eq 0 ]; then
            echo ""
            echo "⚠️  Found placeholder values that need to be replaced:"
        fi
        echo "   $line"
        ((placeholder_count++))
    fi
done < "$SECRETS_FILE"

if [ $placeholder_count -gt 0 ]; then
    echo ""
    echo "📝 Please replace the $placeholder_count placeholder values before running act tests."
else
    echo "✅ No placeholder values found - configuration looks good!"
fi

# Security recommendations
echo ""
echo "🛡️  Security Recommendations"
echo "============================"
echo ""
echo "1. 🔐 Use dedicated test/development API keys when possible"
echo "2. 🏠 Only run this on secure, personal development machines"
echo "3. 🔄 Rotate keys regularly, especially if they may be compromised"
echo "4. 📱 Enable 2FA on all accounts with API access"
echo "5. 🚫 Never commit .secrets to version control"
echo "6. 🗑️  Delete .secrets when not actively testing"
echo ""

# Service-specific setup guides
echo "📚 Service Setup Guides"
echo "======================"
echo ""
echo "Supabase Database:"
echo "- Create project: https://app.supabase.com"
echo "- Get connection string from Settings > Database"
echo "- Copy URL and anon key from Settings > API"
echo ""
echo "OpenAI API:"
echo "- Get API key: https://platform.openai.com/api-keys"
echo "- Consider setting usage limits for safety"
echo ""
echo "Anthropic API:"
echo "- Get API key: https://console.anthropic.com"
echo "- Note: Requires account approval"
echo ""
echo "GitHub Token:"
echo "- Create at: https://github.com/settings/tokens"
echo "- Scopes needed: repo, actions, security_events"
echo ""

# Final instructions
echo ""
echo "✅ Setup completed!"
echo ""
echo "Next steps:"
echo "==========="
echo "1. Edit $SECRETS_FILE with your real credentials"
echo "2. Run 'make act-test' to test your configuration"
echo "3. Run 'make act-list' to see available workflows"
echo "4. Use 'make act-pr' to simulate a full PR workflow"
echo ""
echo "Happy testing! 🚀"