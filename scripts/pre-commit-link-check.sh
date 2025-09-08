#!/bin/bash
# Pre-commit Link Check for AI Knowledge Website
# Fast local validation of internal links before commit

set -e

# Configuration
CONTENT_DIR="apps/site/src/content"
MAX_LINKS_TO_CHECK=50
TIMEOUT=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔗 Pre-commit link validation..."

# Check if content directory exists
if [ ! -d "$CONTENT_DIR" ]; then
    echo -e "${RED}Error: Content directory not found: $CONTENT_DIR${NC}"
    exit 1
fi

# Function to check internal links
check_internal_links() {
    local file="$1"
    local errors=0
    
    # Extract internal links to glossary entries
    local links=$(grep -o '](/apps/site/src/content/glossary/[^)]*\.md)' "$file" 2>/dev/null || true)
    
    if [ -n "$links" ]; then
        while read -r link; do
            # Extract the path from the link
            local path=$(echo "$link" | sed 's/](//; s/)//')
            local full_path="$path"
            
            if [ ! -f "$full_path" ]; then
                echo -e "${RED}  ❌ Broken link: $path${NC}"
                errors=$((errors + 1))
            fi
        done <<< "$links"
    fi
    
    return $errors
}

# Function to validate frontmatter links
check_frontmatter_links() {
    local file="$1"
    local errors=0
    
    # Check 'related' field in frontmatter
    local related_entries=$(sed -n '/^related:/,/^[a-zA-Z]/p' "$file" | grep -E '^\s*-\s*"[^"]*"' | sed 's/^\s*-\s*"\([^"]*\)".*/\1/' || true)
    
    if [ -n "$related_entries" ]; then
        while read -r entry; do
            if [ -n "$entry" ]; then
                local related_file="$CONTENT_DIR/glossary/${entry}.md"
                if [ ! -f "$related_file" ]; then
                    echo -e "${RED}  ❌ Related entry not found: $entry${NC}"
                    errors=$((errors + 1))
                fi
            fi
        done <<< "$related_entries"
    fi
    
    return $errors
}

# Main validation loop
total_errors=0
files_checked=0
links_checked=0

echo "Checking modified markdown files..."

# Get list of modified markdown files
if [ $# -eq 0 ]; then
    # No files passed, check all modified files in git
    files=$(git diff --cached --name-only --diff-filter=AM | grep '\.md$' | grep -E '^(apps/site/src/content|docs|temp-articles)/' || true)
else
    # Files passed as arguments
    files="$*"
fi

if [ -z "$files" ]; then
    echo -e "${GREEN}✅ No markdown files to check${NC}"
    exit 0
fi

for file in $files; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    files_checked=$((files_checked + 1))
    echo "  Checking: $file"
    
    # Check internal links
    check_internal_links "$file"
    link_errors=$?
    total_errors=$((total_errors + link_errors))
    
    # Check frontmatter links (only for glossary entries)
    if [[ "$file" == *"/glossary/"* ]]; then
        check_frontmatter_links "$file"
        fm_errors=$?
        total_errors=$((total_errors + fm_errors))
    fi
    
    # Count links checked
    local file_links=$(grep -o '](.*\.md)' "$file" 2>/dev/null | wc -l || echo "0")
    links_checked=$((links_checked + file_links))
    
    # Limit check to avoid long delays
    if [ $links_checked -gt $MAX_LINKS_TO_CHECK ]; then
        echo -e "${YELLOW}⚠️  Reached link check limit ($MAX_LINKS_TO_CHECK), skipping remaining${NC}"
        break
    fi
done

# Summary
echo ""
echo "📊 Link check summary:"
echo "  Files checked: $files_checked"
echo "  Links validated: $links_checked"

if [ $total_errors -eq 0 ]; then
    echo -e "${GREEN}✅ All links valid${NC}"
    exit 0
else
    echo -e "${RED}❌ Found $total_errors broken links${NC}"
    echo ""
    echo "💡 Fix suggestions:"
    echo "  1. Check that referenced glossary entries exist"
    echo "  2. Verify file paths are correct"
    echo "  3. Ensure related entries in frontmatter match actual file names"
    echo "  4. Run 'make build' to validate all content"
    exit 1
fi