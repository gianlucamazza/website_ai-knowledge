#!/usr/bin/env python3
"""Fix markdown linting issues for article files."""

import re
from pathlib import Path

def fix_markdown_file(filepath):
    """Fix common markdown linting issues."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix MD013: Break long lines at 120 characters
        if len(line.rstrip()) > 120:
            # Try to break at sensible points (periods, commas, spaces)
            if '. ' in line:
                parts = line.split('. ')
                new_lines = []
                current = parts[0]
                for part in parts[1:]:
                    if len(current + '. ' + part) < 120:
                        current += '. ' + part
                    else:
                        new_lines.append(current + '.')
                        current = part
                if current:
                    new_lines.append(current)
                fixed_lines.extend([l + '\n' if not l.endswith('\n') else l for l in new_lines])
            else:
                # Simple word wrap
                words = line.split()
                current_line = []
                for word in words:
                    if sum(len(w) + 1 for w in current_line) + len(word) < 120:
                        current_line.append(word)
                    else:
                        if current_line:
                            fixed_lines.append(' '.join(current_line) + '\n')
                        current_line = [word]
                if current_line:
                    fixed_lines.append(' '.join(current_line) + '\n')
        # Fix MD032: Add blank lines around lists
        elif line.strip().startswith('- ') and i > 0:
            prev_line = lines[i-1] if i > 0 else ''
            # If previous line is not empty and not another list item, add blank line
            if prev_line.strip() and not prev_line.strip().startswith('- '):
                if not (len(fixed_lines) > 0 and fixed_lines[-1].strip() == ''):
                    fixed_lines.append('\n')
            fixed_lines.append(line)
        # Fix MD040: Add language to code blocks
        elif line.strip() == '```':
            # Check if this is opening or closing
            # Look ahead to see if next line looks like code
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() and not next_line.startswith('```'):
                    # This is opening, needs language
                    fixed_lines.append('```text\n')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        # Fix MD031: Add blank lines around code blocks
        elif line.strip().startswith('```'):
            # Add blank line before if needed
            if fixed_lines and fixed_lines[-1].strip() and not fixed_lines[-1].strip().startswith('```'):
                if not (len(fixed_lines) > 1 and fixed_lines[-1].strip() == ''):
                    fixed_lines.append('\n')
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {filepath}")

# Fix the problematic article
article_path = Path('/Users/gianlucamazza/Workspace/website_ai-knowledge/apps/site/src/content/articles/choosing-right-ml-algorithm.md')
if article_path.exists():
    fix_markdown_file(article_path)

# Fix other articles too
articles_dir = Path('/Users/gianlucamazza/Workspace/website_ai-knowledge/apps/site/src/content/articles')
for article in articles_dir.glob('*.md'):
    print(f"Processing {article.name}...")
    fix_markdown_file(article)