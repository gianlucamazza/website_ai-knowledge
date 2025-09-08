#!/usr/bin/env python3
"""
Fix common markdown linting violations in content files.
This script addresses MD025, MD036, and MD013 violations.
"""

import os
import re
import sys
from pathlib import Path


def fix_md025_duplicate_h1(content: str, title: str) -> str:
    """Remove duplicate H1 headings if they match the frontmatter title."""
    lines = content.split('\n')
    new_lines = []
    found_frontmatter_end = False
    
    for line in lines:
        # Skip removing H1 until after frontmatter
        if line.strip() == '---' and found_frontmatter_end:
            found_frontmatter_end = True
            new_lines.append(line)
            continue
        elif line.strip() == '---':
            found_frontmatter_end = True
            new_lines.append(line)
            continue
            
        # Remove H1 that duplicates the title from frontmatter
        if found_frontmatter_end and line.startswith('# '):
            h1_text = line[2:].strip().strip('"')
            title_clean = title.strip().strip('"')
            if h1_text == title_clean:
                # Skip this duplicate H1
                continue
                
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_md036_emphasis_as_heading(content: str) -> str:
    """Convert emphasized text that should be headings to proper headings."""
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        # Look for lines that are just emphasized text followed by content
        if re.match(r'^\*\*[^*]+\*\*\s*:?\s*$', line.strip()):
            # Check if this is followed by text (indicating it should be a heading)
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if next_line.strip() and not next_line.strip().startswith('**'):
                # Convert to heading
                heading_text = line.strip()[2:-2]  # Remove ** from both ends
                heading_text = heading_text.rstrip(':').strip()  # Remove trailing colon
                # Determine heading level based on context (use H4 for subsections)
                new_lines.append(f"#### {heading_text}")
                continue
                
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_md013_line_length(content: str, max_length: int = 120) -> str:
    """Break long lines while preserving code blocks and tables."""
    lines = content.split('\n')
    new_lines = []
    in_code_block = False
    in_frontmatter = False
    
    for line in lines:
        # Track code blocks and frontmatter
        if line.strip() == '```' or line.startswith('```'):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if line.strip() == '---':
            in_frontmatter = not in_frontmatter
            new_lines.append(line)
            continue
            
        # Skip processing if in code block, frontmatter, or table
        if (in_code_block or in_frontmatter or 
            line.strip().startswith('|') or  # Table
            line.strip().startswith('<') or   # HTML
            line.strip().startswith('#')):    # Headings - leave as is
            new_lines.append(line)
            continue
            
        # Fix long lines
        if len(line) > max_length and line.strip():
            # Simple word wrap for regular text
            words = line.split()
            current_line = ""
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            
            for word in words:
                if not current_line:
                    current_line = indent_str + word
                elif len(current_line + " " + word) <= max_length:
                    current_line += " " + word
                else:
                    new_lines.append(current_line)
                    current_line = indent_str + word
                    
            if current_line:
                new_lines.append(current_line)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def extract_title_from_frontmatter(content: str) -> str:
    """Extract the title from YAML frontmatter."""
    lines = content.split('\n')
    in_frontmatter = False
    
    for line in lines:
        if line.strip() == '---':
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                break
                
        if in_frontmatter and line.startswith('title:'):
            title = line.split('title:', 1)[1].strip()
            return title.strip('\'"')
            
    return ""


def fix_md022_heading_spacing(content: str) -> str:
    """Fix heading spacing issues."""
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if line.startswith('#') and line.strip():
            # Add blank line before heading if missing (except first heading or after frontmatter)
            prev_line = lines[i - 1] if i > 0 else ""
            if (i > 0 and prev_line.strip() and 
                prev_line.strip() != '---' and 
                not prev_line.startswith('#')):
                new_lines.append("")
            
            new_lines.append(line)
            
            # Add blank line after heading if missing
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if next_line.strip() and not next_line.startswith('#'):
                new_lines.append("")
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_md026_trailing_punctuation(content: str) -> str:
    """Remove trailing punctuation from headings."""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.startswith('#') and line.strip():
            # Remove trailing punctuation from headings
            cleaned = line.rstrip(':.!?')
            new_lines.append(cleaned)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_md029_ordered_list_prefix(content: str) -> str:
    """Fix ordered list numbering."""
    lines = content.split('\n')
    new_lines = []
    in_ordered_list = False
    list_counter = 1
    
    for i, line in enumerate(lines):
        # Check if line is part of ordered list
        if re.match(r'^\d+\.\s+', line.strip()):
            in_ordered_list = True
            # Replace with correct sequential number
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            content_part = re.sub(r'^\d+\.\s+', '', line.strip())
            new_lines.append(f"{indent_str}{list_counter}. {content_part}")
            list_counter += 1
        elif line.strip() == "" and in_ordered_list:
            # Blank line - check if list continues
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if not re.match(r'^\d+\.\s+', next_line.strip()):
                # List ended
                in_ordered_list = False
                list_counter = 1
            new_lines.append(line)
        elif in_ordered_list and not line.strip():
            new_lines.append(line)
        elif in_ordered_list:
            # Non-list line - list ended
            in_ordered_list = False
            list_counter = 1
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_markdown_file(file_path: Path) -> bool:
    """Fix markdown violations in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        title = extract_title_from_frontmatter(content)
        
        # Apply fixes in logical order
        content = fix_md025_duplicate_h1(content, title)
        content = fix_md036_emphasis_as_heading(content)
        content = fix_md026_trailing_punctuation(content)
        content = fix_md022_heading_spacing(content)
        content = fix_md029_ordered_list_prefix(content)
        content = fix_md013_line_length(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix markdown violations."""
    if len(sys.argv) > 1:
        content_dir = Path(sys.argv[1])
    else:
        content_dir = Path("apps/site/src/content")
        
    if not content_dir.exists():
        print(f"Content directory not found: {content_dir}")
        return 1
        
    print(f"Fixing markdown violations in: {content_dir}")
    
    # Find all markdown files
    md_files = list(content_dir.rglob("*.md"))
    
    if not md_files:
        print("No markdown files found")
        return 1
        
    fixed_count = 0
    for md_file in md_files:
        if fix_markdown_file(md_file):
            fixed_count += 1
            
    print(f"\nProcessed {len(md_files)} files, fixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())