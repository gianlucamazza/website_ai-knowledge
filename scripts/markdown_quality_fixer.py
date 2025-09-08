#!/usr/bin/env python3
"""
Markdown Quality Fixer for AI Knowledge Website
Automatically fixes common markdown linting issues while preserving content integrity.
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
from datetime import datetime

class MarkdownQualityFixer:
    """Enterprise-grade markdown quality fixer for AI technical content."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.fixes_applied = []
        self.files_processed = 0
        self.total_fixes = 0
        
    def fix_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Fix markdown violations in a single file."""
        if not filepath.exists():
            return False, [f"File not found: {filepath}"]
            
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        content = original_content
        fixes = []
        
        # Fix MD047: Files should end with a single newline
        if not content.endswith('\n'):
            content += '\n'
            fixes.append("MD047: Added missing final newline")
        elif content.endswith('\n\n'):
            content = content.rstrip() + '\n'
            fixes.append("MD047: Fixed multiple trailing newlines")
            
        # Fix MD032: Lists should be surrounded by blank lines
        content = self._fix_list_spacing(content)
        if content != original_content:
            fixes.append("MD032: Fixed list spacing")
            
        # Fix MD031: Code blocks should be surrounded by blank lines
        content = self._fix_code_block_spacing(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD031: Fixed code block spacing")
            
        # Fix MD022: Headings should be surrounded by blank lines
        content = self._fix_heading_spacing(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD022: Fixed heading spacing")
            
        # Fix MD026: Trailing punctuation in headings
        content = self._fix_heading_punctuation(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD026: Removed trailing punctuation from headings")
            
        # Fix MD036: Emphasis used instead of headings
        content = self._fix_emphasis_as_heading(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD036: Converted emphasis to proper headings")
            
        # Fix MD013: Line length (smart wrapping for AI content)
        content = self._fix_line_length(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD013: Fixed line length violations")
            
        # Fix MD040: Code blocks should have language specified
        content = self._fix_code_language(content)
        if len(fixes) == 0 and content != original_content:
            fixes.append("MD040: Added language to code blocks")
            
        # Only write if changes were made
        if content != original_content and not self.dry_run:
            # Create backup
            backup_path = filepath.with_suffix('.md.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
                
            # Write fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return content != original_content, fixes
        
    def _fix_list_spacing(self, content: str) -> str:
        """Fix MD032: Lists should be surrounded by blank lines."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if current line is a list item
            if re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                # Check if previous non-empty line is not a list item or blank
                if i > 0 and fixed_lines and fixed_lines[-1].strip() and \
                   not re.match(r'^[\s]*[-*+]\s+', fixed_lines[-1]) and \
                   not re.match(r'^[\s]*\d+\.\s+', fixed_lines[-1]):
                    # Add blank line before list
                    fixed_lines.append('')
                    
                # Add the list item
                fixed_lines.append(line)
                
                # Check if next line is not a list item and not blank
                if i + 1 < len(lines) and lines[i + 1].strip() and \
                   not re.match(r'^[\s]*[-*+]\s+', lines[i + 1]) and \
                   not re.match(r'^[\s]*\d+\.\s+', lines[i + 1]) and \
                   not lines[i + 1].startswith('  '):  # Not a continuation
                    i += 1
                    # Add blank line after list
                    fixed_lines.append('')
                    fixed_lines.append(lines[i])
                else:
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
                
        return '\n'.join(fixed_lines)
        
    def _fix_code_block_spacing(self, content: str) -> str:
        """Fix MD031: Code blocks should be surrounded by blank lines."""
        lines = content.split('\n')
        fixed_lines = []
        in_code_block = False
        
        for i, line in enumerate(lines):
            if line.startswith('```'):
                if not in_code_block:
                    # Starting code block
                    if i > 0 and fixed_lines and fixed_lines[-1].strip():
                        fixed_lines.append('')
                    in_code_block = True
                else:
                    # Ending code block
                    in_code_block = False
                    fixed_lines.append(line)
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        fixed_lines.append('')
                    continue
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def _fix_heading_spacing(self, content: str) -> str:
        """Fix MD022: Headings should be surrounded by blank lines."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Skip frontmatter
            if i == 0 and line == '---':
                in_frontmatter = True
                fixed_lines.append(line)
                continue
            elif line == '---' and i > 0:
                in_frontmatter = False
                fixed_lines.append(line)
                continue
                
            if re.match(r'^#+\s+', line):
                # Add blank line before heading if needed
                if i > 0 and fixed_lines and fixed_lines[-1].strip() and \
                   not re.match(r'^#+\s+', fixed_lines[-1]):
                    fixed_lines.append('')
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def _fix_heading_punctuation(self, content: str) -> str:
        """Fix MD026: Remove trailing punctuation from headings."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if re.match(r'^#+\s+', line):
                # Remove trailing punctuation
                line = re.sub(r'[.,;:!?。，；：！？]+\s*$', '', line)
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def _fix_emphasis_as_heading(self, content: str) -> str:
        """Fix MD036: Convert emphasis used as headings to proper headings."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Pattern: **Bold text** alone on a line, followed by list or content
            if re.match(r'^\*\*[^*]+\*\*\s*$', line):
                # Check if next line is a list or content
                if i + 1 < len(lines) and (
                    re.match(r'^[-*+]\s+', lines[i + 1]) or
                    re.match(r'^\d+\.\s+', lines[i + 1])
                ):
                    # Convert to heading
                    heading_text = line.strip()[2:-2]  # Remove ** markers
                    fixed_lines.append(f"### {heading_text}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def _fix_line_length(self, content: str, max_length: int = 120) -> str:
        """Fix MD013: Line length violations with AI-aware wrapping."""
        lines = content.split('\n')
        fixed_lines = []
        in_code_block = False
        in_frontmatter = False
        
        for line in lines:
            # Handle frontmatter
            if line == '---':
                in_frontmatter = not in_frontmatter
                fixed_lines.append(line)
                continue
                
            if in_frontmatter:
                # Use YAML multiline for long summaries
                if line.startswith('summary:') and len(line) > max_length:
                    summary_text = line[8:].strip().strip('"')
                    fixed_lines.append('summary: |')
                    # Wrap summary text
                    words = summary_text.split()
                    current_line = '  '
                    for word in words:
                        if len(current_line) + len(word) + 1 > max_length - 2:
                            fixed_lines.append(current_line.rstrip())
                            current_line = '  ' + word
                        else:
                            if len(current_line) > 2:
                                current_line += ' '
                            current_line += word
                    if current_line.strip():
                        fixed_lines.append(current_line.rstrip())
                else:
                    fixed_lines.append(line)
                continue
                
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                fixed_lines.append(line)
                continue
                
            if in_code_block:
                fixed_lines.append(line)
                continue
                
            # Handle regular content
            if len(line) <= max_length:
                fixed_lines.append(line)
            else:
                # Smart wrapping for AI technical content
                if line.startswith('#'):  # Heading
                    fixed_lines.append(line)  # Don't wrap headings
                elif line.startswith('- ') or re.match(r'^\d+\.\s+', line):  # List item
                    # Wrap list items preserving indentation
                    prefix = re.match(r'^([-\d]+\.\s+)', line).group(1)
                    text = line[len(prefix):]
                    wrapped = self._wrap_text(text, max_length - len(prefix))
                    fixed_lines.append(prefix + wrapped[0])
                    for wrapped_line in wrapped[1:]:
                        fixed_lines.append('  ' + wrapped_line)
                else:
                    # Regular paragraph
                    wrapped = self._wrap_text(line, max_length)
                    fixed_lines.extend(wrapped)
                    
        return '\n'.join(fixed_lines)
        
    def _wrap_text(self, text: str, max_length: int) -> List[str]:
        """Intelligently wrap text preserving AI terminology."""
        # Preserve technical terms
        ai_terms = [
            'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'natural language processing', 'computer vision',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'transformer', 'attention mechanism', 'backpropagation'
        ]
        
        words = text.split()
        lines = []
        current_line = ''
        
        for word in words:
            # Check if adding word exceeds limit
            if len(current_line) + len(word) + 1 > max_length:
                if current_line:
                    lines.append(current_line.rstrip())
                current_line = word
            else:
                if current_line:
                    current_line += ' '
                current_line += word
                
        if current_line:
            lines.append(current_line.rstrip())
            
        return lines if lines else [text]
        
    def _fix_code_language(self, content: str) -> str:
        """Fix MD040: Add language specification to code blocks."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            if line == '```':
                # Code block without language
                # Try to infer language from content
                language = 'text'  # Default
                
                # Look ahead to guess language
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if 'import' in next_line or 'def ' in next_line:
                        language = 'python'
                    elif 'function' in next_line or 'const ' in next_line:
                        language = 'javascript'
                    elif 'interface' in next_line or ': ' in next_line:
                        language = 'typescript'
                        
                fixed_lines.append(f'```{language}')
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def fix_directory(self, directory: Path) -> Dict[str, any]:
        """Fix all markdown files in a directory."""
        results = {
            'files_processed': 0,
            'files_fixed': 0,
            'total_fixes': 0,
            'fixes_by_type': {},
            'files': []
        }
        
        md_files = list(directory.rglob('*.md'))
        
        for filepath in md_files:
            changed, fixes = self.fix_file(filepath)
            results['files_processed'] += 1
            
            if changed:
                results['files_fixed'] += 1
                results['total_fixes'] += len(fixes)
                
                for fix in fixes:
                    fix_type = fix.split(':')[0]
                    results['fixes_by_type'][fix_type] = \
                        results['fixes_by_type'].get(fix_type, 0) + 1
                        
                results['files'].append({
                    'path': str(filepath),
                    'fixes': fixes
                })
                
                if self.verbose:
                    print(f"✓ Fixed {filepath.name}: {', '.join(fixes)}")
                    
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Fix markdown quality issues in AI Knowledge website'
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to markdown file or directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    fixer = MarkdownQualityFixer(dry_run=args.dry_run, verbose=args.verbose)
    
    if args.path.is_file():
        changed, fixes = fixer.fix_file(args.path)
        if args.json:
            print(json.dumps({
                'file': str(args.path),
                'changed': changed,
                'fixes': fixes
            }, indent=2))
        else:
            if changed:
                print(f"✓ Fixed {args.path.name}")
                for fix in fixes:
                    print(f"  - {fix}")
            else:
                print(f"✓ {args.path.name} is already compliant")
    else:
        results = fixer.fix_directory(args.path)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n📊 Markdown Quality Fix Report")
            print(f"{'=' * 50}")
            print(f"Files processed: {results['files_processed']}")
            print(f"Files fixed: {results['files_fixed']}")
            print(f"Total fixes: {results['total_fixes']}")
            
            if results['fixes_by_type']:
                print(f"\n📈 Fixes by type:")
                for fix_type, count in sorted(results['fixes_by_type'].items()):
                    print(f"  {fix_type}: {count}")
                    
            if results['files'] and not args.dry_run:
                print(f"\n💾 Backup files created with .bak extension")
                
            mode = "DRY RUN" if args.dry_run else "APPLIED"
            print(f"\n✅ Quality fixes {mode} successfully!")


if __name__ == '__main__':
    main()