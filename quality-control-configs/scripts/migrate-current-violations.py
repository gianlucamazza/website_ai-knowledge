#!/usr/bin/env python3
"""
Current Violations Migration Script
Specifically addresses the 8 types of violations found in your current codebase

Usage:
    python migrate-current-violations.py [--dry-run] [--file=FILE]
    
This script addresses:
- MD025: Multiple top-level headings
- MD047: Files should end with single newline
- MD013: Line length violations  
- MD036: Emphasis used instead of heading
- MD032: Lists should be surrounded by blank lines
- MD031: Fenced code blocks surrounded by blank lines
- MD040: Fenced code blocks should have language
"""

import argparse
import re
import os
from pathlib import Path
import frontmatter
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CurrentViolationsFixer:
    """Fixes the specific violations found in your current codebase"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fixes_applied = {
            'MD025': 0,  # Multiple top-level headings
            'MD047': 0,  # Missing final newline
            'MD013': 0,  # Line too long
            'MD036': 0,  # Emphasis as heading
            'MD032': 0,  # Lists without blank lines
            'MD031': 0,  # Code blocks without blank lines  
            'MD040': 0,  # Code blocks without language
        }
    
    def fix_file(self, file_path: str) -> bool:
        """Fix all violations in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Parse frontmatter and content
            post = frontmatter.loads(original_content)
            content_lines = post.content.split('\n')
            
            # Apply fixes in order
            fixed_lines = self._fix_multiple_h1_headings(content_lines, file_path)
            fixed_lines = self._fix_emphasis_as_headings(fixed_lines)
            fixed_lines = self._fix_list_spacing(fixed_lines)
            fixed_lines = self._fix_code_block_spacing(fixed_lines)
            fixed_lines = self._fix_code_block_language(fixed_lines)
            fixed_lines = self._fix_long_lines(fixed_lines, file_path)
            
            # Reconstruct content
            fixed_content = '\n'.join(fixed_lines)
            
            # Fix final newline (MD047)
            if not fixed_content.endswith('\n'):
                fixed_content += '\n'
                self.fixes_applied['MD047'] += 1
                logger.info(f"  ✅ MD047: Added final newline")
            elif fixed_content.endswith('\n\n'):
                fixed_content = fixed_content.rstrip('\n') + '\n'
                self.fixes_applied['MD047'] += 1
                logger.info(f"  ✅ MD047: Fixed multiple trailing newlines")
            
            # Update post content
            post.content = fixed_content
            
            # Write back if changes were made
            new_full_content = frontmatter.dumps(post)
            
            if new_full_content != original_content:
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_full_content)
                logger.info(f"✅ Fixed {file_path}")
                return True
            else:
                logger.info(f"ℹ️  No changes needed for {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False
    
    def _fix_multiple_h1_headings(self, lines: List[str], file_path: str) -> List[str]:
        """Fix MD025: Multiple top-level headings in same document"""
        h1_count = 0
        fixed_lines = []
        
        for line in lines:
            if line.strip().startswith('# ') and not line.strip().startswith('## '):
                h1_count += 1
                if h1_count > 1:
                    # Convert additional H1s to H2s
                    fixed_line = line.replace('# ', '## ', 1)
                    fixed_lines.append(fixed_line)
                    self.fixes_applied['MD025'] += 1
                    logger.info(f"  ✅ MD025: Converted H1 to H2: '{line.strip()[:50]}...'")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return fixed_lines
    
    def _fix_emphasis_as_headings(self, lines: List[str]) -> List[str]:
        """Fix MD036: Emphasis used instead of a heading"""
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if line is standalone emphasis that should be heading
            if (re.match(r'^\*\*([^*]+)\*\*\s*$', stripped) and 
                self._looks_like_heading(stripped, lines, i)):
                
                # Extract text and convert to heading
                match = re.match(r'^\*\*([^*]+)\*\*\s*$', stripped)
                if match:
                    heading_text = match.group(1)
                    # Use ### for section headings (adjust level as needed)
                    indent = len(line) - len(line.lstrip())
                    fixed_line = ' ' * indent + f'### {heading_text}'
                    fixed_lines.append(fixed_line)
                    self.fixes_applied['MD036'] += 1
                    logger.info(f"  ✅ MD036: Converted emphasis to heading: '{heading_text[:30]}...'")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return fixed_lines
    
    def _looks_like_heading(self, line: str, all_lines: List[str], line_index: int) -> bool:
        """Determine if emphasized text should be a heading based on context"""
        # Check if followed by a list or paragraph
        next_line_idx = line_index + 1
        if next_line_idx < len(all_lines):
            next_line = all_lines[next_line_idx].strip()
            # If followed by a list or content, likely a heading
            if next_line.startswith('-') or next_line.startswith('*') or len(next_line) > 20:
                return True
        
        # Check if it's a single word or short phrase (typical heading)
        content = re.sub(r'\*\*([^*]+)\*\*', r'\\1', line).strip()
        if len(content.split()) <= 4 and len(content) < 50:
            return True
        
        return False
    
    def _fix_list_spacing(self, lines: List[str]) -> List[str]:
        """Fix MD032: Lists should be surrounded by blank lines"""
        fixed_lines = []
        
        for i, line in enumerate(lines):
            current_is_list = line.strip().startswith(('-', '*', '+')) or re.match(r'^\\s*\\d+\\.', line.strip())
            prev_line = lines[i-1] if i > 0 else ''
            next_line = lines[i+1] if i < len(lines) - 1 else ''
            
            prev_is_list = prev_line.strip().startswith(('-', '*', '+')) or re.match(r'^\\s*\\d+\\.', prev_line.strip())
            next_is_list = next_line.strip().startswith(('-', '*', '+')) or re.match(r'^\\s*\\d+\\.', next_line.strip())
            
            # Add blank line before list starts
            if (current_is_list and not prev_is_list and 
                prev_line.strip() != '' and not prev_line.strip().startswith('#')):
                if i > 0 and lines[i-1].strip() != '':
                    fixed_lines.append('')
                    self.fixes_applied['MD032'] += 1
                    logger.info(f"  ✅ MD032: Added blank line before list")
            
            fixed_lines.append(line)
            
            # Add blank line after list ends
            if (current_is_list and not next_is_list and 
                next_line.strip() != '' and not next_line.strip().startswith('#')):
                if i < len(lines) - 1 and next_line.strip() != '':
                    fixed_lines.append('')
                    self.fixes_applied['MD032'] += 1
                    logger.info(f"  ✅ MD032: Added blank line after list")
        
        return fixed_lines
    
    def _fix_code_block_spacing(self, lines: List[str]) -> List[str]:
        """Fix MD031: Fenced code blocks should be surrounded by blank lines"""
        fixed_lines = []
        
        for i, line in enumerate(lines):
            is_code_fence = line.strip().startswith('```')
            prev_line = lines[i-1].strip() if i > 0 else ''
            next_line = lines[i+1].strip() if i < len(lines) - 1 else ''
            
            # Add blank line before opening fence
            if (is_code_fence and prev_line != '' and not prev_line.startswith('#')):
                if i > 0 and lines[i-1].strip() != '':
                    fixed_lines.append('')
                    self.fixes_applied['MD031'] += 1
                    logger.info(f"  ✅ MD031: Added blank line before code block")
            
            fixed_lines.append(line)
            
            # Add blank line after closing fence
            if (is_code_fence and line.strip() == '```' and 
                next_line != '' and not next_line.startswith('#')):
                if i < len(lines) - 1 and next_line != '':
                    fixed_lines.append('')
                    self.fixes_applied['MD031'] += 1
                    logger.info(f"  ✅ MD031: Added blank line after code block")
        
        return fixed_lines
    
    def _fix_code_block_language(self, lines: List[str]) -> List[str]:
        """Fix MD040: Fenced code blocks should have a language specified"""
        fixed_lines = []
        
        for line in lines:
            if line.strip() == '```':
                # Try to infer language from context or default to common one
                fixed_line = line.replace('```', '```python')  # Default for AI content
                fixed_lines.append(fixed_line)
                self.fixes_applied['MD040'] += 1
                logger.info(f"  ✅ MD040: Added language to code block")
            else:
                fixed_lines.append(line)
        
        return fixed_lines
    
    def _fix_long_lines(self, lines: List[str], file_path: str) -> List[str]:
        """Fix MD013: Line length violations (smart wrapping)"""
        fixed_lines = []
        max_length = 120
        
        for line in lines:
            if len(line) > max_length and not line.strip().startswith('#'):
                # Smart line wrapping for non-heading lines
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    # List item - don't wrap
                    fixed_lines.append(line)
                elif 'http' in line and ('https://' in line or 'http://' in line):
                    # Contains URL - don't wrap
                    fixed_lines.append(line)
                else:
                    # Try to wrap at sentence boundaries
                    wrapped_lines = self._smart_wrap_line(line, max_length)
                    if len(wrapped_lines) > 1:
                        fixed_lines.extend(wrapped_lines)
                        self.fixes_applied['MD013'] += 1
                        logger.info(f"  ✅ MD013: Wrapped long line ({len(line)} chars)")
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return fixed_lines
    
    def _smart_wrap_line(self, line: str, max_length: int) -> List[str]:
        """Smart line wrapping at sentence/phrase boundaries"""
        if len(line) <= max_length:
            return [line]
        
        # Preserve leading whitespace
        leading_space = len(line) - len(line.lstrip())
        indent = line[:leading_space]
        content = line[leading_space:]
        
        # Split at sentence boundaries
        sentences = re.split(r'(\\. |\\. )', content)
        wrapped_lines = []
        current_line = indent
        
        for part in sentences:
            if len(current_line + part) <= max_length:
                current_line += part
            else:
                if current_line.strip():
                    wrapped_lines.append(current_line.rstrip())
                current_line = indent + part
        
        if current_line.strip():
            wrapped_lines.append(current_line.rstrip())
        
        return wrapped_lines if len(wrapped_lines) > 1 else [line]
    
    def print_summary(self):
        """Print summary of fixes applied"""
        total_fixes = sum(self.fixes_applied.values())
        
        if total_fixes == 0:
            logger.info("✨ No fixes needed - all files are compliant!")
            return
        
        logger.info(f"\\n📊 Migration Summary:")
        logger.info(f"Total fixes applied: {total_fixes}")
        logger.info("\\nFixes by rule:")
        
        for rule, count in self.fixes_applied.items():
            if count > 0:
                descriptions = {
                    'MD025': 'Multiple top-level headings',
                    'MD047': 'Missing final newline', 
                    'MD013': 'Line too long',
                    'MD036': 'Emphasis used as heading',
                    'MD032': 'Lists without blank lines',
                    'MD031': 'Code blocks without blank lines',
                    'MD040': 'Code blocks without language'
                }
                logger.info(f"  • {rule}: {count} fixes - {descriptions[rule]}")


def find_markdown_files(paths: List[str]) -> List[str]:
    """Find all markdown files in given paths"""
    markdown_files = []
    
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.md':
            markdown_files.append(str(path))
        elif path.is_dir():
            markdown_files.extend([str(f) for f in path.rglob('*.md')])
    
    return markdown_files


def main():
    parser = argparse.ArgumentParser(description='Fix current markdown violations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    parser.add_argument('--file', help='Fix specific file instead of all content')
    parser.add_argument('paths', nargs='*', default=['apps/site/src/content'], help='Paths to process')
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = CurrentViolationsFixer(dry_run=args.dry_run)
    
    # Get files to process
    if args.file:
        files_to_process = [args.file]
    else:
        files_to_process = find_markdown_files(args.paths)
    
    if not files_to_process:
        logger.error("No markdown files found to process")
        return 1
    
    logger.info(f"🚀 Processing {len(files_to_process)} markdown files...")
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
    
    # Process files
    files_changed = 0
    for file_path in files_to_process:
        logger.info(f"\\nProcessing: {file_path}")
        if fixer.fix_file(file_path):
            files_changed += 1
    
    # Print summary
    fixer.print_summary()
    
    if args.dry_run:
        logger.info(f"\\n🔍 Would change {files_changed} files")
        logger.info("Run without --dry-run to apply fixes")
    else:
        logger.info(f"\\n✅ Successfully processed {files_changed} files")
        
        if files_changed > 0:
            logger.info("\\n📝 Next steps:")
            logger.info("1. Review the changes with 'git diff'")
            logger.info("2. Run quality check: 'npm run lint'")
            logger.info("3. Commit changes: 'git add . && git commit -m \"Fix markdown quality violations\"'")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())