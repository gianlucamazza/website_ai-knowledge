#!/usr/bin/env python3
"""
Pre-commit Hook for Markdown Quality Control
Lightweight, fast auto-fix capabilities for common violations
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Configure logging for pre-commit context
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class FastMarkdownFixer:
    """Fast, targeted fixes for common markdown issues in pre-commit context"""
    
    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.fixes_applied = 0
        self.violations_found = 0
        
    def check_and_fix_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check and optionally fix a markdown file"""
        violations = []
        content_changed = False
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Fast checks with optional auto-fix
            
            # 1. Check MD047: Final newline
            if not content.endswith('\n'):
                violations.append(f"MD047: File missing final newline")
                if self.auto_fix:
                    content = content + '\n'
                    content_changed = True
                    
            elif content.endswith('\n\n\n'):
                violations.append(f"MD047: File has extra final newlines")
                if self.auto_fix:
                    content = content.rstrip('\n') + '\n'
                    content_changed = True
                    
            # 2. Check MD009: Trailing whitespace (quick check)
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if line.endswith(' ') and not line.endswith('  '):  # Allow intentional line breaks
                    violations.append(f"MD009: Trailing spaces on line {i}")
                    if self.auto_fix:
                        lines[i-1] = line.rstrip()
                        content_changed = True
                        
            # 3. Check MD026: Heading punctuation
            for i, line in enumerate(lines, 1):
                if re.match(r'^#{1,6}\s+.*[.,:;!?]$', line):
                    violations.append(f"MD026: Heading has trailing punctuation on line {i}")
                    if self.auto_fix:
                        lines[i-1] = re.sub(r'[.,:;!?]+$', '', line)
                        content_changed = True
                        
            # 4. Check MD025: Multiple H1s (quick scan)
            h1_count = sum(1 for line in lines if re.match(r'^#\s+', line))
            if h1_count > 1:
                violations.append(f"MD025: Multiple H1 headings found ({h1_count})")
                # Auto-fix would be more complex, skip for pre-commit
                
            # 5. Check MD013: Line length (informational only)
            long_lines = [(i, len(line)) for i, line in enumerate(lines, 1) 
                         if len(line) > 120 
                         and not line.strip().startswith('http')
                         and not line.strip().startswith('source_url:')]
            if long_lines:
                count = len(long_lines)
                violations.append(f"MD013: {count} lines exceed 120 characters (auto-fix available)")
                
            # Apply fixes if content changed
            if self.auto_fix and content_changed:
                if lines != content.split('\n'):
                    content = '\n'.join(lines)
                file_path.write_text(content, encoding='utf-8')
                self.fixes_applied += len([v for v in violations if 'auto-fix available' not in v])
                
            self.violations_found += len(violations)
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False, [f"Error: {str(e)}"]
            
    def process_files(self, file_paths: List[str]) -> int:
        """Process multiple files and return exit code"""
        total_violations = 0
        problem_files = []
        
        logger.info(f"Checking {len(file_paths)} markdown files...")
        if self.auto_fix:
            logger.info("Auto-fix enabled for safe violations")
            
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            is_clean, violations = self.check_and_fix_file(file_path)
            
            if not is_clean:
                total_violations += len(violations)
                problem_files.append((file_path, violations))
                
        # Report results
        if problem_files:
            logger.info(f"Found {total_violations} violations in {len(problem_files)} files")
            
            # Show details for first few problematic files
            for file_path, violations in problem_files[:5]:
                logger.info(f"\n{file_path}:")
                for violation in violations[:3]:  # Show first 3 violations per file
                    logger.info(f"  - {violation}")
                if len(violations) > 3:
                    logger.info(f"  ... and {len(violations) - 3} more")
                    
            if len(problem_files) > 5:
                logger.info(f"\n... and {len(problem_files) - 5} more files with violations")
                
            if self.fixes_applied > 0:
                logger.info(f"\nAuto-fixed {self.fixes_applied} violations")
                logger.info("Re-run to check remaining issues")
                
            # Provide guidance
            logger.info("\nTo fix remaining violations:")
            logger.info("1. Run: python scripts/migrate_current_violations.py apps/site/src/content --apply")
            logger.info("2. Or run: python scripts/markdown_quality_fixer.py apps/site/src/content")
            
            return 1  # Fail pre-commit if violations remain
        else:
            logger.info("✅ All markdown files pass quality checks")
            if self.fixes_applied > 0:
                logger.info(f"Auto-fixed {self.fixes_applied} violations")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Pre-commit markdown quality hook")
    parser.add_argument(
        'files', 
        nargs='*', 
        help='Markdown files to check'
    )
    parser.add_argument(
        '--auto-fix', 
        action='store_true',
        help='Automatically fix safe violations'
    )
    
    args = parser.parse_args()
    
    if not args.files:
        logger.info("No markdown files to check")
        return 0
        
    # Filter to only .md files
    md_files = [f for f in args.files if f.endswith('.md')]
    
    if not md_files:
        logger.info("No markdown files to check")
        return 0
        
    # Process files
    fixer = FastMarkdownFixer(auto_fix=args.auto_fix)
    return fixer.process_files(md_files)


if __name__ == '__main__':
    sys.exit(main())