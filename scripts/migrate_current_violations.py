#!/usr/bin/env python3
"""
Migration Script for Current Markdown Violations
Fixes all 25 glossary entries with targeted precision

Based on analysis:
- MD047: Files missing final newlines (25 files)
- MD013: Line length violations (20+ files)  
- MD025: Multiple H1 headings (1 file)
- MD026: Heading punctuation (5 files)
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from dataclasses import dataclass

try:
    import frontmatter
except ImportError:
    print("Installing required dependency: python-frontmatter")
    os.system("pip install python-frontmatter")
    import frontmatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/markdown_migration.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ViolationFix:
    """Track a specific violation fix"""
    file_path: str
    rule: str
    line_number: int
    description: str
    fix_applied: bool = False

class EnterpriseMarkdownMigrator:
    """Enterprise-grade migrator for current markdown violations"""
    
    def __init__(self, content_dir: str, dry_run: bool = False, backup: bool = True):
        self.content_dir = Path(content_dir)
        self.dry_run = dry_run
        self.backup = backup
        self.fixes_applied: Dict[str, List[ViolationFix]] = {}
        self.critical_failures: List[str] = []
        
        # AI-specific terms that should not be broken across lines
        self.ai_terms = {
            'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'natural language processing', 'computer vision',
            'reinforcement learning', 'generative adversarial network',
            'transformer architecture', 'attention mechanism', 'gradient descent',
            'backpropagation algorithm', 'convolutional neural network'
        }
        
    def backup_file(self, file_path: Path) -> bool:
        """Create timestamped backup of file"""
        if not self.backup or self.dry_run:
            return True
            
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f'.{timestamp}.backup')
            content = file_path.read_text(encoding='utf-8')
            backup_path.write_text(content, encoding='utf-8')
            logger.debug(f"Created backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")
            return False
            
    def analyze_file_violations(self, file_path: Path) -> List[ViolationFix]:
        """Analyze a file and identify specific violations to fix"""
        violations = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check MD047: Files should end with single newline
            if not content.endswith('\n'):
                violations.append(ViolationFix(
                    file_path=str(file_path),
                    rule='MD047',
                    line_number=len(lines),
                    description='File missing final newline'
                ))
            elif content.endswith('\n\n'):
                violations.append(ViolationFix(
                    file_path=str(file_path),
                    rule='MD047',
                    line_number=len(lines),
                    description='File has extra final newlines'
                ))
                
            # Check MD013: Line length violations
            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    # Skip certain types of lines
                    if (line.strip().startswith('http') or 
                        line.strip().startswith('source_url:') or
                        '](/apps/site/src/content/glossary/' in line):
                        continue
                        
                    violations.append(ViolationFix(
                        file_path=str(file_path),
                        rule='MD013',
                        line_number=i,
                        description=f'Line length {len(line)} > 120 characters'
                    ))
                    
            # Check MD025: Multiple H1 headings
            h1_count = 0
            for i, line in enumerate(lines, 1):
                if re.match(r'^#\s+', line):
                    h1_count += 1
                    if h1_count > 1:
                        violations.append(ViolationFix(
                            file_path=str(file_path),
                            rule='MD025',
                            line_number=i,
                            description='Multiple H1 headings found'
                        ))
                        
            # Check MD026: Heading punctuation
            for i, line in enumerate(lines, 1):
                if re.match(r'^#{1,6}\s+.*[.,:;!?]$', line):
                    violations.append(ViolationFix(
                        file_path=str(file_path),
                        rule='MD026',
                        line_number=i,
                        description='Heading has trailing punctuation'
                    ))
                    
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            self.critical_failures.append(str(file_path))
            
        return violations
        
    def fix_md047_final_newline(self, content: str) -> str:
        """Fix MD047: Ensure single final newline"""
        if not content.endswith('\n'):
            content = content + '\n'
            logger.debug("Added missing final newline")
        elif content.endswith('\n\n'):
            # Remove extra newlines but keep one
            content = content.rstrip('\n') + '\n'
            logger.debug("Removed extra final newlines")
        return content
        
    def fix_md013_line_length(self, content: str) -> str:
        """Fix MD013: Line length with AI-aware wrapping"""
        lines = content.split('\n')
        fixed_lines = []
        in_frontmatter = False
        frontmatter_count = 0
        
        for i, line in enumerate(lines):
            # Track frontmatter boundaries
            if line.strip() == '---':
                frontmatter_count += 1
                in_frontmatter = frontmatter_count < 2
                fixed_lines.append(line)
                continue
                
            if in_frontmatter:
                fixed_lines.append(line)
                continue
                
            # Skip certain line types
            if (len(line) <= 120 or
                line.strip().startswith('http') or
                line.strip().startswith('source_url:') or
                line.strip().startswith('|') or  # Table rows
                re.match(r'^\s*```', line) or   # Code fences
                '](/apps/site/src/content/glossary/' in line):  # Internal links
                fixed_lines.append(line)
                continue
                
            # Smart wrapping for long lines
            if len(line) > 120:
                wrapped = self._wrap_line_intelligently(line)
                fixed_lines.extend(wrapped)
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def _wrap_line_intelligently(self, line: str) -> List[str]:
        """Wrap a line while preserving AI terminology and context"""
        # Preserve leading whitespace
        leading_space = len(line) - len(line.lstrip())
        indent = line[:leading_space]
        content = line[leading_space:]
        
        # For list items, preserve list structure
        list_match = re.match(r'^([-*+]|\d+\.)\s+', content)
        if list_match:
            list_marker = list_match.group(0)
            text_content = content[len(list_marker):]
            
            wrapped_lines = []
            current_line = indent + list_marker
            
            # Split respecting AI terms
            words = self._split_respecting_ai_terms(text_content)
            
            for word in words:
                test_line = current_line + (' ' if current_line != indent + list_marker else '') + word
                if len(test_line) <= 120:
                    current_line = test_line
                else:
                    if current_line.strip() != list_marker.strip():
                        wrapped_lines.append(current_line)
                    current_line = indent + '  ' + word  # Continue with proper indentation
                    
            if current_line.strip():
                wrapped_lines.append(current_line)
                
            return wrapped_lines if wrapped_lines else [line]
        
        # For regular paragraphs
        wrapped_lines = []
        words = self._split_respecting_ai_terms(content)
        current_line = indent
        
        for word in words:
            test_line = current_line + (' ' if current_line != indent else '') + word
            if len(test_line) <= 120:
                current_line = test_line
            else:
                if current_line.strip():
                    wrapped_lines.append(current_line)
                current_line = indent + word
                
        if current_line.strip():
            wrapped_lines.append(current_line)
            
        return wrapped_lines if wrapped_lines else [line]
        
    def _split_respecting_ai_terms(self, text: str) -> List[str]:
        """Split text while keeping AI terms together"""
        # Simple word splitting for now
        # In a full implementation, this would use NLP to identify technical terms
        words = text.split()
        
        # Basic protection for common AI terms
        result = []
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for multi-word AI terms
            term_found = False
            for term in self.ai_terms:
                term_words = term.split()
                if (i + len(term_words) <= len(words) and 
                    ' '.join(words[i:i+len(term_words)]).lower() == term):
                    result.append(' '.join(words[i:i+len(term_words)]))
                    i += len(term_words)
                    term_found = True
                    break
                    
            if not term_found:
                result.append(word)
                i += 1
                
        return result
        
    def fix_md025_multiple_h1(self, content: str) -> str:
        """Fix MD025: Convert additional H1s to H2s"""
        lines = content.split('\n')
        fixed_lines = []
        h1_count = 0
        in_frontmatter = False
        frontmatter_count = 0
        
        for line in lines:
            # Track frontmatter
            if line.strip() == '---':
                frontmatter_count += 1
                in_frontmatter = frontmatter_count < 2
                fixed_lines.append(line)
                continue
                
            if in_frontmatter:
                fixed_lines.append(line)
                continue
                
            # Check for H1 headings
            if re.match(r'^#\s+', line):
                h1_count += 1
                if h1_count > 1:
                    # Convert to H2
                    new_line = '#' + line
                    fixed_lines.append(new_line)
                    logger.debug(f"Converted H1 to H2: {line.strip()}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def fix_md026_heading_punctuation(self, content: str) -> str:
        """Fix MD026: Remove trailing punctuation from headings"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if re.match(r'^#{1,6}\s+.*[.,:;!?]$', line):
                # Remove trailing punctuation
                new_line = re.sub(r'[.,:;!?]+$', '', line)
                fixed_lines.append(new_line)
                logger.debug(f"Removed punctuation from heading: {line.strip()}")
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def fix_file_violations(self, file_path: Path, violations: List[ViolationFix]) -> bool:
        """Apply fixes for all violations in a file"""
        if not violations:
            return False
            
        logger.info(f"Fixing {len(violations)} violations in {file_path}")
        
        try:
            # Create backup
            if not self.backup_file(file_path):
                logger.error(f"Failed to backup {file_path}, skipping fixes")
                return False
                
            # Read content
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Apply fixes in order of safety (least likely to break content first)
            
            # 1. Fix final newlines (safest)
            if any(v.rule == 'MD047' for v in violations):
                content = self.fix_md047_final_newline(content)
                
            # 2. Fix heading punctuation (safe)
            if any(v.rule == 'MD026' for v in violations):
                content = self.fix_md026_heading_punctuation(content)
                
            # 3. Fix multiple H1s (safe)
            if any(v.rule == 'MD025' for v in violations):
                content = self.fix_md025_multiple_h1(content)
                
            # 4. Fix line length (most complex, do last)
            if any(v.rule == 'MD013' for v in violations):
                content = self.fix_md013_line_length(content)
                
            # Write changes
            if not self.dry_run and content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
            # Mark violations as fixed
            for violation in violations:
                violation.fix_applied = True
                if violation.rule not in self.fixes_applied:
                    self.fixes_applied[violation.rule] = []
                self.fixes_applied[violation.rule].append(violation)
                
            if content != original_content:
                action = "Would fix" if self.dry_run else "Fixed"
                logger.info(f"{action} {len(violations)} violations in {file_path}")
                return True
            
        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")
            self.critical_failures.append(str(file_path))
            
        return False
        
    def generate_migration_report(self) -> str:
        """Generate comprehensive migration report"""
        report_lines = [
            "# Markdown Quality Migration Report",
            "",
            f"**Migration Mode**: {'DRY RUN' if self.dry_run else 'LIVE'}",
            f"**Timestamp**: {__import__('datetime').datetime.now().isoformat()}",
            f"**Content Directory**: {self.content_dir}",
            ""
        ]
        
        # Summary statistics
        total_files = sum(len(fixes) for fixes in self.fixes_applied.values())
        unique_files = len(set(fix.file_path for fixes in self.fixes_applied.values() for fix in fixes))
        
        report_lines.extend([
            "## Summary",
            "",
            f"- **Files analyzed**: {unique_files}",
            f"- **Total violations fixed**: {sum(len(fixes) for fixes in self.fixes_applied.values())}",
            f"- **Critical failures**: {len(self.critical_failures)}",
            ""
        ])
        
        # Fixes by rule
        if self.fixes_applied:
            report_lines.extend([
                "## Fixes Applied by Rule",
                ""
            ])
            
            for rule in sorted(self.fixes_applied.keys()):
                fixes = self.fixes_applied[rule]
                rule_descriptions = {
                    'MD047': 'Final newline issues',
                    'MD013': 'Line length violations', 
                    'MD025': 'Multiple H1 headings',
                    'MD026': 'Heading punctuation'
                }
                
                report_lines.extend([
                    f"### {rule} - {rule_descriptions.get(rule, 'Unknown rule')}",
                    "",
                    f"**Fixes applied**: {len(fixes)}",
                    ""
                ])
                
                # Group by file
                files = {}
                for fix in fixes:
                    if fix.file_path not in files:
                        files[fix.file_path] = []
                    files[fix.file_path].append(fix)
                    
                for file_path, file_fixes in files.items():
                    report_lines.append(f"- `{Path(file_path).name}`: {len(file_fixes)} fixes")
                    
                report_lines.append("")
                
        # Critical failures
        if self.critical_failures:
            report_lines.extend([
                "## Critical Failures",
                "",
                "The following files could not be processed:",
                ""
            ])
            
            for failure in self.critical_failures:
                report_lines.append(f"- `{failure}`")
                
            report_lines.append("")
            
        # Next steps
        report_lines.extend([
            "## Next Steps",
            "",
            "1. **Validation**: Run `npm run lint` to verify fixes",
            "2. **Testing**: Test site build with `npm run build`", 
            "3. **Review**: Check git diff for any unexpected changes",
            "4. **Commit**: Commit fixes with descriptive message",
            ""
        ])
        
        if self.dry_run:
            report_lines.extend([
                "**Note**: This was a DRY RUN - no files were modified.",
                "Re-run with `--apply` to make changes.",
                ""
            ])
            
        return '\n'.join(report_lines)
        
    def migrate(self) -> Dict[str, any]:
        """Run the complete migration process"""
        logger.info(f"Starting markdown migration {'(DRY RUN)' if self.dry_run else ''}")
        logger.info(f"Content directory: {self.content_dir}")
        
        if not self.content_dir.exists():
            logger.error(f"Content directory does not exist: {self.content_dir}")
            return {'success': False, 'error': 'Directory not found'}
            
        # Find all markdown files
        md_files = list(self.content_dir.rglob('*.md'))
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Analyze and fix each file
        files_processed = 0
        files_with_fixes = 0
        
        for md_file in md_files:
            files_processed += 1
            logger.debug(f"Processing {md_file}")
            
            # Analyze violations
            violations = self.analyze_file_violations(md_file)
            
            if violations:
                # Apply fixes
                if self.fix_file_violations(md_file, violations):
                    files_with_fixes += 1
                    
        # Generate report
        report = self.generate_migration_report()
        
        # Save report
        report_path = Path('/tmp/markdown_migration_report.md')
        report_path.write_text(report, encoding='utf-8')
        
        logger.info(f"Migration complete!")
        logger.info(f"- Files processed: {files_processed}")
        logger.info(f"- Files with fixes: {files_with_fixes}")
        logger.info(f"- Report saved: {report_path}")
        
        if self.critical_failures:
            logger.warning(f"Critical failures: {len(self.critical_failures)}")
            
        return {
            'success': True,
            'files_processed': files_processed,
            'files_with_fixes': files_with_fixes,
            'fixes_applied': self.fixes_applied,
            'critical_failures': self.critical_failures,
            'report_path': str(report_path)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate current markdown violations in AI Knowledge Website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze violations (dry run)
  python migrate_current_violations.py apps/site/src/content --dry-run
  
  # Apply fixes
  python migrate_current_violations.py apps/site/src/content --apply
  
  # Apply fixes without backups (not recommended)  
  python migrate_current_violations.py apps/site/src/content --apply --no-backup
        """
    )
    
    parser.add_argument(
        'content_dir',
        help='Directory containing markdown files to migrate'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        default=True,
        help='Analyze violations without applying fixes (default)'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true', 
        help='Apply fixes to files (overrides --dry-run)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files (not recommended)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Determine if this is a dry run
    dry_run = not args.apply
    
    if dry_run:
        logger.info("Running in DRY RUN mode - no files will be modified")
        logger.info("Use --apply to make actual changes")
    else:
        logger.info("Running in APPLY mode - files will be modified")
        
    # Initialize migrator
    migrator = EnterpriseMarkdownMigrator(
        content_dir=args.content_dir,
        dry_run=dry_run,
        backup=not args.no_backup
    )
    
    try:
        # Run migration
        result = migrator.migrate()
        
        if result['success']:
            logger.info("Migration completed successfully")
            
            if result['report_path']:
                print(f"\nDetailed report saved to: {result['report_path']}")
                
            sys.exit(0)
        else:
            logger.error(f"Migration failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()