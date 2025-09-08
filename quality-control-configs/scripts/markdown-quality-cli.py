#!/usr/bin/env python3
"""
Markdown Quality Control CLI
Enterprise-grade markdown quality control for AI Knowledge Website

Usage:
    python markdown-quality-cli.py check [--fix] [--config=CONFIG] [PATH]
    python markdown-quality-cli.py migrate [--dry-run] [--batch-size=N]
    python markdown-quality-cli.py report [--format=json|html|md] [--output=FILE]
    python markdown-quality-cli.py --help

Commands:
    check       Run quality checks on markdown files
    migrate     Migrate existing content to meet quality standards
    report      Generate quality assessment report
    
Options:
    --fix               Auto-fix issues where possible
    --config=CONFIG     Custom configuration file
    --dry-run           Show what would be changed without modifying files
    --batch-size=N      Process N files at a time for migration [default: 10]
    --format=FORMAT     Report format: json, html, or markdown [default: md]
    --output=FILE       Output file for report
    -v, --verbose       Verbose output
    -h, --help          Show this help message
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import frontmatter
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """Represents a markdown quality issue"""
    file_path: str
    rule: str
    line: Optional[int]
    column: Optional[int] 
    severity: str  # error, warning, info
    message: str
    fixable: bool = False

@dataclass
class QualityReport:
    """Quality assessment report"""
    timestamp: str
    total_files: int
    files_with_issues: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_rule: Dict[str, int]
    fixable_issues: int
    files_processed: List[str]
    issues: List[QualityIssue]

class MarkdownQualityController:
    """Main quality control orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ".markdownlint-cli2.yaml"
        self.config = self._load_config()
        self.auto_fix_rules = {
            'MD022', 'MD032', 'MD047', 'MD012', 'MD009'  # Safe auto-fix rules
        }
        
    def _load_config(self) -> dict:
        """Load markdown quality configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration for quality control"""
        return {
            'globs': ['apps/site/src/content/**/*.md'],
            'config': {
                'default': True,
                'MD013': {'line_length': 120},
                'MD025': {'front_matter_title': '^\\s*title\\s*[:=]'},
                'MD041': {'front_matter_title': '^\\s*title\\s*[:=]'}
            }
        }
    
    def check_files(self, paths: List[str], auto_fix: bool = False) -> QualityReport:
        """Run quality checks on markdown files"""
        logger.info(f"Checking {len(paths)} markdown files...")
        
        issues = []
        files_processed = []
        
        for path in paths:
            if not path.endswith('.md'):
                continue
                
            files_processed.append(path)
            file_issues = self._check_single_file(path, auto_fix)
            issues.extend(file_issues)
        
        return self._generate_report(files_processed, issues)
    
    def _check_single_file(self, file_path: str, auto_fix: bool = False) -> List[QualityIssue]:
        """Check a single markdown file"""
        issues = []
        
        # Run markdownlint-cli2
        cmd = ['markdownlint-cli2', '--config', self.config_path, file_path]
        if auto_fix:
            cmd.append('--fix')
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                issues.extend(self._parse_markdownlint_output(result.stdout, file_path))
        except subprocess.SubprocessError as e:
            logger.error(f"Error running markdownlint on {file_path}: {e}")
        
        # Additional custom checks
        issues.extend(self._run_custom_checks(file_path))
        
        return issues
    
    def _parse_markdownlint_output(self, output: str, file_path: str) -> List[QualityIssue]:
        """Parse markdownlint output into structured issues"""
        issues = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if not line or not ':' in line:
                continue
                
            # Parse format: file:line:column rule/alias description
            match = re.match(r'^([^:]+):(\d+):?(\d+)?\s+(\S+)\s+(.+)$', line)
            if match:
                file, line_num, col_num, rule, message = match.groups()
                
                issues.append(QualityIssue(
                    file_path=file,
                    rule=rule.split('/')[0],  # Get base rule ID
                    line=int(line_num) if line_num else None,
                    column=int(col_num) if col_num else None,
                    severity='error',
                    message=message,
                    fixable=rule.split('/')[0] in self.auto_fix_rules
                ))
        
        return issues
    
    def _run_custom_checks(self, file_path: str) -> List[QualityIssue]:
        """Run custom quality checks specific to AI content"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                content = post.content
                metadata = post.metadata
            
            # Check frontmatter completeness
            required_fields = ['title', 'summary', 'tags', 'updated']
            for field in required_fields:
                if field not in metadata:
                    issues.append(QualityIssue(
                        file_path=file_path,
                        rule='FRONTMATTER001',
                        line=1,
                        column=1,
                        severity='error',
                        message=f"Missing required frontmatter field: {field}",
                        fixable=False
                    ))
            
            # Check summary length (120-160 words optimal for SEO)
            if 'summary' in metadata:
                summary_length = len(metadata['summary'].split())
                if summary_length < 120:
                    issues.append(QualityIssue(
                        file_path=file_path,
                        rule='CONTENT001',
                        line=1,
                        column=1,
                        severity='warning',
                        message=f"Summary too short: {summary_length} words (recommended: 120-160)",
                        fixable=False
                    ))
                elif summary_length > 160:
                    issues.append(QualityIssue(
                        file_path=file_path,
                        rule='CONTENT002',
                        line=1,
                        column=1,
                        severity='warning',
                        message=f"Summary too long: {summary_length} words (recommended: 120-160)",
                        fixable=False
                    ))
            
            # Check for AI-specific terminology consistency
            terminology_issues = self._check_terminology_consistency(content, file_path)
            issues.extend(terminology_issues)
            
        except Exception as e:
            logger.error(f"Error in custom checks for {file_path}: {e}")
        
        return issues
    
    def _check_terminology_consistency(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check AI/ML terminology consistency"""
        issues = []
        
        # Common AI terminology inconsistencies
        terminology_patterns = {
            r'\bA\.I\.\b': 'Use "AI" instead of "A.I."',
            r'\bM\.L\.\b': 'Use "ML" instead of "M.L."',
            r'\bmachine-learning\b': 'Use "machine learning" (no hyphen) except in URLs',
            r'\bdeep-learning\b': 'Use "deep learning" (no hyphen) except in URLs',
            r'\bartificial-intelligence\b': 'Use "artificial intelligence" (no hyphen) except in URLs'
        }
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, message in terminology_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(QualityIssue(
                        file_path=file_path,
                        rule='TERMINOLOGY001',
                        line=line_num,
                        column=1,
                        severity='warning',
                        message=message,
                        fixable=True
                    ))
        
        return issues
    
    def _generate_report(self, files_processed: List[str], issues: List[QualityIssue]) -> QualityReport:
        """Generate comprehensive quality report"""
        issues_by_severity = {'error': 0, 'warning': 0, 'info': 0}
        issues_by_rule = {}
        fixable_issues = 0
        files_with_issues = set()
        
        for issue in issues:
            issues_by_severity[issue.severity] += 1
            issues_by_rule[issue.rule] = issues_by_rule.get(issue.rule, 0) + 1
            if issue.fixable:
                fixable_issues += 1
            files_with_issues.add(issue.file_path)
        
        return QualityReport(
            timestamp=datetime.now().isoformat(),
            total_files=len(files_processed),
            files_with_issues=len(files_with_issues),
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues_by_rule=issues_by_rule,
            fixable_issues=fixable_issues,
            files_processed=files_processed,
            issues=issues
        )
    
    def migrate_files(self, paths: List[str], dry_run: bool = False, batch_size: int = 10) -> None:
        """Migrate existing files to meet quality standards"""
        logger.info(f"Migrating {len(paths)} files (dry_run={dry_run})")
        
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            for file_path in batch:
                self._migrate_single_file(file_path, dry_run)
    
    def _migrate_single_file(self, file_path: str, dry_run: bool = False) -> None:
        """Migrate a single file to meet quality standards"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                post = frontmatter.loads(original_content)
            
            # Track changes
            changes_made = []
            
            # Fix frontmatter issues
            metadata_fixes = self._fix_frontmatter_issues(post.metadata, file_path)
            if metadata_fixes:
                changes_made.extend(metadata_fixes)
            
            # Fix content issues
            content_fixes = self._fix_content_issues(post.content)
            if content_fixes:
                post.content = content_fixes['content']
                changes_made.extend(content_fixes['changes'])
            
            if changes_made:
                if dry_run:
                    logger.info(f"Would fix {file_path}: {', '.join(changes_made)}")
                else:
                    new_content = frontmatter.dumps(post)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    logger.info(f"Fixed {file_path}: {', '.join(changes_made)}")
                    
        except Exception as e:
            logger.error(f"Error migrating {file_path}: {e}")
    
    def _fix_frontmatter_issues(self, metadata: dict, file_path: str) -> List[str]:
        """Fix common frontmatter issues"""
        changes = []
        
        # Ensure required fields exist
        if 'updated' not in metadata:
            metadata['updated'] = datetime.now().strftime("%Y-%m-%d")
            changes.append("added updated date")
        
        # Standardize tags format
        if 'tags' in metadata and isinstance(metadata['tags'], str):
            metadata['tags'] = [tag.strip() for tag in metadata['tags'].split(',')]
            changes.append("standardized tags format")
        
        return changes
    
    def _fix_content_issues(self, content: str) -> Optional[dict]:
        """Fix common content issues"""
        changes = []
        fixed_content = content
        
        # Fix common emphasis-as-heading issues (MD036)
        emphasis_heading_pattern = r'^\*\*([^*]+)\*\*\s*$'
        lines = fixed_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if re.match(emphasis_heading_pattern, line):
                # Convert **Bold text** to ### Bold text
                match = re.match(emphasis_heading_pattern, line)
                if match:
                    fixed_lines.append(f"### {match.group(1)}")
                    changes.append("converted emphasis to heading")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if changes:
            fixed_content = '\n'.join(fixed_lines)
            
            # Ensure file ends with single newline
            if not fixed_content.endswith('\n'):
                fixed_content += '\n'
                changes.append("added final newline")
            elif fixed_content.endswith('\n\n'):
                fixed_content = fixed_content.rstrip('\n') + '\n'
                changes.append("fixed multiple trailing newlines")
            
            return {'content': fixed_content, 'changes': changes}
        
        return None


def find_markdown_files(paths: List[str]) -> List[str]:
    """Find all markdown files in given paths"""
    markdown_files = []
    
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.md':
            markdown_files.append(str(path))
        elif path.is_dir():
            markdown_files.extend([
                str(f) for f in path.rglob('*.md')
            ])
    
    return markdown_files


def main():
    parser = argparse.ArgumentParser(description='Markdown Quality Control CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Run quality checks')
    check_parser.add_argument('paths', nargs='*', default=['apps/site/src/content'], help='Paths to check')
    check_parser.add_argument('--fix', action='store_true', help='Auto-fix issues where possible')
    check_parser.add_argument('--config', help='Custom configuration file')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate content to quality standards')
    migrate_parser.add_argument('paths', nargs='*', default=['apps/site/src/content'], help='Paths to migrate')
    migrate_parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    migrate_parser.add_argument('--batch-size', type=int, default=10, help='Process N files at a time')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate quality report')
    report_parser.add_argument('paths', nargs='*', default=['apps/site/src/content'], help='Paths to analyze')
    report_parser.add_argument('--format', choices=['json', 'html', 'md'], default='md', help='Report format')
    report_parser.add_argument('--output', help='Output file')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize quality controller
    config_path = getattr(args, 'config', None)
    controller = MarkdownQualityController(config_path)
    
    # Find markdown files
    markdown_files = find_markdown_files(args.paths)
    if not markdown_files:
        logger.error("No markdown files found in specified paths")
        return 1
    
    # Execute command
    if args.command == 'check':
        report = controller.check_files(markdown_files, args.fix)
        print_quality_report(report)
        return 0 if report.issues_by_severity['error'] == 0 else 1
        
    elif args.command == 'migrate':
        controller.migrate_files(markdown_files, args.dry_run, args.batch_size)
        return 0
        
    elif args.command == 'report':
        report = controller.check_files(markdown_files)
        output_report(report, args.format, args.output)
        return 0


def print_quality_report(report: QualityReport):
    """Print quality report to console"""
    print(f"\n📊 Markdown Quality Report")
    print(f"=" * 50)
    print(f"Files processed: {report.total_files}")
    print(f"Files with issues: {report.files_with_issues}")
    print(f"Total issues: {report.total_issues}")
    print(f"Fixable issues: {report.fixable_issues}")
    
    if report.issues_by_severity:
        print(f"\n📈 Issues by Severity:")
        for severity, count in report.issues_by_severity.items():
            if count > 0:
                icon = "🔴" if severity == "error" else "🟡" if severity == "warning" else "🔵"
                print(f"  {icon} {severity.title()}: {count}")
    
    if report.issues_by_rule:
        print(f"\n📋 Top Issues by Rule:")
        sorted_rules = sorted(report.issues_by_rule.items(), key=lambda x: x[1], reverse=True)
        for rule, count in sorted_rules[:10]:  # Top 10
            print(f"  • {rule}: {count}")
    
    if report.issues:
        print(f"\n🔍 Issue Details:")
        for issue in report.issues[:20]:  # Show first 20
            location = f"{issue.line}:{issue.column}" if issue.line else "N/A"
            fixable = "✅" if issue.fixable else "❌"
            print(f"  {Path(issue.file_path).name}:{location} {issue.rule} {fixable} {issue.message}")


def output_report(report: QualityReport, format_type: str, output_file: Optional[str]):
    """Output report in specified format"""
    if format_type == 'json':
        content = json.dumps(asdict(report), indent=2, default=str)
    elif format_type == 'html':
        content = generate_html_report(report)
    else:  # markdown
        content = generate_markdown_report(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(content)
        logger.info(f"Report saved to {output_file}")
    else:
        print(content)


def generate_markdown_report(report: QualityReport) -> str:
    """Generate markdown format report"""
    md = f"""# Markdown Quality Report

**Generated:** {report.timestamp}

## Summary

- **Files Processed:** {report.total_files}
- **Files with Issues:** {report.files_with_issues}  
- **Total Issues:** {report.total_issues}
- **Fixable Issues:** {report.fixable_issues}

## Issues by Severity

"""
    
    for severity, count in report.issues_by_severity.items():
        if count > 0:
            md += f"- **{severity.title()}:** {count}\n"
    
    md += "\n## Issues by Rule\n\n"
    sorted_rules = sorted(report.issues_by_rule.items(), key=lambda x: x[1], reverse=True)
    for rule, count in sorted_rules:
        md += f"- **{rule}:** {count}\n"
    
    if report.issues:
        md += "\n## Detailed Issues\n\n"
        for issue in report.issues:
            location = f"{issue.line}:{issue.column}" if issue.line else "N/A"
            fixable = "✅ Fixable" if issue.fixable else "❌ Manual"
            md += f"- `{Path(issue.file_path).name}:{location}` **{issue.rule}** {fixable} - {issue.message}\n"
    
    return md


def generate_html_report(report: QualityReport) -> str:
    """Generate HTML format report"""
    # Simplified HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Markdown Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .issue {{ margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }}
            .error {{ border-color: #f44336; }}
            .warning {{ border-color: #ff9800; }}
            .info {{ border-color: #2196f3; }}
        </style>
    </head>
    <body>
        <h1>Markdown Quality Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Files Processed:</strong> {report.total_files}</p>
            <p><strong>Files with Issues:</strong> {report.files_with_issues}</p>
            <p><strong>Total Issues:</strong> {report.total_issues}</p>
            <p><strong>Fixable Issues:</strong> {report.fixable_issues}</p>
        </div>
        <h2>Issues</h2>
    """
    
    for issue in report.issues:
        location = f"{issue.line}:{issue.column}" if issue.line else "N/A"
        fixable = "✅ Fixable" if issue.fixable else "❌ Manual"
        html += f"""
        <div class="issue {issue.severity}">
            <strong>{Path(issue.file_path).name}:{location}</strong> {issue.rule} {fixable}<br>
            {issue.message}
        </div>
        """
    
    html += "</body></html>"
    return html


if __name__ == '__main__':
    sys.exit(main())