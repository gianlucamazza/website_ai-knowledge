#!/usr/bin/env python3
"""
Markdown Quality Dashboard
Real-time quality metrics and violation tracking for AI Knowledge Website
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

try:
    import frontmatter
except ImportError:
    print("Installing required dependency: python-frontmatter")
    os.system("pip install python-frontmatter")
    import frontmatter

@dataclass
class QualityMetrics:
    """Quality metrics for a content file"""
    file_path: str
    file_size_lines: int
    word_count: int
    violations_count: int
    violations_by_rule: Dict[str, int]
    last_modified: str
    content_type: str  # glossary, article, doc
    has_frontmatter: bool
    frontmatter_valid: bool
    
@dataclass 
class ProjectQualityReport:
    """Overall project quality report"""
    timestamp: str
    total_files: int
    total_violations: int
    violations_by_rule: Dict[str, int]
    violations_by_file_type: Dict[str, int]
    file_metrics: List[QualityMetrics]
    quality_score: float
    top_violators: List[Tuple[str, int]]
    improvement_suggestions: List[str]

class QualityDashboard:
    """Enterprise quality dashboard with comprehensive metrics"""
    
    def __init__(self, content_dir: str, project_root: str = None):
        self.content_dir = Path(content_dir)
        self.project_root = Path(project_root) if project_root else self.content_dir.parent.parent.parent
        
        # Rule descriptions for better reporting
        self.rule_descriptions = {
            'MD001': 'Heading levels increment by one',
            'MD003': 'Heading style consistency', 
            'MD004': 'Unordered list style consistency',
            'MD007': 'Unordered list indentation',
            'MD009': 'Trailing spaces',
            'MD012': 'Multiple consecutive blank lines',
            'MD013': 'Line length violations',
            'MD022': 'Headings surrounded by blank lines',
            'MD025': 'Multiple top-level headings',
            'MD026': 'Trailing punctuation in headings',
            'MD030': 'Spaces after list markers',
            'MD031': 'Fenced code blocks spacing',
            'MD032': 'Lists surrounded by blank lines',
            'MD033': 'Inline HTML usage',
            'MD034': 'Bare URL usage', 
            'MD036': 'Emphasis used instead of heading',
            'MD040': 'Fenced code blocks language',
            'MD041': 'First line heading level',
            'MD045': 'Images alt text',
            'MD046': 'Code block style consistency',
            'MD047': 'Files end with newline',
            'MD048': 'Code fence style consistency',
            'MD049': 'Emphasis style consistency',
            'MD050': 'Strong style consistency',
            'MD051': 'Link fragments',
            'MD052': 'Reference links',
            'MD053': 'Link reference definitions'
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0,     # 0 violations
            'good': 5,          # 1-5 violations
            'fair': 15,         # 6-15 violations  
            'poor': 30,         # 16-30 violations
            'critical': 999     # 30+ violations
        }
        
    def run_markdownlint(self) -> Tuple[str, int]:
        """Run markdownlint and return output and exit code"""
        try:
            # Change to the site directory to run npm command
            site_dir = self.project_root / 'apps' / 'site'
            
            result = subprocess.run(
                ['npm', 'run', 'lint'],
                cwd=site_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return "Error: markdownlint timeout", 1
        except FileNotFoundError:
            return "Error: npm not found", 1
        except Exception as e:
            return f"Error: {str(e)}", 1
            
    def parse_violations(self, lint_output: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse markdownlint output into structured violations"""
        violations_by_file = defaultdict(list)
        
        # Pattern for markdownlint output: file:line:column rule/name description [Context: "text"]
        pattern = r'^(.+):(\d+)(?::(\d+))?\s+(MD\d+)/([^\s]+)\s+(.+?)(?:\s+\[Context:\s*"([^"]*)".*\])?$'
        
        for line in lint_output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(pattern, line)
            if match:
                file_path, line_num, col_num, rule_code, rule_name, description, context = match.groups()
                
                violation = {
                    'file': file_path,
                    'line': int(line_num),
                    'column': int(col_num) if col_num else None,
                    'rule_code': rule_code,
                    'rule_name': rule_name,
                    'description': description,
                    'context': context,
                    'rule_description': self.rule_descriptions.get(rule_code, 'Unknown rule')
                }
                
                violations_by_file[file_path].append(violation)
                
        return dict(violations_by_file)
        
    def analyze_file_metrics(self, file_path: Path) -> QualityMetrics:
        """Analyze detailed metrics for a single file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Basic metrics
            file_size_lines = len(lines)
            word_count = len(content.split())
            
            # Determine content type from path
            if 'glossary' in str(file_path):
                content_type = 'glossary'
            elif 'articles' in str(file_path):
                content_type = 'article'
            elif 'docs' in str(file_path):
                content_type = 'doc'
            else:
                content_type = 'other'
                
            # Check frontmatter
            has_frontmatter = content.startswith('---')
            frontmatter_valid = False
            
            if has_frontmatter:
                try:
                    post = frontmatter.loads(content)
                    frontmatter_valid = True
                except:
                    frontmatter_valid = False
                    
            # Get file modification time
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            
            return QualityMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                file_size_lines=file_size_lines,
                word_count=word_count,
                violations_count=0,  # Will be populated later
                violations_by_rule={},  # Will be populated later
                last_modified=last_modified,
                content_type=content_type,
                has_frontmatter=has_frontmatter,
                frontmatter_valid=frontmatter_valid
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
            
    def calculate_quality_score(self, violations_count: int, file_size: int) -> float:
        """Calculate quality score (0-100) based on violations and file size"""
        if violations_count == 0:
            return 100.0
            
        # Normalize by file size (violations per 100 lines)
        normalized_violations = (violations_count / max(file_size, 1)) * 100
        
        # Score formula: 100 - (violations_per_100_lines * penalty_factor)
        penalty_factor = 10  # Each violation per 100 lines costs 10 points
        score = max(0, 100 - (normalized_violations * penalty_factor))
        
        return round(score, 1)
        
    def generate_improvement_suggestions(self, violations_by_rule: Dict[str, int]) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Rule-specific suggestions
        rule_suggestions = {
            'MD047': 'Run auto-fix: All files should end with a single newline',
            'MD013': 'Consider breaking long lines at sentence boundaries or using line breaks',
            'MD025': 'Use only one H1 heading per document (usually handled by frontmatter title)',
            'MD026': 'Remove trailing punctuation from headings',
            'MD032': 'Add blank lines before and after lists',
            'MD031': 'Add blank lines before and after code blocks',
            'MD040': 'Specify programming language for code blocks (```python, ```javascript)',
            'MD022': 'Add blank lines before and after headings',
            'MD009': 'Remove trailing spaces from lines',
            'MD036': 'Use proper headings instead of bold text for section titles'
        }
        
        # Prioritize by frequency
        sorted_rules = sorted(violations_by_rule.items(), key=lambda x: x[1], reverse=True)
        
        for rule, count in sorted_rules[:5]:  # Top 5 most frequent violations
            if rule in rule_suggestions:
                suggestions.append(f"{rule} ({count} violations): {rule_suggestions[rule]}")
                
        # General suggestions
        if sum(violations_by_rule.values()) > 20:
            suggestions.append("Run the migration script: python scripts/migrate_current_violations.py apps/site/src/content --apply")
            
        if violations_by_rule.get('MD013', 0) > 10:
            suggestions.append("Consider using a text editor with word wrap for better content editing")
            
        return suggestions[:8]  # Limit to 8 suggestions
        
    def generate_report(self) -> ProjectQualityReport:
        """Generate comprehensive quality report"""
        print("🔍 Analyzing markdown quality...")
        
        # Run markdownlint
        lint_output, exit_code = self.run_markdownlint()
        
        # Parse violations
        violations_by_file = self.parse_violations(lint_output)
        
        # Find all markdown files
        md_files = []
        for pattern in ['**/*.md']:
            md_files.extend(self.content_dir.rglob(pattern))
            
        # Also check docs directory
        docs_dir = self.project_root / 'docs'
        if docs_dir.exists():
            md_files.extend(docs_dir.rglob('*.md'))
            
        print(f"📊 Found {len(md_files)} markdown files")
        
        # Analyze each file
        file_metrics = []
        total_violations = 0
        violations_by_rule = Counter()
        violations_by_file_type = Counter()
        
        for md_file in md_files:
            metrics = self.analyze_file_metrics(md_file)
            if metrics:
                # Add violation data
                file_violations = violations_by_file.get(str(md_file), [])
                metrics.violations_count = len(file_violations)
                metrics.violations_by_rule = Counter(v['rule_code'] for v in file_violations)
                
                # Update totals
                total_violations += metrics.violations_count
                violations_by_rule.update(metrics.violations_by_rule)
                violations_by_file_type[metrics.content_type] += metrics.violations_count
                
                file_metrics.append(metrics)
                
        # Calculate overall quality score
        avg_violations_per_file = total_violations / max(len(file_metrics), 1)
        avg_file_size = sum(m.file_size_lines for m in file_metrics) / max(len(file_metrics), 1)
        overall_quality_score = self.calculate_quality_score(avg_violations_per_file, avg_file_size / len(file_metrics) if file_metrics else 1)
        
        # Find top violators
        top_violators = sorted(
            [(m.file_path, m.violations_count) for m in file_metrics if m.violations_count > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Generate improvement suggestions
        improvement_suggestions = self.generate_improvement_suggestions(dict(violations_by_rule))
        
        return ProjectQualityReport(
            timestamp=datetime.now().isoformat(),
            total_files=len(file_metrics),
            total_violations=total_violations,
            violations_by_rule=dict(violations_by_rule),
            violations_by_file_type=dict(violations_by_file_type),
            file_metrics=file_metrics,
            quality_score=overall_quality_score,
            top_violators=top_violators,
            improvement_suggestions=improvement_suggestions
        )
        
    def print_dashboard(self, report: ProjectQualityReport):
        """Print formatted quality dashboard to console"""
        print("\\n" + "="*80)
        print("🎯 AI KNOWLEDGE WEBSITE - QUALITY DASHBOARD")
        print("="*80)
        
        # Header
        timestamp = datetime.fromisoformat(report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"📅 Generated: {timestamp}")
        print(f"📁 Content Directory: {self.content_dir}")
        print()
        
        # Quality Score
        score = report.quality_score
        if score >= 95:
            score_emoji = "🏆"
            score_level = "EXCELLENT"
        elif score >= 85:
            score_emoji = "🥇"
            score_level = "VERY GOOD"
        elif score >= 75:
            score_emoji = "🥈"
            score_level = "GOOD"
        elif score >= 60:
            score_emoji = "🥉"
            score_level = "FAIR"
        else:
            score_emoji = "⚠️"
            score_level = "NEEDS IMPROVEMENT"
            
        print(f"{score_emoji} OVERALL QUALITY SCORE: {score}/100 ({score_level})")
        print()
        
        # Summary Stats
        print("📈 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Total Files: {report.total_files}")
        print(f"Total Violations: {report.total_violations}")
        print(f"Average Violations per File: {report.total_violations / max(report.total_files, 1):.1f}")
        print()
        
        # Violations by Rule (Top 10)
        if report.violations_by_rule:
            print("🚨 TOP VIOLATION TYPES")
            print("-" * 40)
            sorted_rules = sorted(report.violations_by_rule.items(), key=lambda x: x[1], reverse=True)
            
            for i, (rule, count) in enumerate(sorted_rules[:10], 1):
                description = self.rule_descriptions.get(rule, "Unknown rule")
                percentage = (count / report.total_violations * 100) if report.total_violations > 0 else 0
                print(f"{i:2}. {rule}: {count:3} violations ({percentage:4.1f}%) - {description}")
            print()
            
        # Violations by Content Type
        if report.violations_by_file_type:
            print("📂 VIOLATIONS BY CONTENT TYPE")
            print("-" * 40)
            for content_type, count in sorted(report.violations_by_file_type.items(), key=lambda x: x[1], reverse=True):
                print(f"{content_type.capitalize():12}: {count:3} violations")
            print()
            
        # Top Violators
        if report.top_violators:
            print("🎯 FILES NEEDING MOST ATTENTION")
            print("-" * 40)
            for i, (file_path, violations) in enumerate(report.top_violators[:10], 1):
                file_name = Path(file_path).name
                print(f"{i:2}. {file_name:30} ({violations:2} violations)")
            print()
            
        # Improvement Suggestions
        if report.improvement_suggestions:
            print("💡 IMPROVEMENT SUGGESTIONS")
            print("-" * 40)
            for i, suggestion in enumerate(report.improvement_suggestions, 1):
                print(f"{i}. {suggestion}")
            print()
            
        # Quick Actions
        print("⚡ QUICK ACTIONS")
        print("-" * 40)
        print("1. Fix common issues: python scripts/migrate_current_violations.py apps/site/src/content --apply")
        print("2. Run full fixer: python scripts/markdown_quality_fixer.py apps/site/src/content")
        print("3. Check results: npm run lint")
        print("4. Install pre-commit: pip install pre-commit && pre-commit install")
        print()
        
        print("="*80)
        
    def save_json_report(self, report: ProjectQualityReport, output_path: str):
        """Save detailed report as JSON"""
        report_dict = asdict(report)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        print(f"📄 Detailed JSON report saved: {output_path}")
        
    def save_html_report(self, report: ProjectQualityReport, output_path: str):
        """Save interactive HTML report"""
        # Simple HTML report template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Knowledge Website - Quality Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8f9fa; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
        .score { font-size: 2em; font-weight: bold; text-align: center; margin: 20px 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-card h3 { margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .violation-item { padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
        .violation-count { font-weight: bold; color: #d63384; }
        .suggestions { background: #e7f3ff; border-left: 4px solid #0066cc; padding: 15px; margin: 20px 0; }
        .file-list { max-height: 400px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        .excellent { color: #28a745; }
        .good { color: #20c997; }
        .fair { color: #ffc107; }
        .poor { color: #fd7e14; }
        .critical { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 AI Knowledge Website Quality Dashboard</h1>
        <p>Generated: {timestamp}</p>
        <p>Content Directory: {content_dir}</p>
    </div>
    
    <div class="score {score_class}">
        Quality Score: {quality_score}/100 ({score_level})
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>📊 Summary</h3>
            <p><strong>Total Files:</strong> {total_files}</p>
            <p><strong>Total Violations:</strong> {total_violations}</p>
            <p><strong>Avg Violations/File:</strong> {avg_violations:.1f}</p>
        </div>
        
        <div class="metric-card">
            <h3>🚨 Top Violation Types</h3>
            {violations_html}
        </div>
        
        <div class="metric-card">
            <h3>📂 By Content Type</h3>
            {content_types_html}
        </div>
    </div>
    
    <div class="metric-card">
        <h3>🎯 Files Needing Attention</h3>
        <div class="file-list">
            <table>
                <thead>
                    <tr><th>File</th><th>Violations</th><th>Type</th></tr>
                </thead>
                <tbody>
                    {top_files_html}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="suggestions">
        <h3>💡 Improvement Suggestions</h3>
        {suggestions_html}
    </div>
    
    <div class="metric-card">
        <h3>⚡ Quick Actions</h3>
        <ol>
            <li><code>python scripts/migrate_current_violations.py apps/site/src/content --apply</code></li>
            <li><code>python scripts/markdown_quality_fixer.py apps/site/src/content</code></li>
            <li><code>npm run lint</code></li>
            <li><code>pip install pre-commit && pre-commit install</code></li>
        </ol>
    </div>
</body>
</html>
        """
        
        # Prepare data
        timestamp = datetime.fromisoformat(report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        score_class = 'excellent' if report.quality_score >= 95 else 'good' if report.quality_score >= 85 else 'fair' if report.quality_score >= 75 else 'poor' if report.quality_score >= 60 else 'critical'
        score_level = 'EXCELLENT' if report.quality_score >= 95 else 'VERY GOOD' if report.quality_score >= 85 else 'GOOD' if report.quality_score >= 75 else 'FAIR' if report.quality_score >= 60 else 'NEEDS IMPROVEMENT'
        
        # Generate violations HTML
        violations_html = ""
        sorted_rules = sorted(report.violations_by_rule.items(), key=lambda x: x[1], reverse=True)[:10]
        for rule, count in sorted_rules:
            description = self.rule_descriptions.get(rule, "Unknown rule")
            violations_html += f'<div class="violation-item"><span class="violation-count">{rule}</span>: {count} - {description}</div>'
            
        # Generate content types HTML
        content_types_html = ""
        for content_type, count in sorted(report.violations_by_file_type.items(), key=lambda x: x[1], reverse=True):
            content_types_html += f'<div class="violation-item"><strong>{content_type.capitalize()}</strong>: {count} violations</div>'
            
        # Generate top files HTML
        top_files_html = ""
        for file_path, violations in report.top_violators[:15]:
            file_name = Path(file_path).name
            content_type = next((m.content_type for m in report.file_metrics if m.file_path == file_path), 'unknown')
            top_files_html += f'<tr><td>{file_name}</td><td>{violations}</td><td>{content_type}</td></tr>'
            
        # Generate suggestions HTML
        suggestions_html = "<ol>"
        for suggestion in report.improvement_suggestions:
            suggestions_html += f"<li>{suggestion}</li>"
        suggestions_html += "</ol>"
        
        # Fill template
        html_content = html_template.format(
            timestamp=timestamp,
            content_dir=self.content_dir,
            quality_score=report.quality_score,
            score_class=score_class,
            score_level=score_level,
            total_files=report.total_files,
            total_violations=report.total_violations,
            avg_violations=report.total_violations / max(report.total_files, 1),
            violations_html=violations_html,
            content_types_html=content_types_html,
            top_files_html=top_files_html,
            suggestions_html=suggestions_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        print(f"🌐 Interactive HTML report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Knowledge Website Quality Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dashboard
  python quality_dashboard.py apps/site/src/content
  
  # Save detailed reports
  python quality_dashboard.py apps/site/src/content --json-output quality_report.json --html-output quality_report.html
        """
    )
    
    parser.add_argument(
        'content_dir',
        help='Directory containing markdown content'
    )
    
    parser.add_argument(
        '--project-root',
        help='Project root directory (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--json-output',
        help='Save detailed JSON report to file'
    )
    
    parser.add_argument(
        '--html-output', 
        help='Save interactive HTML report to file'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, suppress detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize dashboard
        dashboard = QualityDashboard(args.content_dir, args.project_root)
        
        # Generate report
        report = dashboard.generate_report()
        
        # Display dashboard
        if not args.quiet:
            dashboard.print_dashboard(report)
        else:
            score = report.quality_score
            print(f"Quality Score: {score}/100 | Files: {report.total_files} | Violations: {report.total_violations}")
            
        # Save reports if requested
        if args.json_output:
            dashboard.save_json_report(report, args.json_output)
            
        if args.html_output:
            dashboard.save_html_report(report, args.html_output)
            
        # Exit code based on quality
        if report.total_violations == 0:
            sys.exit(0)  # Perfect quality
        elif report.total_violations <= 10:
            sys.exit(0)  # Acceptable quality
        else:
            sys.exit(1)  # Quality needs improvement
            
    except KeyboardInterrupt:
        print("\\nDashboard generation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()