#!/usr/bin/env python3
"""
Content Freshness Monitoring Script

This script monitors content freshness and generates alerts when content
becomes stale or when there are content update issues.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics


@dataclass
class FreshnessMetric:
    file_path: str
    last_modified: datetime
    age_days: int
    content_type: str
    title: str = ""
    is_stale: bool = False
    is_very_stale: bool = False


@dataclass
class FreshnessReport:
    timestamp: datetime
    total_files: int
    fresh_files: int
    stale_files: int
    very_stale_files: int
    avg_age_days: float
    oldest_file: Optional[FreshnessMetric]
    newest_file: Optional[FreshnessMetric]
    stale_threshold_days: int
    very_stale_threshold_days: int
    alert_triggered: bool = False
    alert_reason: str = ""


class ContentFreshnessChecker:
    def __init__(self, content_dir: Path, verbose: bool = False):
        self.content_dir = Path(content_dir)
        self.verbose = verbose
        self.metrics: List[FreshnessMetric] = []
        
        # Thresholds (configurable)
        self.stale_threshold_days = 30
        self.very_stale_threshold_days = 90
        self.alert_threshold_ratio = 0.5  # Alert if >50% of content is stale

    def log(self, message: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level.upper()}] {message}")

    def extract_title_from_content(self, file_path: Path) -> str:
        """Extract title from markdown frontmatter."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip().startswith('---'):
                import yaml
                try:
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        if isinstance(frontmatter, dict) and 'title' in frontmatter:
                            return str(frontmatter['title'])
                except yaml.YAMLError:
                    pass
            
            return file_path.stem
            
        except Exception:
            return file_path.stem

    def determine_content_type(self, file_path: Path) -> str:
        """Determine content type based on file path."""
        if 'articles' in file_path.parts:
            return 'article'
        elif 'glossary' in file_path.parts:
            return 'glossary'
        elif 'docs' in file_path.parts:
            return 'documentation'
        else:
            return 'content'

    def analyze_file_freshness(self, file_path: Path) -> FreshnessMetric:
        """Analyze freshness of a single file."""
        try:
            # Get file modification time
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.fromtimestamp(mtime)
            
            # Calculate age
            now = datetime.now()
            age_days = (now - last_modified).days
            
            # Extract metadata
            title = self.extract_title_from_content(file_path)
            content_type = self.determine_content_type(file_path)
            
            # Determine staleness
            is_stale = age_days > self.stale_threshold_days
            is_very_stale = age_days > self.very_stale_threshold_days
            
            return FreshnessMetric(
                file_path=str(file_path.relative_to(self.content_dir)),
                last_modified=last_modified,
                age_days=age_days,
                content_type=content_type,
                title=title,
                is_stale=is_stale,
                is_very_stale=is_very_stale
            )
            
        except Exception as e:
            self.log(f"Error analyzing {file_path}: {e}", 'error')
            return None

    def scan_content_directory(self) -> List[FreshnessMetric]:
        """Scan all content files and analyze freshness."""
        self.log("Scanning content directory for freshness analysis...", 'info')
        
        # Find all markdown files
        md_files = list(self.content_dir.rglob('*.md'))
        
        if not md_files:
            self.log("No markdown files found", 'warning')
            return []
        
        self.log(f"Found {len(md_files)} content files to analyze", 'info')
        
        # Analyze each file
        metrics = []
        for file_path in md_files:
            metric = self.analyze_file_freshness(file_path)
            if metric:
                metrics.append(metric)
        
        self.metrics = metrics
        return metrics

    def generate_freshness_report(self, max_age_days: int = None, 
                                 alert_threshold: float = None) -> FreshnessReport:
        """Generate comprehensive freshness report."""
        if not self.metrics:
            self.scan_content_directory()
        
        if not self.metrics:
            return FreshnessReport(
                timestamp=datetime.now(),
                total_files=0,
                fresh_files=0,
                stale_files=0,
                very_stale_files=0,
                avg_age_days=0.0,
                oldest_file=None,
                newest_file=None,
                stale_threshold_days=self.stale_threshold_days,
                very_stale_threshold_days=self.very_stale_threshold_days
            )
        
        # Override thresholds if provided
        if max_age_days:
            self.stale_threshold_days = max_age_days
        if alert_threshold:
            self.alert_threshold_ratio = alert_threshold
        
        # Calculate metrics
        total_files = len(self.metrics)
        fresh_files = sum(1 for m in self.metrics if not m.is_stale)
        stale_files = sum(1 for m in self.metrics if m.is_stale and not m.is_very_stale)
        very_stale_files = sum(1 for m in self.metrics if m.is_very_stale)
        
        ages = [m.age_days for m in self.metrics]
        avg_age_days = statistics.mean(ages) if ages else 0.0
        
        # Find oldest and newest files
        oldest_file = max(self.metrics, key=lambda m: m.age_days) if self.metrics else None
        newest_file = min(self.metrics, key=lambda m: m.age_days) if self.metrics else None
        
        # Determine if alert should be triggered
        stale_ratio = (stale_files + very_stale_files) / total_files if total_files > 0 else 0
        alert_triggered = stale_ratio > self.alert_threshold_ratio
        alert_reason = ""
        
        if alert_triggered:
            alert_reason = f"Stale content ratio ({stale_ratio:.1%}) exceeds threshold ({self.alert_threshold_ratio:.1%})"
        elif very_stale_files > 0:
            alert_reason = f"{very_stale_files} files are very stale (>{self.very_stale_threshold_days} days)"
        
        return FreshnessReport(
            timestamp=datetime.now(),
            total_files=total_files,
            fresh_files=fresh_files,
            stale_files=stale_files,
            very_stale_files=very_stale_files,
            avg_age_days=avg_age_days,
            oldest_file=oldest_file,
            newest_file=newest_file,
            stale_threshold_days=self.stale_threshold_days,
            very_stale_threshold_days=self.very_stale_threshold_days,
            alert_triggered=alert_triggered,
            alert_reason=alert_reason
        )

    def print_report(self, report: FreshnessReport):
        """Print freshness report to console."""
        print("\n" + "="*60)
        print("CONTENT FRESHNESS REPORT")
        print("="*60)
        
        print(f"Analysis Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Content Directory: {self.content_dir}")
        print()
        
        # Summary stats
        print("SUMMARY:")
        print(f"  Total files: {report.total_files}")
        print(f"  Fresh files (<{report.stale_threshold_days} days): {report.fresh_files}")
        print(f"  Stale files ({report.stale_threshold_days}-{report.very_stale_threshold_days} days): {report.stale_files}")
        print(f"  Very stale files (>{report.very_stale_threshold_days} days): {report.very_stale_files}")
        print(f"  Average age: {report.avg_age_days:.1f} days")
        print()
        
        # Percentages
        if report.total_files > 0:
            fresh_pct = (report.fresh_files / report.total_files) * 100
            stale_pct = (report.stale_files / report.total_files) * 100
            very_stale_pct = (report.very_stale_files / report.total_files) * 100
            
            print("PERCENTAGES:")
            print(f"  Fresh: {fresh_pct:.1f}%")
            print(f"  Stale: {stale_pct:.1f}%")
            print(f"  Very stale: {very_stale_pct:.1f}%")
            print()
        
        # Extremes
        if report.oldest_file:
            print(f"OLDEST FILE:")
            print(f"  {report.oldest_file.file_path}")
            print(f"  Title: {report.oldest_file.title}")
            print(f"  Age: {report.oldest_file.age_days} days")
            print(f"  Last modified: {report.oldest_file.last_modified.strftime('%Y-%m-%d')}")
            print()
        
        if report.newest_file:
            print(f"NEWEST FILE:")
            print(f"  {report.newest_file.file_path}")
            print(f"  Title: {report.newest_file.title}")
            print(f"  Age: {report.newest_file.age_days} days")
            print(f"  Last modified: {report.newest_file.last_modified.strftime('%Y-%m-%d')}")
            print()
        
        # Alert status
        if report.alert_triggered:
            print("🚨 ALERT TRIGGERED:")
            print(f"  Reason: {report.alert_reason}")
        else:
            print("✅ No alerts triggered")
        
        # Show stale files if verbose or if there are alerts
        if self.verbose or report.alert_triggered:
            stale_files = [m for m in self.metrics if m.is_stale]
            if stale_files:
                print(f"\nSTALE FILES ({len(stale_files)}):")
                print("-" * 40)
                
                # Group by content type
                by_type = {}
                for metric in stale_files:
                    if metric.content_type not in by_type:
                        by_type[metric.content_type] = []
                    by_type[metric.content_type].append(metric)
                
                for content_type, files in sorted(by_type.items()):
                    print(f"\n{content_type.title()} ({len(files)} files):")
                    
                    # Sort by age (oldest first)
                    files.sort(key=lambda m: m.age_days, reverse=True)
                    
                    for metric in files[:10]:  # Show top 10 oldest
                        indicator = "🔴" if metric.is_very_stale else "🟡"
                        print(f"  {indicator} {metric.file_path}")
                        print(f"     Title: {metric.title}")
                        print(f"     Age: {metric.age_days} days (modified {metric.last_modified.strftime('%Y-%m-%d')})")
                    
                    if len(files) > 10:
                        print(f"  ... and {len(files) - 10} more files")

    def save_report_json(self, report: FreshnessReport, output_file: Path):
        """Save report to JSON file."""
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'content_directory': str(self.content_dir),
            'thresholds': {
                'stale_days': report.stale_threshold_days,
                'very_stale_days': report.very_stale_threshold_days,
                'alert_threshold': self.alert_threshold_ratio
            },
            'summary': {
                'total_files': report.total_files,
                'fresh_files': report.fresh_files,
                'stale_files': report.stale_files,
                'very_stale_files': report.very_stale_files,
                'avg_age_days': report.avg_age_days,
                'fresh_percentage': (report.fresh_files / report.total_files * 100) if report.total_files > 0 else 0,
                'stale_percentage': ((report.stale_files + report.very_stale_files) / report.total_files * 100) if report.total_files > 0 else 0
            },
            'alert': {
                'triggered': report.alert_triggered,
                'reason': report.alert_reason
            },
            'extremes': {
                'oldest_file': {
                    'path': report.oldest_file.file_path,
                    'title': report.oldest_file.title,
                    'age_days': report.oldest_file.age_days,
                    'last_modified': report.oldest_file.last_modified.isoformat()
                } if report.oldest_file else None,
                'newest_file': {
                    'path': report.newest_file.file_path,
                    'title': report.newest_file.title,
                    'age_days': report.newest_file.age_days,
                    'last_modified': report.newest_file.last_modified.isoformat()
                } if report.newest_file else None
            },
            'files': [
                {
                    'path': m.file_path,
                    'title': m.title,
                    'content_type': m.content_type,
                    'age_days': m.age_days,
                    'last_modified': m.last_modified.isoformat(),
                    'is_stale': m.is_stale,
                    'is_very_stale': m.is_very_stale
                }
                for m in self.metrics
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.log(f"Report saved to {output_file}", 'info')

    def send_alert(self, report: FreshnessReport):
        """Send alert notifications if configured."""
        if not report.alert_triggered:
            return
        
        self.log("Sending freshness alert...", 'info')
        
        # Slack notification (if configured)
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            try:
                import requests
                
                message = {
                    "text": "🚨 Content Freshness Alert",
                    "attachments": [{
                        "color": "warning",
                        "title": "AI Knowledge Website - Content Freshness Alert",
                        "fields": [
                            {"title": "Alert Reason", "value": report.alert_reason, "short": False},
                            {"title": "Total Files", "value": str(report.total_files), "short": True},
                            {"title": "Stale Files", "value": str(report.stale_files + report.very_stale_files), "short": True},
                            {"title": "Very Stale Files", "value": str(report.very_stale_files), "short": True},
                            {"title": "Average Age", "value": f"{report.avg_age_days:.1f} days", "short": True}
                        ],
                        "footer": "Content Freshness Monitor",
                        "ts": int(report.timestamp.timestamp())
                    }]
                }
                
                response = requests.post(slack_webhook, json=message, timeout=10)
                if response.status_code == 200:
                    self.log("Slack alert sent successfully", 'info')
                else:
                    self.log(f"Slack alert failed: {response.status_code}", 'warning')
                    
            except Exception as e:
                self.log(f"Failed to send Slack alert: {e}", 'error')
        
        # Email notification (if configured)
        email_webhook = os.environ.get('EMAIL_WEBHOOK_URL')
        if email_webhook:
            # Similar implementation for email alerts
            self.log("Email notifications not implemented yet", 'info')
        
        # Generic webhook (if configured)
        webhook_url = os.environ.get('ALERT_WEBHOOK_URL')
        if webhook_url:
            try:
                import requests
                
                alert_data = {
                    'alert_type': 'content_freshness',
                    'severity': 'warning',
                    'message': report.alert_reason,
                    'timestamp': report.timestamp.isoformat(),
                    'metrics': {
                        'total_files': report.total_files,
                        'stale_files': report.stale_files + report.very_stale_files,
                        'stale_percentage': ((report.stale_files + report.very_stale_files) / report.total_files * 100) if report.total_files > 0 else 0
                    }
                }
                
                response = requests.post(webhook_url, json=alert_data, timeout=10)
                if response.status_code == 200:
                    self.log("Generic webhook alert sent successfully", 'info')
                else:
                    self.log(f"Generic webhook alert failed: {response.status_code}", 'warning')
                    
            except Exception as e:
                self.log(f"Failed to send webhook alert: {e}", 'error')


def main():
    parser = argparse.ArgumentParser(description='Check content freshness for AI Knowledge Website')
    parser.add_argument('--content-dir',
                       type=str,
                       default='apps/site/src/content',
                       help='Path to content directory')
    parser.add_argument('--max-age-days',
                       type=int,
                       default=30,
                       help='Maximum age in days before content is considered stale')
    parser.add_argument('--alert-threshold',
                       type=float,
                       default=0.5,
                       help='Alert if more than this fraction of content is stale (0.0-1.0)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--output-report',
                       type=str,
                       help='Save report to JSON file')
    parser.add_argument('--send-alerts',
                       action='store_true',
                       help='Send alerts if thresholds are exceeded')
    parser.add_argument('--fail-on-alerts',
                       action='store_true',
                       help='Exit with error code if alerts are triggered')
    
    args = parser.parse_args()
    
    # Validate content directory
    content_dir = Path(args.content_dir)
    if not content_dir.exists():
        print(f"Error: Content directory not found: {content_dir}")
        sys.exit(1)
    
    # Create freshness checker
    checker = ContentFreshnessChecker(content_dir, verbose=args.verbose)
    
    try:
        # Generate report
        report = checker.generate_freshness_report(
            max_age_days=args.max_age_days,
            alert_threshold=args.alert_threshold
        )
        
        # Print report
        checker.print_report(report)
        
        # Save report if requested
        if args.output_report:
            checker.save_report_json(report, Path(args.output_report))
        
        # Send alerts if requested
        if args.send_alerts and report.alert_triggered:
            checker.send_alert(report)
        
        # Determine exit code
        if report.alert_triggered and args.fail_on_alerts:
            print(f"\n❌ Content freshness check FAILED: {report.alert_reason}")
            sys.exit(1)
        elif report.alert_triggered:
            print(f"\n⚠️ Content freshness alert: {report.alert_reason}")
        else:
            print(f"\n✅ Content freshness check PASSED")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error checking content freshness: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()