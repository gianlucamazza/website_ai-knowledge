#!/usr/bin/env python3
"""
Quality Dashboard Generator
Creates HTML dashboard for markdown quality metrics and trends

Usage:
    python quality-dashboard.py [--data-dir=DIR] [--output=FILE] [--days=N]
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import glob

class QualityDashboard:
    """Generate HTML dashboard for quality metrics"""
    
    def __init__(self, data_dir: str = "quality-metrics", days: int = 30):
        self.data_dir = Path(data_dir)
        self.days = days
        
    def generate_dashboard(self, output_file: str = "quality-dashboard.html") -> str:
        """Generate complete quality dashboard"""
        
        # Load quality data
        metrics_data = self._load_metrics_data()
        trend_data = self._calculate_trends(metrics_data)
        
        # Generate HTML
        html = self._generate_html_dashboard(metrics_data, trend_data)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html)
            
        return output_file
    
    def _load_metrics_data(self) -> List[Dict]:
        """Load all quality metrics files"""
        metrics = []
        
        if not self.data_dir.exists():
            return metrics
            
        # Load all JSON files
        pattern = str(self.data_dir / "quality-metrics-*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    # Add filename for reference
                    data['source_file'] = Path(file_path).name
                    metrics.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.get('timestamp', ''))
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=self.days)
        filtered_metrics = []
        
        for metric in metrics:
            try:
                metric_date = datetime.fromisoformat(metric['timestamp'])
                if metric_date >= cutoff_date:
                    filtered_metrics.append(metric)
            except:
                # Include if we can't parse the date
                filtered_metrics.append(metric)
        
        return filtered_metrics
    
    def _calculate_trends(self, metrics_data: List[Dict]) -> Dict:
        """Calculate trend analysis from metrics"""
        if not metrics_data:
            return {'trends': {}, 'current': {}}
        
        # Current metrics (latest)
        current = metrics_data[-1] if metrics_data else {}
        
        # Trend calculations
        trends = {}
        
        if len(metrics_data) >= 2:
            previous = metrics_data[-2]
            
            # Quality score trend
            current_score = current.get('quality_score', 0)
            previous_score = previous.get('quality_score', 0)
            trends['quality_score'] = {
                'current': current_score,
                'previous': previous_score,
                'change': current_score - previous_score,
                'direction': 'up' if current_score > previous_score else 'down' if current_score < previous_score else 'stable'
            }
            
            # Issue count trends
            current_issues = current.get('total_issues', 0)
            previous_issues = previous.get('total_issues', 0)
            trends['total_issues'] = {
                'current': current_issues,
                'previous': previous_issues,
                'change': current_issues - previous_issues,
                'direction': 'down' if current_issues < previous_issues else 'up' if current_issues > previous_issues else 'stable'
            }
            
            # File count trend
            current_files = current.get('total_files', 0)
            previous_files = previous.get('total_files', 0)
            trends['total_files'] = {
                'current': current_files,
                'previous': previous_files,
                'change': current_files - previous_files,
                'direction': 'up' if current_files > previous_files else 'down' if current_files < previous_files else 'stable'
            }
        
        # Historical trends over time
        if len(metrics_data) > 1:
            dates = []
            quality_scores = []
            issue_counts = []
            
            for metric in metrics_data[-10:]:  # Last 10 data points
                try:
                    date = datetime.fromisoformat(metric['timestamp'])
                    dates.append(date.strftime('%m/%d'))
                    quality_scores.append(metric.get('quality_score', 0))
                    issue_counts.append(metric.get('total_issues', 0))
                except:
                    continue
            
            trends['historical'] = {
                'dates': dates,
                'quality_scores': quality_scores,
                'issue_counts': issue_counts
            }
        
        return {
            'trends': trends,
            'current': current,
            'history': metrics_data
        }
    
    def _generate_html_dashboard(self, metrics_data: List[Dict], trend_data: Dict) -> str:
        """Generate the HTML dashboard"""
        
        current = trend_data['current']
        trends = trend_data['trends']
        
        # Dashboard header
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Knowledge - Markdown Quality Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #334155;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            color: #1e293b;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #64748b;
            font-size: 1.1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }}
        
        .metric-card.quality::before {{ background: linear-gradient(90deg, #10b981, #059669); }}
        .metric-card.files::before {{ background: linear-gradient(90deg, #3b82f6, #1d4ed8); }}
        .metric-card.issues::before {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
        .metric-card.critical::before {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        
        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .metric-title {{
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
        }}
        
        .metric-trend {{
            display: flex;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .trend-up {{ color: #10b981; }}
        .trend-down {{ color: #ef4444; }}
        .trend-stable {{ color: #64748b; }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 5px;
        }}
        
        .metric-subtitle {{
            color: #64748b;
            font-size: 0.9rem;
        }}
        
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
        }}
        
        .issues-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .issue-type {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .issue-type.error {{
            background: #fef2f2;
            border: 1px solid #fecaca;
        }}
        
        .issue-type.warning {{
            background: #fffbeb;
            border: 1px solid #fed7aa;
        }}
        
        .issue-type.info {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
        }}
        
        .issue-count {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .issue-type.error .issue-count {{ color: #dc2626; }}
        .issue-type.warning .issue-count {{ color: #d97706; }}
        .issue-type.info .issue-count {{ color: #1d4ed8; }}
        
        .issue-label {{
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .top-issues {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .top-issues h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
        }}
        
        .issue-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .issue-item:last-child {{
            border-bottom: none;
        }}
        
        .issue-rule {{
            font-family: monospace;
            background: #f1f5f9;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #475569;
        }}
        
        .issue-frequency {{
            font-weight: 600;
            color: #1e293b;
        }}
        
        .footer {{
            text-align: center;
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 40px;
            padding: 20px;
        }}
        
        .arrow {{
            margin-left: 5px;
        }}
        
        .arrow-up::after {{ content: "↗"; }}
        .arrow-down::after {{ content: "↘"; }}
        .arrow-stable::after {{ content: "→"; }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Markdown Quality Dashboard</h1>
            <div class="subtitle">AI Knowledge Website Content Quality Metrics</div>
        </div>
        """
        
        # Current metrics cards
        html += self._generate_metrics_cards(current, trends)
        
        # Issues breakdown
        if current.get('issues_by_severity'):
            html += self._generate_issues_breakdown(current['issues_by_severity'])
        
        # Top issues
        if current.get('top_rules'):
            html += self._generate_top_issues(current['top_rules'])
        
        # Footer
        last_updated = current.get('timestamp', 'Unknown')
        try:
            last_updated = datetime.fromisoformat(last_updated).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
            
        html += f"""
        <div class="footer">
            Last updated: {last_updated} | 
            Total data points: {len(metrics_data)} | 
            Generated by AI Knowledge Quality System
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_metrics_cards(self, current: Dict, trends: Dict) -> str:
        """Generate the main metrics cards"""
        
        quality_score = current.get('quality_score', 0)
        total_files = current.get('total_files', 0)
        total_issues = current.get('total_issues', 0)
        critical_issues = current.get('issues_by_severity', {}).get('error', 0)
        
        # Trend indicators
        quality_trend = trends.get('quality_score', {})
        files_trend = trends.get('total_files', {})
        issues_trend = trends.get('total_issues', {})
        
        def format_trend(trend_data):
            if not trend_data:
                return ""
            direction = trend_data.get('direction', 'stable')
            change = trend_data.get('change', 0)
            if direction == 'stable':
                return f'<span class="metric-trend trend-stable">Stable <span class="arrow arrow-stable"></span></span>'
            elif direction == 'up':
                return f'<span class="metric-trend trend-up">+{change:.1f} <span class="arrow arrow-up"></span></span>'
            else:
                return f'<span class="metric-trend trend-down">{change:.1f} <span class="arrow arrow-down"></span></span>'
        
        return f"""
        <div class="metrics-grid">
            <div class="metric-card quality">
                <div class="metric-header">
                    <div class="metric-title">Quality Score</div>
                    {format_trend(quality_trend)}
                </div>
                <div class="metric-value">{quality_score:.1f}</div>
                <div class="metric-subtitle">out of 100</div>
            </div>
            
            <div class="metric-card files">
                <div class="metric-header">
                    <div class="metric-title">Total Files</div>
                    {format_trend(files_trend)}
                </div>
                <div class="metric-value">{total_files}</div>
                <div class="metric-subtitle">markdown files</div>
            </div>
            
            <div class="metric-card issues">
                <div class="metric-header">
                    <div class="metric-title">Total Issues</div>
                    {format_trend(issues_trend)}
                </div>
                <div class="metric-value">{total_issues}</div>
                <div class="metric-subtitle">all severities</div>
            </div>
            
            <div class="metric-card critical">
                <div class="metric-header">
                    <div class="metric-title">Critical Issues</div>
                </div>
                <div class="metric-value">{critical_issues}</div>
                <div class="metric-subtitle">must fix</div>
            </div>
        </div>
        """
    
    def _generate_issues_breakdown(self, issues_by_severity: Dict) -> str:
        """Generate issues breakdown chart"""
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Issues Breakdown by Severity</div>
            <div class="issues-breakdown">
                <div class="issue-type error">
                    <div class="issue-count">{issues_by_severity.get('error', 0)}</div>
                    <div class="issue-label">Errors</div>
                </div>
                <div class="issue-type warning">
                    <div class="issue-count">{issues_by_severity.get('warning', 0)}</div>
                    <div class="issue-label">Warnings</div>
                </div>
                <div class="issue-type info">
                    <div class="issue-count">{issues_by_severity.get('info', 0)}</div>
                    <div class="issue-label">Info</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_top_issues(self, top_rules: List) -> str:
        """Generate top issues list"""
        
        html = """
        <div class="top-issues">
            <h3>Most Common Issues</h3>
        """
        
        for rule, count in top_rules[:10]:  # Top 10
            html += f"""
            <div class="issue-item">
                <div class="issue-rule">{rule}</div>
                <div class="issue-frequency">{count}</div>
            </div>
            """
        
        html += "</div>"
        return html


def main():
    parser = argparse.ArgumentParser(description='Generate Quality Dashboard')
    parser.add_argument('--data-dir', default='quality-metrics', help='Directory containing quality metrics')
    parser.add_argument('--output', default='quality-dashboard.html', help='Output HTML file')
    parser.add_argument('--days', type=int, default=30, help='Days of history to include')
    
    args = parser.parse_args()
    
    dashboard = QualityDashboard(args.data_dir, args.days)
    output_file = dashboard.generate_dashboard(args.output)
    
    print(f"✅ Quality dashboard generated: {output_file}")
    print(f"📊 Data directory: {args.data_dir}")
    print(f"📅 History: {args.days} days")


if __name__ == '__main__':
    main()