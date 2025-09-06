"""
Compliance and Copyright Monitoring System

Provides comprehensive compliance checking for content including:
- Copyright and license compliance
- GDPR and privacy compliance
- Content attribution and source tracking
- Ethical AI content usage monitoring
"""

import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import yaml


logger = logging.getLogger(__name__)


@dataclass
class ComplianceConfig:
    """Configuration for compliance checking."""
    
    # Copyright checking
    check_copyright: bool = True
    copyright_patterns_file: str = "security/copyright_patterns.yaml"
    require_attribution: bool = True
    
    # License checking
    check_licenses: bool = True
    allowed_licenses: Set[str] = field(default_factory=lambda: {
        'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 
        'GPL-3.0', 'LGPL-3.0', 'CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0'
    })
    prohibited_licenses: Set[str] = field(default_factory=lambda: {
        'GPL-2.0', 'AGPL-3.0'  # Example of restrictive licenses
    })
    
    # GDPR compliance
    gdpr_compliance_enabled: bool = True
    data_retention_days: int = 365
    require_consent_tracking: bool = True
    
    # Content source tracking
    track_content_sources: bool = True
    require_source_attribution: bool = True
    max_content_age_days: int = 30
    
    # Robots.txt compliance
    respect_robots_txt: bool = True
    user_agent: str = "AI-Knowledge-Bot/1.0"
    
    # Ethical AI guidelines
    ethical_ai_enabled: bool = True
    prohibited_content_types: Set[str] = field(default_factory=lambda: {
        'personal_data', 'private_communications', 'confidential_documents'
    })
    
    # Rate limiting for compliance checks
    max_requests_per_minute: int = 30
    request_delay: float = 2.0
    
    # Reporting
    compliance_report_path: str = ".security/compliance_report.json"
    violations_log_path: str = ".security/compliance_violations.log"


@dataclass
class ContentSource:
    """Information about content source."""
    
    url: str
    domain: str
    title: str = ""
    author: str = ""
    publication_date: Optional[datetime] = None
    scraped_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    license: Optional[str] = None
    copyright_notice: Optional[str] = None
    robots_txt_allowed: bool = True
    attribution_required: bool = False
    content_type: str = "article"
    
    
@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    
    violation_type: str  # COPYRIGHT, LICENSE, GDPR, ROBOTS, ETHICAL
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    source_url: Optional[str] = None
    content_id: Optional[str] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_status: str = "OPEN"  # OPEN, ACKNOWLEDGED, RESOLVED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GDPRDataRecord:
    """GDPR data processing record."""
    
    data_type: str  # personal_data, usage_analytics, content_metadata
    purpose: str
    legal_basis: str  # consent, legitimate_interest, contract, legal_obligation
    data_subject_category: str  # website_visitors, content_authors, subscribers
    retention_period: int  # days
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consent_obtained: bool = False
    consent_timestamp: Optional[datetime] = None


class ComplianceChecker:
    """
    Comprehensive compliance monitoring system.
    
    Features:
    - Copyright and license detection
    - GDPR compliance tracking
    - Content attribution monitoring
    - Robots.txt compliance
    - Ethical AI content usage
    - Violation detection and reporting
    """
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self._load_copyright_patterns()
        self._setup_violation_logging()
        
        # Tracking stores
        self.content_sources: Dict[str, ContentSource] = {}
        self.violations: List[ComplianceViolation] = []
        self.gdpr_records: List[GDPRDataRecord] = []
        
    def _load_copyright_patterns(self) -> None:
        """Load copyright detection patterns."""
        patterns_file = Path(self.config.copyright_patterns_file)
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = yaml.safe_load(f)
                    
                self.copyright_patterns = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    for pattern in patterns_data.get('copyright_patterns', [])
                ]
                
                self.license_patterns = {
                    license_name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    for license_name, pattern in patterns_data.get('license_patterns', {}).items()
                }
                
            except Exception as e:
                logger.error(f"Failed to load copyright patterns: {e}")
                self._create_default_patterns()
        else:
            self._create_default_patterns()
            
    def _create_default_patterns(self) -> None:
        """Create default copyright and license patterns."""
        self.copyright_patterns = [
            re.compile(r'©\s*(\d{4})', re.IGNORECASE),
            re.compile(r'Copyright\s*©?\s*(\d{4})', re.IGNORECASE),
            re.compile(r'\(c\)\s*(\d{4})', re.IGNORECASE),
            re.compile(r'All rights reserved', re.IGNORECASE)
        ]
        
        self.license_patterns = {
            'MIT': re.compile(r'MIT License|MIT licence', re.IGNORECASE),
            'Apache-2.0': re.compile(r'Apache License[,\s]*Version 2\.0', re.IGNORECASE),
            'GPL-3.0': re.compile(r'GNU General Public License[,\s]*version 3', re.IGNORECASE),
            'BSD-3-Clause': re.compile(r'BSD 3-Clause|New BSD License', re.IGNORECASE),
            'CC-BY-4.0': re.compile(r'Creative Commons Attribution 4\.0', re.IGNORECASE)
        }
        
    def _setup_violation_logging(self) -> None:
        """Setup logging for compliance violations."""
        violations_file = Path(self.config.violations_log_path)
        violations_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.violations_logger = logging.getLogger("compliance_violations")
        self.violations_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(violations_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.violations_logger.addHandler(handler)
        
    def check_content_compliance(self, 
                                content: str, 
                                source_url: str,
                                content_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check content for compliance violations.
        
        Args:
            content: Content text to check
            source_url: URL where content was sourced from
            content_id: Optional content identifier
            
        Returns:
            Compliance check results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'source_info': {},
            'recommendations': []
        }
        
        try:
            # Get source information
            source_info = self._analyze_content_source(source_url)
            results['source_info'] = self._source_to_dict(source_info)
            
            # Copyright compliance check
            if self.config.check_copyright:
                copyright_results = self._check_copyright_compliance(content, source_info)
                results['violations'].extend(copyright_results['violations'])
                results['warnings'].extend(copyright_results['warnings'])
                
            # License compliance check
            if self.config.check_licenses:
                license_results = self._check_license_compliance(content, source_info)
                results['violations'].extend(license_results['violations'])
                results['warnings'].extend(license_results['warnings'])
                
            # Robots.txt compliance check
            if self.config.respect_robots_txt:
                robots_results = self._check_robots_compliance(source_url)
                if not robots_results['allowed']:
                    results['violations'].append(robots_results['violation'])
                    
            # Ethical AI compliance check
            if self.config.ethical_ai_enabled:
                ethical_results = self._check_ethical_compliance(content, source_info)
                results['violations'].extend(ethical_results['violations'])
                results['warnings'].extend(ethical_results['warnings'])
                
            # Determine overall compliance
            results['compliant'] = len(results['violations']) == 0
            
            # Log violations
            for violation in results['violations']:
                self._log_violation(violation, content_id)
                
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"Error checking compliance for {source_url}: {e}")
            results['error'] = str(e)
            
        return results
        
    def _analyze_content_source(self, url: str) -> ContentSource:
        """Analyze content source for compliance information."""
        if url in self.content_sources:
            return self.content_sources[url]
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        source = ContentSource(
            url=url,
            domain=domain
        )
        
        try:
            # Fetch page content for analysis
            headers = {'User-Agent': self.config.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                source.title = title_tag.get_text().strip()
                
            # Look for author information
            author_selectors = [
                'meta[name="author"]',
                '.author',
                '[rel="author"]',
                '.byline'
            ]
            
            for selector in author_selectors:
                author_element = soup.select_one(selector)
                if author_element:
                    content_attr = author_element.get('content')
                    source.author = content_attr if content_attr else author_element.get_text().strip()
                    break
                    
            # Look for publication date
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="date"]',
                'time[datetime]',
                '.date'
            ]
            
            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    date_str = date_element.get('content') or date_element.get('datetime') or date_element.get_text()
                    try:
                        source.publication_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        break
                    except (ValueError, AttributeError):
                        continue
                        
            # Detect copyright and license information
            page_text = soup.get_text()
            
            # Check for copyright notices
            for pattern in self.copyright_patterns:
                match = pattern.search(page_text)
                if match:
                    source.copyright_notice = match.group(0)
                    source.attribution_required = True
                    break
                    
            # Check for license information
            for license_name, pattern in self.license_patterns.items():
                if pattern.search(page_text):
                    source.license = license_name
                    break
                    
            # Check for Creative Commons licenses in links
            cc_links = soup.find_all('a', href=re.compile(r'creativecommons\.org/licenses'))
            if cc_links:
                cc_url = cc_links[0].get('href')
                cc_match = re.search(r'/([^/]+)/\d', cc_url)
                if cc_match:
                    source.license = f"CC-{cc_match.group(1).upper()}"
                    
        except Exception as e:
            logger.warning(f"Failed to analyze source {url}: {e}")
            
        # Cache the source information
        self.content_sources[url] = source
        
        return source
        
    def _check_copyright_compliance(self, 
                                  content: str, 
                                  source: ContentSource) -> Dict[str, List]:
        """Check copyright compliance."""
        results = {'violations': [], 'warnings': []}
        
        # Check if content has copyright notice and requires attribution
        if source.copyright_notice and source.attribution_required:
            # Look for attribution in content
            attribution_found = False
            attribution_patterns = [
                re.compile(rf'{re.escape(source.author)}', re.IGNORECASE),
                re.compile(rf'{re.escape(source.domain)}', re.IGNORECASE),
                re.compile(rf'{re.escape(source.title)}', re.IGNORECASE)
            ]
            
            for pattern in attribution_patterns:
                if pattern.search(content):
                    attribution_found = True
                    break
                    
            if not attribution_found:
                violation = ComplianceViolation(
                    violation_type="COPYRIGHT",
                    severity="HIGH",
                    description=f"Content from {source.domain} requires attribution but none found",
                    source_url=source.url,
                    metadata={'copyright_notice': source.copyright_notice}
                )
                results['violations'].append(violation)
                
        # Check for potential copyright infringement patterns
        copyright_indicators = [
            r'All rights reserved',
            r'Proprietary and confidential',
            r'Unauthorized reproduction',
            r'©.*(?:Inc\.|Ltd\.|Corp\.|Company)'
        ]
        
        for pattern_str in copyright_indicators:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(content):
                warning = f"Content contains potential copyright-protected material: {pattern_str}"
                results['warnings'].append(warning)
                
        return results
        
    def _check_license_compliance(self, 
                                content: str, 
                                source: ContentSource) -> Dict[str, List]:
        """Check license compliance."""
        results = {'violations': [], 'warnings': []}
        
        if not source.license:
            results['warnings'].append("No license information found for source content")
            return results
            
        # Check if license is allowed
        if source.license in self.config.prohibited_licenses:
            violation = ComplianceViolation(
                violation_type="LICENSE",
                severity="CRITICAL",
                description=f"Content uses prohibited license: {source.license}",
                source_url=source.url,
                metadata={'license': source.license}
            )
            results['violations'].append(violation)
            
        elif source.license not in self.config.allowed_licenses:
            violation = ComplianceViolation(
                violation_type="LICENSE",
                severity="MEDIUM",
                description=f"Content uses unrecognized license: {source.license}",
                source_url=source.url,
                metadata={'license': source.license}
            )
            results['violations'].append(violation)
            
        # Check license-specific requirements
        if source.license and 'CC-BY' in source.license:
            # Creative Commons Attribution required
            if not self._has_attribution(content, source):
                violation = ComplianceViolation(
                    violation_type="LICENSE",
                    severity="HIGH",
                    description=f"Creative Commons license requires attribution",
                    source_url=source.url,
                    metadata={'license': source.license}
                )
                results['violations'].append(violation)
                
        return results
        
    def _check_robots_compliance(self, url: str) -> Dict[str, Any]:
        """Check robots.txt compliance."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = requests.get(robots_url, timeout=5)
            if response.status_code == 200:
                robots_content = response.text
                
                # Parse robots.txt (simplified)
                user_agent_blocks = self._parse_robots_txt(robots_content)
                
                # Check if our user agent is disallowed
                for block in user_agent_blocks:
                    if (block['user_agent'] == '*' or 
                        self.config.user_agent.lower() in block['user_agent'].lower()):
                        
                        for disallow_path in block['disallowed_paths']:
                            if parsed_url.path.startswith(disallow_path):
                                violation = ComplianceViolation(
                                    violation_type="ROBOTS",
                                    severity="HIGH",
                                    description=f"URL disallowed by robots.txt: {disallow_path}",
                                    source_url=url,
                                    metadata={'robots_url': robots_url}
                                )
                                return {'allowed': False, 'violation': violation}
                                
            return {'allowed': True}
            
        except Exception as e:
            logger.warning(f"Failed to check robots.txt for {url}: {e}")
            return {'allowed': True}  # Default to allowed if can't check
            
    def _check_ethical_compliance(self, 
                                content: str, 
                                source: ContentSource) -> Dict[str, List]:
        """Check ethical AI compliance."""
        results = {'violations': [], 'warnings': []}
        
        # Check for prohibited content types
        content_lower = content.lower()
        
        ethical_concerns = [
            {
                'pattern': r'\b(email|e-mail)\s*:\s*\S+@\S+',
                'type': 'personal_data',
                'description': 'Content contains email addresses'
            },
            {
                'pattern': r'\b\d{3}-?\d{2}-?\d{4}\b',
                'type': 'personal_data',
                'description': 'Content may contain SSN or similar sensitive numbers'
            },
            {
                'pattern': r'\bphone\s*(?:number)?:?\s*[\+\d\s\-\(\)]+',
                'type': 'personal_data',
                'description': 'Content contains phone numbers'
            },
            {
                'pattern': r'\b(?:confidential|proprietary|internal only|do not distribute)\b',
                'type': 'confidential_documents',
                'description': 'Content marked as confidential'
            }
        ]
        
        for concern in ethical_concerns:
            pattern = re.compile(concern['pattern'], re.IGNORECASE)
            matches = pattern.findall(content)
            
            if matches and concern['type'] in self.config.prohibited_content_types:
                violation = ComplianceViolation(
                    violation_type="ETHICAL",
                    severity="HIGH",
                    description=concern['description'],
                    source_url=source.url,
                    metadata={
                        'content_type': concern['type'],
                        'matches_count': len(matches)
                    }
                )
                results['violations'].append(violation)
                
        return results
        
    def _has_attribution(self, content: str, source: ContentSource) -> bool:
        """Check if content has proper attribution."""
        attribution_elements = [
            source.author,
            source.domain,
            source.title,
            source.url
        ]
        
        content_lower = content.lower()
        
        for element in attribution_elements:
            if element and element.lower() in content_lower:
                return True
                
        return False
        
    def _parse_robots_txt(self, robots_content: str) -> List[Dict]:
        """Parse robots.txt content (simplified parser)."""
        blocks = []
        current_block = None
        
        for line in robots_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.lower().startswith('user-agent:'):
                if current_block:
                    blocks.append(current_block)
                current_block = {
                    'user_agent': line.split(':', 1)[1].strip(),
                    'disallowed_paths': []
                }
            elif line.lower().startswith('disallow:') and current_block:
                path = line.split(':', 1)[1].strip()
                if path:
                    current_block['disallowed_paths'].append(path)
                    
        if current_block:
            blocks.append(current_block)
            
        return blocks
        
    def _source_to_dict(self, source: ContentSource) -> Dict[str, Any]:
        """Convert ContentSource to dictionary."""
        return {
            'url': source.url,
            'domain': source.domain,
            'title': source.title,
            'author': source.author,
            'publication_date': source.publication_date.isoformat() if source.publication_date else None,
            'scraped_date': source.scraped_date.isoformat(),
            'license': source.license,
            'copyright_notice': source.copyright_notice,
            'attribution_required': source.attribution_required,
            'content_type': source.content_type
        }
        
    def _log_violation(self, violation: ComplianceViolation, content_id: Optional[str]) -> None:
        """Log compliance violation."""
        violation_data = {
            'violation_type': violation.violation_type,
            'severity': violation.severity,
            'description': violation.description,
            'source_url': violation.source_url,
            'content_id': content_id,
            'detected_at': violation.detected_at.isoformat(),
            'metadata': violation.metadata
        }
        
        self.violations_logger.warning(json.dumps(violation_data))
        self.violations.append(violation)
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        for violation in results['violations']:
            if violation.violation_type == "COPYRIGHT":
                recommendations.append("Add proper attribution to the content")
                recommendations.append("Consider reaching out to original author for permission")
                
            elif violation.violation_type == "LICENSE":
                recommendations.append("Review license terms and ensure compliance")
                recommendations.append("Consider finding content with compatible license")
                
            elif violation.violation_type == "ROBOTS":
                recommendations.append("Remove content that violates robots.txt")
                recommendations.append("Request permission from site owner")
                
            elif violation.violation_type == "ETHICAL":
                recommendations.append("Remove or anonymize personal data")
                recommendations.append("Review content for ethical AI guidelines")
                
        # Remove duplicates
        return list(set(recommendations))
        
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        now = datetime.now(timezone.utc)
        
        # Count violations by type and severity
        violations_by_type = {}
        violations_by_severity = {}
        
        for violation in self.violations:
            violations_by_type[violation.violation_type] = violations_by_type.get(violation.violation_type, 0) + 1
            violations_by_severity[violation.severity] = violations_by_severity.get(violation.severity, 0) + 1
            
        # Calculate compliance score
        total_checks = len(self.content_sources)
        total_violations = len(self.violations)
        compliance_score = max(0, 100 - (total_violations / max(1, total_checks) * 100))
        
        report = {
            'report_generated_at': now.isoformat(),
            'summary': {
                'total_sources_checked': total_checks,
                'total_violations': total_violations,
                'compliance_score': round(compliance_score, 2),
                'violations_by_type': violations_by_type,
                'violations_by_severity': violations_by_severity
            },
            'recent_violations': [
                {
                    'type': v.violation_type,
                    'severity': v.severity,
                    'description': v.description,
                    'source_url': v.source_url,
                    'detected_at': v.detected_at.isoformat()
                }
                for v in sorted(self.violations, key=lambda x: x.detected_at, reverse=True)[:10]
            ],
            'sources_summary': [
                {
                    'url': source.url,
                    'domain': source.domain,
                    'license': source.license,
                    'attribution_required': source.attribution_required,
                    'scraped_date': source.scraped_date.isoformat()
                }
                for source in self.content_sources.values()
            ]
        }
        
        return report
        
    def save_compliance_report(self) -> None:
        """Save compliance report to file."""
        report = self.get_compliance_report()
        
        report_file = Path(self.config.compliance_report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Compliance report saved to {report_file}")
        
    def create_gdpr_record(self, 
                          data_type: str,
                          purpose: str,
                          legal_basis: str,
                          data_subject_category: str,
                          retention_days: int,
                          consent_obtained: bool = False) -> GDPRDataRecord:
        """Create GDPR data processing record."""
        record = GDPRDataRecord(
            data_type=data_type,
            purpose=purpose,
            legal_basis=legal_basis,
            data_subject_category=data_subject_category,
            retention_period=retention_days,
            consent_obtained=consent_obtained,
            consent_timestamp=datetime.now(timezone.utc) if consent_obtained else None
        )
        
        self.gdpr_records.append(record)
        logger.info(f"Created GDPR record for {data_type}")
        
        return record
        
    def get_gdpr_compliance_status(self) -> Dict[str, Any]:
        """Get GDPR compliance status."""
        now = datetime.now(timezone.utc)
        
        # Check for records approaching retention limit
        approaching_limit = []
        expired_records = []
        
        for record in self.gdpr_records:
            retention_end = record.created_at + timedelta(days=record.retention_period)
            days_remaining = (retention_end - now).days
            
            if days_remaining < 0:
                expired_records.append(record)
            elif days_remaining <= 30:
                approaching_limit.append({
                    'data_type': record.data_type,
                    'days_remaining': days_remaining
                })
                
        return {
            'total_records': len(self.gdpr_records),
            'records_with_consent': sum(1 for r in self.gdpr_records if r.consent_obtained),
            'records_approaching_retention_limit': approaching_limit,
            'expired_records_count': len(expired_records),
            'compliance_issues': len(expired_records) > 0
        }


# Factory function
def create_compliance_checker(config: Optional[ComplianceConfig] = None) -> ComplianceChecker:
    """Create configured compliance checker instance."""
    config = config or ComplianceConfig()
    return ComplianceChecker(config)