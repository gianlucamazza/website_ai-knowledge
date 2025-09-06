"""
Content Sanitization Module

Provides comprehensive HTML and Markdown sanitization to prevent XSS attacks,
content injection, and other malicious content processing vulnerabilities.
"""

import re
import html
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import bleach
from bleach import linkifier
from markdown import markdown
from bs4 import BeautifulSoup, Comment
import validators


logger = logging.getLogger(__name__)


@dataclass
class SanitizationConfig:
    """Configuration for content sanitization."""
    
    # Allowed HTML tags
    allowed_tags: Set[str] = field(default_factory=lambda: {
        'a', 'abbr', 'acronym', 'b', 'blockquote', 'br', 'code', 'dd', 'dl', 'dt',
        'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'ol',
        'p', 'pre', 'strong', 'sub', 'sup', 'table', 'tbody', 'td', 'th', 'thead',
        'tr', 'ul', 'span', 'div'
    })
    
    # Allowed attributes per tag
    allowed_attributes: Dict[str, List[str]] = field(default_factory=lambda: {
        'a': ['href', 'title', 'rel', 'target'],
        'img': ['src', 'alt', 'width', 'height', 'title'],
        'abbr': ['title'],
        'acronym': ['title'],
        'span': ['class'],
        'div': ['class'],
        'table': ['class'],
        'td': ['colspan', 'rowspan'],
        'th': ['colspan', 'rowspan']
    })
    
    # Allowed protocols for links
    allowed_protocols: Set[str] = field(default_factory=lambda: {'http', 'https', 'mailto'})
    
    # Allowed domains for external links (empty means allow all)
    allowed_domains: Set[str] = field(default_factory=set)
    
    # Maximum content length
    max_content_length: int = 1_000_000  # 1MB
    
    # Whether to remove comments
    strip_comments: bool = True
    
    # Whether to validate URLs
    validate_urls: bool = True
    
    # Whether to add rel="nofollow" to external links
    nofollow_external: bool = True
    
    # Maximum URL length
    max_url_length: int = 2048
    
    # Blocked content patterns (regex)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'javascript:',
        r'vbscript:',
        r'data:(?!image)',  # Allow data: URLs for images only
        r'on\w+=',  # Event handlers
        r'<script',
        r'</script',
        r'<iframe',
        r'<object',
        r'<embed',
        r'<link',
        r'<meta'
    ])


class ContentSanitizer:
    """
    Comprehensive content sanitizer for HTML and Markdown content.
    
    Provides defense against:
    - XSS attacks
    - Content injection
    - Malicious links
    - Unsafe HTML elements
    - Script injection
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()
        self._setup_bleach()
        self._compile_patterns()
        
    def _setup_bleach(self) -> None:
        """Setup bleach cleaner with configuration."""
        self.cleaner = bleach.Cleaner(
            tags=self.config.allowed_tags,
            attributes=self.config.allowed_attributes,
            protocols=self.config.allowed_protocols,
            strip=True,
            strip_comments=self.config.strip_comments
        )
        
    def _compile_patterns(self) -> None:
        """Compile regex patterns for content blocking."""
        self.blocked_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.blocked_patterns
        ]
        
    def sanitize_html(self, content: str) -> str:
        """
        Sanitize HTML content.
        
        Args:
            content: Raw HTML content to sanitize
            
        Returns:
            Sanitized HTML content
            
        Raises:
            ValueError: If content is too long or contains blocked patterns
        """
        if not content:
            return ""
            
        # Check content length
        if len(content) > self.config.max_content_length:
            raise ValueError(f"Content exceeds maximum length of {self.config.max_content_length}")
            
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                logger.warning(f"Blocked pattern detected: {pattern.pattern}")
                raise ValueError(f"Content contains blocked pattern: {pattern.pattern}")
                
        # HTML decode first to handle encoded malicious content
        content = html.unescape(content)
        
        # Use BeautifulSoup for initial parsing and comment removal
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove comments
        if self.config.strip_comments:
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
                
        # Convert back to string for bleach processing
        content = str(soup)
        
        # Sanitize with bleach
        sanitized = self.cleaner.clean(content)
        
        # Additional URL validation
        if self.config.validate_urls:
            sanitized = self._validate_and_fix_urls(sanitized)
            
        return sanitized.strip()
        
    def sanitize_markdown(self, content: str) -> str:
        """
        Sanitize Markdown content by converting to HTML and sanitizing.
        
        Args:
            content: Raw Markdown content to sanitize
            
        Returns:
            Sanitized HTML content
        """
        if not content:
            return ""
            
        # Convert Markdown to HTML
        html_content = markdown(
            content,
            extensions=['fenced_code', 'tables', 'toc'],
            output_format='html5'
        )
        
        # Sanitize the resulting HTML
        return self.sanitize_html(html_content)
        
    def _validate_and_fix_urls(self, content: str) -> str:
        """
        Validate and fix URLs in content.
        
        Args:
            content: HTML content with potential URLs
            
        Returns:
            Content with validated and fixed URLs
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Process all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip if URL is too long
            if len(href) > self.config.max_url_length:
                logger.warning(f"URL too long, removing: {href[:100]}...")
                link.decompose()
                continue
                
            # Validate URL format
            if not self._is_valid_url(href):
                logger.warning(f"Invalid URL format, removing: {href}")
                link.decompose()
                continue
                
            # Check domain restrictions
            if self.config.allowed_domains and not self._is_domain_allowed(href):
                logger.warning(f"Domain not allowed, removing: {href}")
                link.decompose()
                continue
                
            # Add security attributes for external links
            if self._is_external_url(href):
                if self.config.nofollow_external:
                    rel = link.get('rel', [])
                    if isinstance(rel, str):
                        rel = rel.split()
                    if 'nofollow' not in rel:
                        rel.append('nofollow')
                    link['rel'] = ' '.join(rel)
                    
                # Add target="_blank" and security attributes
                link['target'] = '_blank'
                link['rel'] = link.get('rel', '') + ' noopener noreferrer'
                
        # Process all images
        for img in soup.find_all('img', src=True):
            src = img['src']
            
            # Skip if URL is too long
            if len(src) > self.config.max_url_length:
                logger.warning(f"Image URL too long, removing: {src[:100]}...")
                img.decompose()
                continue
                
            # Validate image URL
            if not self._is_valid_image_url(src):
                logger.warning(f"Invalid image URL, removing: {src}")
                img.decompose()
                continue
                
            # Add loading="lazy" for performance
            img['loading'] = 'lazy'
            
        return str(soup)
        
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and uses allowed protocol."""
        try:
            parsed = urlparse(url)
            
            # Check protocol
            if parsed.scheme and parsed.scheme not in self.config.allowed_protocols:
                return False
                
            # Use validators library for comprehensive validation
            return validators.url(url) if parsed.scheme else True  # Allow relative URLs
            
        except Exception:
            return False
            
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if image URL is valid."""
        if not self._is_valid_url(url):
            return False
            
        # Allow data URLs for images
        if url.startswith('data:image/'):
            # Basic validation of data URL format
            return ',' in url and len(url) < 100000  # 100KB limit for data URLs
            
        return True
        
    def _is_domain_allowed(self, url: str) -> bool:
        """Check if URL domain is in allowed list."""
        if not self.config.allowed_domains:
            return True  # No restrictions
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain in self.config.allowed_domains
            
        except Exception:
            return False
            
    def _is_external_url(self, url: str) -> bool:
        """Check if URL is external (has a domain)."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc)
        except Exception:
            return False
            
    def validate_content_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize content metadata (frontmatter).
        
        Args:
            metadata: Dictionary of metadata fields
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized_metadata = {}
        
        for key, value in metadata.items():
            # Sanitize string values
            if isinstance(value, str):
                # Check for suspicious content in metadata
                for pattern in self.blocked_patterns:
                    if pattern.search(value):
                        logger.warning(f"Blocked pattern in metadata {key}: {pattern.pattern}")
                        continue
                        
                # HTML escape the value
                sanitized_metadata[key] = html.escape(value.strip())
                
            # Handle lists
            elif isinstance(value, list):
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_item = html.escape(item.strip())
                        if sanitized_item:
                            sanitized_list.append(sanitized_item)
                            
                sanitized_metadata[key] = sanitized_list
                
            # Handle other types (numbers, booleans)
            else:
                sanitized_metadata[key] = value
                
        return sanitized_metadata
        
    def get_security_report(self, content: str) -> Dict[str, Any]:
        """
        Generate a security report for content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Security analysis report
        """
        report = {
            'content_length': len(content),
            'blocked_patterns_found': [],
            'suspicious_elements': [],
            'external_links': 0,
            'images': 0,
            'security_score': 100  # Start with perfect score
        }
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            matches = pattern.findall(content)
            if matches:
                report['blocked_patterns_found'].append({
                    'pattern': pattern.pattern,
                    'matches': len(matches)
                })
                report['security_score'] -= 20
                
        # Parse with BeautifulSoup for element analysis
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Count external links
            for link in soup.find_all('a', href=True):
                if self._is_external_url(link['href']):
                    report['external_links'] += 1
                    
            # Count images
            report['images'] = len(soup.find_all('img'))
            
            # Check for suspicious elements
            suspicious_tags = ['script', 'iframe', 'object', 'embed', 'link', 'meta']
            for tag in suspicious_tags:
                elements = soup.find_all(tag)
                if elements:
                    report['suspicious_elements'].append({
                        'tag': tag,
                        'count': len(elements)
                    })
                    report['security_score'] -= 10
                    
        except Exception as e:
            logger.warning(f"Error parsing content for security report: {e}")
            report['security_score'] -= 5
            
        # Ensure security score doesn't go below 0
        report['security_score'] = max(0, report['security_score'])
        
        return report


# Default instance for easy usage
default_sanitizer = ContentSanitizer()


def sanitize_html(content: str) -> str:
    """Quick function for HTML sanitization using default config."""
    return default_sanitizer.sanitize_html(content)


def sanitize_markdown(content: str) -> str:
    """Quick function for Markdown sanitization using default config."""
    return default_sanitizer.sanitize_markdown(content)