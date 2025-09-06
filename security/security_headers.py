"""
Security Headers Configuration

Provides comprehensive security headers configuration including CSP, HSTS,
and other security headers for protection against common web vulnerabilities.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class CSPDirective(Enum):
    """Content Security Policy directive names."""
    
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    FONT_SRC = "font-src"
    CONNECT_SRC = "connect-src"
    MEDIA_SRC = "media-src"
    OBJECT_SRC = "object-src"
    CHILD_SRC = "child-src"
    FRAME_SRC = "frame-src"
    WORKER_SRC = "worker-src"
    MANIFEST_SRC = "manifest-src"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    FRAME_ANCESTORS = "frame-ancestors"
    REPORT_URI = "report-uri"
    REPORT_TO = "report-to"


@dataclass
class HeaderConfig:
    """Configuration for security headers."""
    
    # Content Security Policy
    csp_enabled: bool = True
    csp_report_only: bool = False
    csp_report_uri: Optional[str] = "/security/csp-report"
    csp_directives: Dict[str, List[str]] = field(default_factory=lambda: {
        "default-src": ["'self'"],
        "script-src": ["'self'", "'unsafe-inline'"],  # Astro needs inline scripts
        "style-src": ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
        "img-src": ["'self'", "data:", "https:"],
        "font-src": ["'self'", "fonts.gstatic.com"],
        "connect-src": ["'self'"],
        "media-src": ["'self'"],
        "object-src": ["'none'"],
        "frame-src": ["'none'"],
        "base-uri": ["'self'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'none'"],
    })
    
    # HTTP Strict Transport Security
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    # X-Frame-Options
    x_frame_options: str = "DENY"  # DENY, SAMEORIGIN, or ALLOW-FROM
    
    # X-Content-Type-Options
    x_content_type_options: str = "nosniff"
    
    # X-XSS-Protection (deprecated but still useful for older browsers)
    x_xss_protection: str = "1; mode=block"
    
    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    # Permissions Policy (formerly Feature Policy)
    permissions_policy_enabled: bool = True
    permissions_policy_directives: Dict[str, str] = field(default_factory=lambda: {
        "geolocation": "()",
        "microphone": "()",
        "camera": "()",
        "payment": "()",
        "usb": "()",
        "magnetometer": "()",
        "gyroscope": "()",
        "accelerometer": "()",
        "fullscreen": "(self)",
    })
    
    # Cross-Origin headers
    cross_origin_embedder_policy: str = "require-corp"
    cross_origin_opener_policy: str = "same-origin"
    cross_origin_resource_policy: str = "same-site"
    
    # Cache Control for sensitive pages
    cache_control_sensitive: str = "no-cache, no-store, must-revalidate"
    pragma_sensitive: str = "no-cache"
    expires_sensitive: str = "0"
    
    # Server header obfuscation
    hide_server_header: bool = True
    custom_server_header: Optional[str] = None
    
    # Additional security headers
    expect_ct_enabled: bool = False
    expect_ct_max_age: int = 86400
    expect_ct_enforce: bool = False
    expect_ct_report_uri: Optional[str] = None
    
    # Development mode adjustments
    development_mode: bool = False


class SecurityHeaders:
    """
    Security headers management for web applications.
    
    Provides comprehensive security headers to protect against:
    - XSS attacks
    - Clickjacking
    - CSRF attacks
    - Content type sniffing
    - Man-in-the-middle attacks
    - Information disclosure
    """
    
    def __init__(self, config: HeaderConfig):
        self.config = config
        
    def get_security_headers(self, 
                           request_path: str = "",
                           is_sensitive: bool = False) -> Dict[str, str]:
        """
        Get all security headers for a request.
        
        Args:
            request_path: The request path (for path-specific headers)
            is_sensitive: Whether this is a sensitive page requiring strict caching
            
        Returns:
            Dictionary of header names to values
        """
        headers = {}
        
        # Content Security Policy
        if self.config.csp_enabled:
            csp_header = self._build_csp_header()
            if self.config.csp_report_only:
                headers["Content-Security-Policy-Report-Only"] = csp_header
            else:
                headers["Content-Security-Policy"] = csp_header
                
        # HTTP Strict Transport Security
        if self.config.hsts_enabled:
            headers["Strict-Transport-Security"] = self._build_hsts_header()
            
        # X-Frame-Options
        headers["X-Frame-Options"] = self.config.x_frame_options
        
        # X-Content-Type-Options
        headers["X-Content-Type-Options"] = self.config.x_content_type_options
        
        # X-XSS-Protection
        headers["X-XSS-Protection"] = self.config.x_xss_protection
        
        # Referrer Policy
        headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Permissions Policy
        if self.config.permissions_policy_enabled:
            headers["Permissions-Policy"] = self._build_permissions_policy_header()
            
        # Cross-Origin headers
        headers["Cross-Origin-Embedder-Policy"] = self.config.cross_origin_embedder_policy
        headers["Cross-Origin-Opener-Policy"] = self.config.cross_origin_opener_policy
        headers["Cross-Origin-Resource-Policy"] = self.config.cross_origin_resource_policy
        
        # Cache control for sensitive pages
        if is_sensitive:
            headers["Cache-Control"] = self.config.cache_control_sensitive
            headers["Pragma"] = self.config.pragma_sensitive
            headers["Expires"] = self.config.expires_sensitive
            
        # Server header obfuscation
        if self.config.hide_server_header:
            if self.config.custom_server_header:
                headers["Server"] = self.config.custom_server_header
            else:
                headers["Server"] = ""
                
        # Expect-CT
        if self.config.expect_ct_enabled:
            headers["Expect-CT"] = self._build_expect_ct_header()
            
        # Additional security headers
        headers.update(self._get_additional_headers(request_path))
        
        return headers
        
    def _build_csp_header(self) -> str:
        """Build Content Security Policy header value."""
        directives = []
        
        for directive, sources in self.config.csp_directives.items():
            if sources:
                directive_value = f"{directive} {' '.join(sources)}"
                directives.append(directive_value)
                
        csp_value = "; ".join(directives)
        
        # Add report URI if configured
        if self.config.csp_report_uri:
            csp_value += f"; report-uri {self.config.csp_report_uri}"
            
        return csp_value
        
    def _build_hsts_header(self) -> str:
        """Build HSTS header value."""
        hsts_parts = [f"max-age={self.config.hsts_max_age}"]
        
        if self.config.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")
            
        if self.config.hsts_preload:
            hsts_parts.append("preload")
            
        return "; ".join(hsts_parts)
        
    def _build_permissions_policy_header(self) -> str:
        """Build Permissions Policy header value."""
        policies = []
        
        for feature, allowlist in self.config.permissions_policy_directives.items():
            policies.append(f"{feature}={allowlist}")
            
        return ", ".join(policies)
        
    def _build_expect_ct_header(self) -> str:
        """Build Expect-CT header value."""
        expect_ct_parts = [f"max-age={self.config.expect_ct_max_age}"]
        
        if self.config.expect_ct_enforce:
            expect_ct_parts.append("enforce")
            
        if self.config.expect_ct_report_uri:
            expect_ct_parts.append(f'report-uri="{self.config.expect_ct_report_uri}"')
            
        return ", ".join(expect_ct_parts)
        
    def _get_additional_headers(self, request_path: str) -> Dict[str, str]:
        """Get additional path-specific headers."""
        headers = {}
        
        # API-specific headers
        if request_path.startswith("/api/"):
            headers["X-Robots-Tag"] = "noindex, nofollow"
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            
        # Admin-specific headers
        if "/admin" in request_path:
            headers["X-Frame-Options"] = "DENY"
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            
        return headers
        
    def validate_csp_policy(self) -> List[str]:
        """
        Validate CSP policy and return list of potential issues.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for unsafe directives
        unsafe_sources = ["'unsafe-inline'", "'unsafe-eval'"]
        for directive, sources in self.config.csp_directives.items():
            for source in sources:
                if source in unsafe_sources:
                    issues.append(f"Unsafe source '{source}' in {directive}")
                    
        # Check for overly permissive policies
        permissive_sources = ["*", "data:", "https:"]
        for directive, sources in self.config.csp_directives.items():
            if directive in ["script-src", "object-src"]:
                for source in sources:
                    if source in permissive_sources:
                        issues.append(f"Overly permissive source '{source}' in {directive}")
                        
        # Check for missing important directives
        important_directives = ["default-src", "script-src", "object-src", "base-uri"]
        for directive in important_directives:
            if directive not in self.config.csp_directives:
                issues.append(f"Missing important directive: {directive}")
                
        # Check object-src policy
        if "object-src" in self.config.csp_directives:
            object_sources = self.config.csp_directives["object-src"]
            if "'none'" not in object_sources:
                issues.append("object-src should be set to 'none' for security")
                
        # Check base-uri policy
        if "base-uri" in self.config.csp_directives:
            base_sources = self.config.csp_directives["base-uri"]
            if "'self'" not in base_sources:
                issues.append("base-uri should include 'self' or be more restrictive")
                
        return issues
        
    def get_csp_report_handler(self):
        """
        Get a function to handle CSP violation reports.
        
        Returns:
            Function that can process CSP reports
        """
        def handle_csp_report(report_data: Dict) -> None:
            """Handle CSP violation report."""
            try:
                if 'csp-report' in report_data:
                    violation = report_data['csp-report']
                    
                    # Log the violation
                    logger.warning(
                        f"CSP Violation: {violation.get('violated-directive', 'unknown')} "
                        f"blocked {violation.get('blocked-uri', 'unknown')} "
                        f"on {violation.get('document-uri', 'unknown')}"
                    )
                    
                    # Here you could store violations in a database for analysis
                    # or send alerts for repeated violations
                    
            except Exception as e:
                logger.error(f"Error processing CSP report: {e}")
                
        return handle_csp_report
        
    def adjust_for_development(self) -> None:
        """Adjust security headers for development environment."""
        if not self.config.development_mode:
            return
            
        logger.warning("Adjusting security headers for development mode")
        
        # Allow localhost and development sources
        dev_sources = ["localhost:*", "127.0.0.1:*", "'unsafe-eval'"]
        
        for directive in ["script-src", "connect-src"]:
            if directive in self.config.csp_directives:
                self.config.csp_directives[directive].extend(dev_sources)
                
        # Disable HSTS in development
        self.config.hsts_enabled = False
        
        # Allow frames for development tools
        self.config.x_frame_options = "SAMEORIGIN"
        
    def get_security_score(self) -> Dict[str, any]:
        """
        Calculate security score based on configured headers.
        
        Returns:
            Dictionary with security assessment
        """
        score = 100
        issues = []
        recommendations = []
        
        # CSP assessment
        if not self.config.csp_enabled:
            score -= 30
            issues.append("Content Security Policy is disabled")
        else:
            csp_issues = self.validate_csp_policy()
            score -= len(csp_issues) * 5
            issues.extend(csp_issues)
            
        # HSTS assessment
        if not self.config.hsts_enabled:
            score -= 20
            issues.append("HSTS is disabled")
        elif self.config.hsts_max_age < 31536000:  # 1 year
            score -= 5
            recommendations.append("Consider increasing HSTS max-age to 1 year or more")
            
        # Frame options assessment
        if self.config.x_frame_options not in ["DENY", "SAMEORIGIN"]:
            score -= 10
            issues.append("X-Frame-Options should be DENY or SAMEORIGIN")
            
        # Permissions policy assessment
        if not self.config.permissions_policy_enabled:
            score -= 10
            recommendations.append("Consider enabling Permissions Policy")
            
        # Development mode penalty
        if self.config.development_mode:
            score -= 15
            issues.append("Development mode reduces security")
            
        return {
            "score": max(0, score),
            "grade": self._get_grade(max(0, score)),
            "issues": issues,
            "recommendations": recommendations,
            "total_headers": len(self.get_security_headers())
        }
        
    def _get_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


# Default configurations for different environments
def get_default_config(environment: str = "production") -> HeaderConfig:
    """Get default security header configuration for environment."""
    config = HeaderConfig()
    
    if environment == "development":
        config.development_mode = True
        config.hsts_enabled = False
        config.csp_report_only = True
        
        # More permissive CSP for development
        config.csp_directives["script-src"].extend([
            "localhost:*", 
            "127.0.0.1:*",
            "'unsafe-eval'"  # For dev tools
        ])
        config.csp_directives["connect-src"].extend([
            "localhost:*",
            "127.0.0.1:*",
            "ws://localhost:*"  # WebSocket for hot reload
        ])
        
    elif environment == "staging":
        config.csp_report_only = True  # Test CSP in report-only mode
        config.hsts_max_age = 3600  # Shorter HSTS for testing
        
    # Production uses default strict configuration
    
    return config


# Utility functions
def create_security_headers(environment: str = "production") -> SecurityHeaders:
    """Create SecurityHeaders instance with environment-appropriate config."""
    config = get_default_config(environment)
    return SecurityHeaders(config)