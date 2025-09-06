"""
Input Validation Framework

Provides comprehensive input validation for all user inputs and external content
to prevent injection attacks, data corruption, and security vulnerabilities.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, date
from urllib.parse import urlparse
import validators
from email_validator import validate_email, EmailNotValidError


logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Individual validation rule."""
    
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    
    # Maximum string lengths
    max_string_length: int = 10000
    max_url_length: int = 2048
    max_email_length: int = 254
    max_filename_length: int = 255
    
    # Allowed file extensions
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.txt', '.md', '.json', '.csv', '.xml', '.yaml', '.yml'
    })
    
    # Blocked patterns in strings
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'vbscript:',    # VBScript URLs
        r'on\w+\s*=',    # Event handlers
        r'eval\s*\(',    # eval() calls
        r'exec\s*\(',    # exec() calls
        r'\bFROM\s+INFORMATION_SCHEMA\b',  # SQL injection pattern
        r'\bUNION\s+SELECT\b',  # SQL injection pattern
        r'\bDROP\s+TABLE\b',    # SQL injection pattern
        r'\.\.\/|\.\.\\',       # Path traversal
        r'\$\{.*\}',           # Template injection
        r'\{\{.*\}\}',         # Template injection
    ])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_validation_attempts: int = 5
    
    # Content type validation
    validate_content_types: bool = True
    allowed_content_types: Set[str] = field(default_factory=lambda: {
        'text/plain', 'text/html', 'text/markdown', 'application/json',
        'text/csv', 'application/xml', 'text/xml', 'application/yaml'
    })


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, severity: str = "error"):
        self.message = message
        self.field = field
        self.severity = severity
        super().__init__(message)


class InputValidator:
    """
    Comprehensive input validation system.
    
    Provides validation for:
    - String inputs with XSS/injection prevention
    - URLs and email addresses
    - File uploads and content
    - Structured data (JSON, XML)
    - Database inputs
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._compile_patterns()
        self._setup_builtin_rules()
        self._rate_limit_tracker = {}
        
    def _compile_patterns(self) -> None:
        """Compile regex patterns for blocked content."""
        self.blocked_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for pattern in self.config.blocked_patterns
        ]
        
    def _setup_builtin_rules(self) -> None:
        """Setup built-in validation rules."""
        self.builtin_rules = {
            'no_script_tags': ValidationRule(
                name="no_script_tags",
                validator=lambda x: not re.search(r'<script\b', str(x), re.IGNORECASE),
                error_message="Script tags are not allowed"
            ),
            'no_javascript_urls': ValidationRule(
                name="no_javascript_urls", 
                validator=lambda x: 'javascript:' not in str(x).lower(),
                error_message="JavaScript URLs are not allowed"
            ),
            'no_event_handlers': ValidationRule(
                name="no_event_handlers",
                validator=lambda x: not re.search(r'on\w+\s*=', str(x), re.IGNORECASE),
                error_message="Event handlers are not allowed"
            ),
            'no_sql_keywords': ValidationRule(
                name="no_sql_keywords",
                validator=lambda x: not re.search(
                    r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION|FROM|WHERE|INFORMATION_SCHEMA)\b',
                    str(x), re.IGNORECASE
                ),
                error_message="SQL keywords detected in input"
            ),
            'no_path_traversal': ValidationRule(
                name="no_path_traversal",
                validator=lambda x: not re.search(r'\.\.[\\/]', str(x)),
                error_message="Path traversal patterns detected"
            )
        }
        
    def validate_string(self, 
                       value: str, 
                       field_name: str = "input",
                       max_length: Optional[int] = None,
                       min_length: int = 0,
                       custom_rules: Optional[List[ValidationRule]] = None) -> str:
        """
        Validate string input.
        
        Args:
            value: String to validate
            field_name: Name of the field being validated
            max_length: Maximum allowed length
            min_length: Minimum required length
            custom_rules: Additional custom validation rules
            
        Returns:
            Validated and sanitized string
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string for {field_name}, got {type(value)}", field_name)
            
        # Check length constraints
        max_len = max_length or self.config.max_string_length
        if len(value) > max_len:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_len} characters", 
                field_name
            )
            
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters long", 
                field_name
            )
            
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(value):
                logger.warning(f"Blocked pattern detected in {field_name}: {pattern.pattern}")
                raise ValidationError(
                    f"{field_name} contains prohibited content pattern", 
                    field_name, 
                    "error"
                )
                
        # Apply built-in rules
        for rule in self.builtin_rules.values():
            if not rule.validator(value):
                raise ValidationError(f"{field_name}: {rule.error_message}", field_name, rule.severity)
                
        # Apply custom rules
        if custom_rules:
            for rule in custom_rules:
                if not rule.validator(value):
                    raise ValidationError(f"{field_name}: {rule.error_message}", field_name, rule.severity)
                    
        # Basic sanitization - remove null bytes and control characters
        sanitized = value.replace('\x00', '').replace('\r\n', '\n')
        
        # Remove or replace dangerous Unicode characters
        sanitized = self._sanitize_unicode(sanitized)
        
        return sanitized.strip()
        
    def validate_url(self, url: str, field_name: str = "url") -> str:
        """
        Validate URL input.
        
        Args:
            url: URL to validate
            field_name: Name of the field
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url:
            raise ValidationError(f"{field_name} is required", field_name)
            
        # Check length
        if len(url) > self.config.max_url_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {self.config.max_url_length}", 
                field_name
            )
            
        # Validate URL format
        try:
            if not validators.url(url):
                raise ValidationError(f"Invalid URL format for {field_name}", field_name)
        except Exception:
            raise ValidationError(f"Invalid URL format for {field_name}", field_name)
            
        # Parse URL and check components
        try:
            parsed = urlparse(url)
            
            # Check protocol
            if parsed.scheme not in {'http', 'https'}:
                raise ValidationError(f"Only HTTP/HTTPS URLs allowed for {field_name}", field_name)
                
            # Check for suspicious elements
            if any(danger in url.lower() for danger in ['javascript:', 'vbscript:', 'data:', 'file:']):
                raise ValidationError(f"Suspicious URL scheme detected in {field_name}", field_name)
                
        except Exception as e:
            raise ValidationError(f"URL parsing error for {field_name}: {str(e)}", field_name)
            
        return url
        
    def validate_email(self, email: str, field_name: str = "email") -> str:
        """
        Validate email address.
        
        Args:
            email: Email to validate
            field_name: Name of the field
            
        Returns:
            Validated email address
            
        Raises:
            ValidationError: If email is invalid
        """
        if not email:
            raise ValidationError(f"{field_name} is required", field_name)
            
        # Check length
        if len(email) > self.config.max_email_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {self.config.max_email_length}", 
                field_name
            )
            
        # Validate email format
        try:
            validated_email = validate_email(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise ValidationError(f"Invalid email format for {field_name}: {str(e)}", field_name)
            
    def validate_json(self, json_str: str, field_name: str = "json_data") -> Dict[str, Any]:
        """
        Validate JSON input.
        
        Args:
            json_str: JSON string to validate
            field_name: Name of the field
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not json_str.strip():
            raise ValidationError(f"{field_name} cannot be empty", field_name)
            
        # Check for dangerous patterns in JSON string
        dangerous_patterns = [
            r'__proto__',  # Prototype pollution
            r'constructor',  # Constructor pollution
            r'eval\s*\(',  # eval calls
            r'Function\s*\(',  # Function constructor
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, json_str, re.IGNORECASE):
                raise ValidationError(f"Dangerous pattern detected in {field_name}", field_name)
                
        # Parse JSON
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {field_name}: {str(e)}", field_name)
            
        # Validate parsed data doesn't contain dangerous values
        self._validate_json_data(parsed_data, field_name)
        
        return parsed_data
        
    def _validate_json_data(self, data: Any, field_name: str, path: str = "") -> None:
        """Recursively validate JSON data structure."""
        current_path = f"{field_name}.{path}" if path else field_name
        
        if isinstance(data, dict):
            # Check for dangerous keys
            dangerous_keys = {'__proto__', 'constructor', 'prototype'}
            for key in data.keys():
                if key in dangerous_keys:
                    raise ValidationError(f"Dangerous key '{key}' in {current_path}", field_name)
                    
                # Recursively validate values
                self._validate_json_data(data[key], field_name, f"{path}.{key}" if path else key)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_json_data(item, field_name, f"{path}[{i}]" if path else f"[{i}]")
                
        elif isinstance(data, str):
            # Validate string values in JSON
            try:
                self.validate_string(data, current_path, max_length=1000)
            except ValidationError:
                # Re-raise with JSON context
                raise ValidationError(f"Invalid string value in {current_path}", field_name)
                
    def validate_file_upload(self, 
                           filename: str, 
                           content: bytes = None, 
                           content_type: str = None) -> Dict[str, Any]:
        """
        Validate file upload.
        
        Args:
            filename: Name of the uploaded file
            content: File content bytes
            content_type: MIME type of the file
            
        Returns:
            Validation results dictionary
            
        Raises:
            ValidationError: If file is invalid
        """
        results = {
            'filename': filename,
            'safe': True,
            'warnings': [],
            'content_type': content_type
        }
        
        # Validate filename
        if len(filename) > self.config.max_filename_length:
            raise ValidationError(f"Filename too long: {len(filename)} > {self.config.max_filename_length}")
            
        # Check for path traversal in filename
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValidationError("Path traversal patterns in filename")
            
        # Check file extension
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if file_ext and file_ext not in self.config.allowed_file_extensions:
            raise ValidationError(f"File extension '{file_ext}' not allowed")
            
        # Validate content type if provided
        if content_type and self.config.validate_content_types:
            if content_type not in self.config.allowed_content_types:
                results['warnings'].append(f"Unusual content type: {content_type}")
                
        # Validate content if provided
        if content:
            # Check for embedded malicious content
            content_str = content.decode('utf-8', errors='ignore')[:10000]  # First 10KB
            
            for pattern in self.blocked_patterns:
                if pattern.search(content_str):
                    raise ValidationError("Malicious content detected in file")
                    
            results['content_size'] = len(content)
            
        return results
        
    def validate_database_input(self, value: Any, field_name: str) -> Any:
        """
        Validate input intended for database operations.
        
        Args:
            value: Value to validate
            field_name: Name of the field
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If input is unsafe for database
        """
        if isinstance(value, str):
            # Extra strict validation for database inputs
            sql_injection_patterns = [
                r"';\s*(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER)\s+",
                r"UNION\s+SELECT\s+",
                r"1\s*=\s*1|1\s*=\s*'1'",
                r"OR\s+1\s*=\s*1",
                r"'.*OR.*'.*=.*'",
                r";\s*--",
                r"/\*.*\*/",
                r"xp_\w+",  # SQL Server extended procedures
                r"sp_\w+",  # SQL Server stored procedures
            ]
            
            for pattern in sql_injection_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValidationError(f"SQL injection pattern detected in {field_name}", field_name)
                    
            return self.validate_string(value, field_name)
            
        elif isinstance(value, (int, float)):
            # Validate numeric ranges
            if isinstance(value, float):
                if not (-1e10 <= value <= 1e10):  # Reasonable float range
                    raise ValidationError(f"Numeric value out of range for {field_name}", field_name)
            return value
            
        elif isinstance(value, bool):
            return value
            
        elif value is None:
            return None
            
        else:
            raise ValidationError(f"Unsupported data type for database input: {type(value)}", field_name)
            
    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize Unicode characters that could be dangerous."""
        # Remove or replace dangerous Unicode characters
        dangerous_unicode = [
            '\ufeff',  # BOM
            '\u200b',  # Zero width space
            '\u200c',  # Zero width non-joiner
            '\u200d',  # Zero width joiner
            '\u2060',  # Word joiner
            '\u180e',  # Mongolian vowel separator
        ]
        
        for char in dangerous_unicode:
            text = text.replace(char, '')
            
        return text
        
    def batch_validate(self, data: Dict[str, Any], validation_schema: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Validate multiple fields using a schema.
        
        Args:
            data: Dictionary of field names to values
            validation_schema: Dictionary of field validation configurations
            
        Returns:
            Dictionary of validated data
            
        Raises:
            ValidationError: If any field validation fails
        """
        validated_data = {}
        errors = []
        
        for field_name, field_value in data.items():
            if field_name not in validation_schema:
                continue  # Skip unknown fields
                
            schema = validation_schema[field_name]
            field_type = schema.get('type', 'string')
            
            try:
                if field_type == 'string':
                    validated_data[field_name] = self.validate_string(
                        field_value, 
                        field_name,
                        max_length=schema.get('max_length'),
                        min_length=schema.get('min_length', 0)
                    )
                elif field_type == 'url':
                    validated_data[field_name] = self.validate_url(field_value, field_name)
                elif field_type == 'email':
                    validated_data[field_name] = self.validate_email(field_value, field_name)
                elif field_type == 'json':
                    validated_data[field_name] = self.validate_json(field_value, field_name)
                else:
                    validated_data[field_name] = field_value
                    
            except ValidationError as e:
                errors.append(e)
                
        if errors:
            # Combine all error messages
            error_messages = [str(e) for e in errors]
            raise ValidationError(f"Validation failed: {'; '.join(error_messages)}")
            
        return validated_data
        
    def get_validation_report(self, data: Any, field_name: str = "input") -> Dict[str, Any]:
        """
        Generate validation report without raising exceptions.
        
        Args:
            data: Data to analyze
            field_name: Name of the field
            
        Returns:
            Validation analysis report
        """
        report = {
            'field_name': field_name,
            'data_type': type(data).__name__,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'security_score': 100
        }
        
        if isinstance(data, str):
            report['length'] = len(data)
            
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                matches = pattern.findall(data)
                if matches:
                    report['errors'].append(f"Blocked pattern: {pattern.pattern}")
                    report['is_valid'] = False
                    report['security_score'] -= 20
                    
            # Check built-in rules
            for rule in self.builtin_rules.values():
                if not rule.validator(data):
                    if rule.severity == "error":
                        report['errors'].append(rule.error_message)
                        report['is_valid'] = False
                        report['security_score'] -= 10
                    else:
                        report['warnings'].append(rule.error_message)
                        report['security_score'] -= 5
                        
        return report


# Default instance for easy usage
default_validator = InputValidator()


def validate_string(value: str, field_name: str = "input") -> str:
    """Quick function for string validation using default config."""
    return default_validator.validate_string(value, field_name)


def validate_url(url: str) -> str:
    """Quick function for URL validation using default config."""
    return default_validator.validate_url(url)


def validate_email(email: str) -> str:
    """Quick function for email validation using default config."""
    return default_validator.validate_email(email)