"""
Input validation and sanitization for pipeline processing.

Provides comprehensive validation and sanitization for all external inputs
to prevent injection attacks and ensure data integrity.
"""

import html
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class InputValidationError(Exception):
    """Input validation error."""

    pass

class InputValidator:
    """Validates and sanitizes input data for pipeline processing."""

    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",
        r"(\bAND\b\s*\d+\s*=\s*\d+)",
        r"(';|\";\s*--)",
        r"(\bxp_cmdshell\b|\bsp_executesql\b)",
    ]

    # XSS patterns to detect
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
        r"data:text/html",
        r"vbscript:",
        r"<img[^>]*onerror\s*=",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\" r"%2e%2e/",
        r"%252e%252e/",
        r"\.\.%2f",
        r"\.\.%5c",
    ]

    # Maximum string lengths
    MAX_URL_LENGTH = 2048
    MAX_CONTENT_LENGTH = 1000000  # 1MB
    MAX_TITLE_LENGTH = 500
    MAX_FIELD_LENGTH = 1000

    def __init__(self):
        self.compiled_sql_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS
        ]
        self.compiled_xss_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS
        ]
        self.compiled_path_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS
        ]

    def validate_string(self, value: str, field_name: str, max_length: Optional[int] = None) -> str:
        """
        Validate and sanitize a string input.

        Args:
            value: String to validate
            field_name: Name of the field for error messages
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise InputValidationError(f"{field_name} must be a string")

        # Check length
        if max_length is None:
            max_length = self.MAX_FIELD_LENGTH

        if len(value) > max_length:
            raise InputValidationError(f"{field_name} exceeds maximum length of {max_length}")

        # Check for SQL injection patterns
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value):
                logger.warning(f"Potential SQL injection detected in {field_name}")
                raise InputValidationError(f"Invalid characters in {field_name}")

        # Check for XSS patterns
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value):
                logger.warning(f"Potential XSS detected in {field_name}")
                raise InputValidationError(f"Invalid HTML/JavaScript in {field_name}")

        # HTML escape the string
        sanitized = html.escape(value, quote=False)

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        return sanitized

    def validate_url(self, url: str, field_name: str = "URL") -> str:
        """
        Validate and sanitize a URL.

        Args:
            url: URL to validate
            field_name: Name of the field for error messages

        Returns:
            Sanitized URL

        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(url, str):
            raise InputValidationError(f"{field_name} must be a string")

        # Check length
        if len(url) > self.MAX_URL_LENGTH:
            raise InputValidationError(f"{field_name} exceeds maximum length")

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise InputValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in ["http", "https", "ftp"]:
            raise InputValidationError(f"Invalid URL scheme: {parsed.scheme}")

        # Check for localhost/private IPs (prevent SSRF)
        hostname = parsed.hostname
        if hostname:
            if hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:  # nosec B104 - Intentional localhost check
                raise InputValidationError("URLs to localhost are not allowed")

            # Check for private IP ranges
            if (
                hostname.startswith("192.168.")
                or hostname.startswith("10.")
                or hostname.startswith("172.")
            ):
                raise InputValidationError("URLs to private IP addresses are not allowed")

        # Check for path traversal
        for pattern in self.compiled_path_patterns:
            if pattern.search(url):
                raise InputValidationError("Path traversal detected in URL")

        # Reconstruct clean URL
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"
        if parsed.fragment:
            clean_url += f"#{parsed.fragment}"

        return clean_url

    def validate_html_content(self, content: str, field_name: str = "content") -> str:
        """
        Validate and sanitize HTML content.

        Args:
            content: HTML content to validate
            field_name: Name of the field for error messages

        Returns:
            Sanitized HTML content

        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(content, str):
            raise InputValidationError(f"{field_name} must be a string")

        # Check length
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise InputValidationError(f"{field_name} exceeds maximum length")

        # Remove dangerous tags and attributes
        # Remove script tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove style tags with potentially dangerous content
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove event handlers
        content = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', "", content, flags=re.IGNORECASE)
        content = re.sub(r"\bon\w+\s*=\s*[^\s>]+", "", content, flags=re.IGNORECASE)

        # Remove javascript: protocol
        content = re.sub(r"javascript:", "", content, flags=re.IGNORECASE)
        content = re.sub(r"vbscript:", "", content, flags=re.IGNORECASE)

        # Remove data URIs that could contain scripts
        content = re.sub(r'data:text/html[^"\']*', "", content, flags=re.IGNORECASE)

        # Remove iframe, embed, object tags
        content = re.sub(
            r"<(iframe|embed|object)[^>]*>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove form tags to prevent CSRF
        content = re.sub(r"<form[^>]*>.*?</form>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove meta refresh
        content = re.sub(
            r"<meta[^>]*http-equiv[^>]*refresh[^>]*>", "", content, flags=re.IGNORECASE
        )

        return content

    def validate_json(self, data: Union[str, dict], field_name: str = "data") -> dict:
        """
        Validate JSON data.

        Args:
            data: JSON string or dictionary to validate
            field_name: Name of the field for error messages

        Returns:
            Validated dictionary

        Raises:
            InputValidationError: If validation fails
        """
        import json

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise InputValidationError(f"Invalid JSON in {field_name}: {e}")

        if not isinstance(data, dict):
            raise InputValidationError(f"{field_name} must be a JSON object")

        # Recursively validate all string values
        def validate_dict_values(d: dict) -> dict:
            result = {}
            for key, value in d.items():
                if isinstance(value, str):
                    # Basic string sanitization
                    result[key] = self.validate_string(
                        value, f"{field_name}.{key}", max_length=10000
                    )
                elif isinstance(value, dict):
                    result[key] = validate_dict_values(value)
                elif isinstance(value, list):
                    result[key] = [
                        (
                            validate_dict_values(item)
                            if isinstance(item, dict)
                            else (
                                self.validate_string(
                                    item, f"{field_name}.{key}[]", max_length=10000
                                )
                                if isinstance(item, str)
                                else item
                            )
                        )
                        for item in value
                    ]
                else:
                    result[key] = value
            return result

        return validate_dict_values(data)

    def validate_file_path(self, path: str, field_name: str = "path") -> Path:
        """
        Validate a file path to prevent path traversal.

        Args:
            path: File path to validate
            field_name: Name of the field for error messages

        Returns:
            Validated Path object

        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(path, str):
            raise InputValidationError(f"{field_name} must be a string")

        # Check for path traversal patterns
        for pattern in self.compiled_path_patterns:
            if pattern.search(path):
                raise InputValidationError(f"Path traversal detected in {field_name}")

        # Convert to Path and resolve
        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            raise InputValidationError(f"Invalid path: {e}")

        # Ensure path doesn't escape the project directory
        # You should configure this based on your allowed directories
        # For now, we'll just ensure it's not accessing system directories
        forbidden_paths = [
            Path("/etc"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/boot"),
            Path("/dev"),
            Path("/proc"),
            Path("/sys"),
            Path.home() / ".ssh",
            Path.home() / ".aws",
        ]

        for forbidden in forbidden_paths:
            try:
                path_obj.relative_to(forbidden)
                raise InputValidationError(f"Access to {forbidden} is not allowed")
            except ValueError:
                # Path is not relative to forbidden path, which is good
                pass

        return path_obj

    def validate_batch(
        self, items: List[Dict[str, Any]], validators: Dict[str, callable]
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of items.

        Args:
            items: List of items to validate
            validators: Dictionary of field names to validation functions

        Returns:
            List of validated items

        Raises:
            InputValidationError: If validation fails
        """
        validated_items = []

        for i, item in enumerate(items):
            validated_item = {}

            for field_name, validator_func in validators.items():
                if field_name in item:
                    try:
                        validated_item[field_name] = validator_func(
                            item[field_name], f"item[{i}].{field_name}"
                        )
                    except InputValidationError as e:
                        logger.error(f"Validation failed for item {i}, field {field_name}: {e}")
                        raise

            validated_items.append(validated_item)

        return validated_items

# Global validator instance
default_validator = InputValidator()

# Convenience functions
def validate_string(value: str, field_name: str, max_length: Optional[int] = None) -> str:
    """Validate and sanitize a string input."""
    return default_validator.validate_string(value, field_name, max_length)

def validate_url(url: str, field_name: str = "URL") -> str:
    """Validate and sanitize a URL."""
    return default_validator.validate_url(url, field_name)

def validate_html_content(content: str, field_name: str = "content") -> str:
    """Validate and sanitize HTML content."""
    return default_validator.validate_html_content(content, field_name)

def validate_json(data: Union[str, dict], field_name: str = "data") -> dict:
    """Validate JSON data."""
    return default_validator.validate_json(data, field_name)

def validate_file_path(path: str, field_name: str = "path") -> Path:
    """Validate a file path."""
    return default_validator.validate_file_path(path, field_name)
