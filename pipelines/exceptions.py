"""
Custom exceptions for the content pipeline.

Defines specific exception types for different pipeline stages
and error conditions to enable better error handling.
"""

from typing import Any, Dict, Optional


class PipelineException(Exception):
    """Base exception for all pipeline-related errors."""

    def __init__(
        self, message: str, stage: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.stage = stage
        self.context = context or {}
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "stage": self.stage,
            "context": self.context,
        }


class ConfigurationError(PipelineException):
    """Raised when there's a configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key


class DatabaseError(PipelineException):
    """Raised when there's a database-related error."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation


class IngestionError(PipelineException):
    """Raised during content ingestion failures."""

    def __init__(
        self, message: str, source: Optional[str] = None, url: Optional[str] = None, **kwargs
    ):
        super().__init__(message, stage="ingest", **kwargs)
        self.source = source
        self.url = url


class NormalizationError(PipelineException):
    """Raised during content normalization failures."""

    def __init__(self, message: str, article_id: Optional[str] = None, **kwargs):
        super().__init__(message, stage="normalize", **kwargs)
        self.article_id = article_id


class DeduplicationError(PipelineException):
    """Raised during deduplication process failures."""

    def __init__(self, message: str, article_id: Optional[str] = None, **kwargs):
        super().__init__(message, stage="dedup", **kwargs)
        self.article_id = article_id


class EnrichmentError(PipelineException):
    """Raised during content enrichment failures."""

    def __init__(
        self,
        message: str,
        article_id: Optional[str] = None,
        enrichment_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, stage="enrich", **kwargs)
        self.article_id = article_id
        self.enrichment_type = enrichment_type


class PublishingError(PipelineException):
    """Raised during content publishing failures."""

    def __init__(
        self,
        message: str,
        article_id: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, stage="publish", **kwargs)
        self.article_id = article_id
        self.file_path = file_path


class ValidationError(PipelineException):
    """Raised when content validation fails."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        field: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.field = field


class RateLimitError(PipelineException):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.service = service
        self.retry_after = retry_after


class NetworkError(PipelineException):
    """Raised for network-related errors."""

    def __init__(
        self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code


class ContentError(PipelineException):
    """Raised when content is invalid or problematic."""

    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        quality_issue: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.content_type = content_type
        self.quality_issue = quality_issue


class WorkflowError(PipelineException):
    """Raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        node: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.node = node
        self.workflow_state = state


class RetryableError(PipelineException):
    """Base class for errors that can be retried."""

    def __init__(self, message: str, max_retries: int = 3, backoff_factor: float = 1.0, **kwargs):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor


class NonRetryableError(PipelineException):
    """Base class for errors that should not be retried."""

    pass


# Specific retryable errors
class TemporaryNetworkError(RetryableError, NetworkError):
    """Network error that may be temporary."""

    pass


class TemporaryServiceError(RetryableError):
    """Service error that may be temporary."""

    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(message, **kwargs)
        self.service = service


class TemporaryDatabaseError(RetryableError, DatabaseError):
    """Database error that may be temporary."""

    pass


# Specific non-retryable errors
class PermanentValidationError(NonRetryableError, ValidationError):
    """Validation error that won't resolve with retries."""

    pass


class PermanentConfigurationError(NonRetryableError, ConfigurationError):
    """Configuration error that won't resolve with retries."""

    pass


class AuthenticationError(NonRetryableError):
    """Authentication/authorization error."""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.service = service


# Helper functions for error handling
def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable."""
    return isinstance(error, RetryableError) and not isinstance(error, NonRetryableError)


def get_retry_delay(error: RetryableError, attempt: int) -> float:
    """Calculate retry delay with exponential backoff."""
    if not isinstance(error, RetryableError):
        return 0.0

    base_delay = error.backoff_factor
    return min(base_delay * (2**attempt), 300.0)  # Max 5 minutes


def create_error_context(
    article_id: Optional[str] = None,
    source: Optional[str] = None,
    url: Optional[str] = None,
    stage: Optional[str] = None,
    **extra_context,
) -> Dict[str, Any]:
    """Create error context dictionary."""
    context = {}

    if article_id:
        context["article_id"] = article_id
    if source:
        context["source"] = source
    if url:
        context["url"] = url
    if stage:
        context["stage"] = stage

    context.update(extra_context)
    return context
