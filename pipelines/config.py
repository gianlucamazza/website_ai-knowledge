"""
Pipeline Configuration Management

Centralized configuration for the content pipeline with environment variable support
and validation using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    database: str = Field(default="ai_knowledge", env="DATABASE_NAME")
    username: str = Field(default="postgres", env="DATABASE_USERNAME")
    password: Optional[SecretStr] = Field(default=None, env="DATABASE_PASSWORD")
    # Conservative pool settings to prevent resource exhaustion
    pool_size: int = 10  # Reduced from 50
    max_overflow: int = 20  # Reduced from 100
    echo: bool = False
    # Connection performance settings
    pool_timeout: int = 30  # Seconds to wait for connection
    pool_recycle: int = 3600  # Recycle connections every hour
    query_timeout: int = 300  # 5 minute query timeout
    # Cache settings
    enable_query_cache: bool = True
    query_cache_size: int = 1000

    @validator("password")
    def validate_password(cls, v):
        if v is None:
            # Password is optional for local development
            return None
        # Ensure password is strong enough when provided
        password_str = (
            v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
        )
        if password_str and len(password_str) < 12:
            raise ValueError(
                "Database password must be at least 12 characters long when provided"
            )
        return v


class ScrapingConfig(BaseModel):
    """Web scraping configuration."""

    respect_robots_txt: bool = True
    user_agent: str = "AI-Knowledge-Bot/1.0 (+https://ai-knowledge.org/bot)"
    request_delay: float = 1.0  # Ethical delay between requests
    max_retries: int = 3
    timeout: int = 30  # Reasonable timeout
    concurrent_requests: int = 5  # Conservative concurrency
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    # Conservative connection settings
    connection_pool_size: int = 20
    connection_per_host: int = 5
    dns_cache_ttl: int = 300  # 5 minutes DNS cache
    keepalive_timeout: int = 60  # Keep connections alive
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: int = 300  # 5 minutes circuit breaker timeout


class DeduplicationConfig(BaseModel):
    """Deduplication algorithm configuration."""

    simhash_k: int = 3  # Number of differences allowed in SimHash
    minhash_threshold: float = 0.85  # Similarity threshold for MinHash
    lsh_num_perm: int = 256  # Number of permutations for LSH
    lsh_threshold: float = 0.8  # LSH similarity threshold
    content_min_length: int = 100  # Minimum content length for dedup


class EnrichmentConfig(BaseModel):
    """Content enrichment configuration."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    max_summary_length: int = 500
    enable_cross_linking: bool = True
    similarity_threshold: float = 0.7
    max_related_articles: int = 5


class PublishConfig(BaseModel):
    """Publishing configuration."""

    output_directory: str = "apps/site/src/content"
    articles_subdir: str = "articles"
    glossary_subdir: str = "glossary"
    taxonomies_subdir: str = "taxonomies"
    validate_frontmatter: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5


class PipelineConfig(BaseSettings):
    """Main pipeline configuration."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Core settings
    project_root: str = Field(default_factory=lambda: str(Path(__file__).parent.parent))
    data_directory: str = "data"
    temp_directory: str = "tmp"

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    publishing: PublishConfig = Field(default_factory=PublishConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Sources configuration
    sources_config_path: str = "ingest/sources.yaml"

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

    @validator("project_root")
    def validate_project_root(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Project root directory does not exist: {v}")
        return str(path.resolve())

    def get_data_path(self, *parts: str) -> Path:
        """Get path within the data directory."""
        return Path(self.project_root) / self.data_directory / Path(*parts)

    def get_temp_path(self, *parts: str) -> Path:
        """Get path within the temp directory."""
        return Path(self.project_root) / self.temp_directory / Path(*parts)

    def get_output_path(self, *parts: str) -> Path:
        """Get path within the output directory."""
        return Path(self.project_root) / self.publishing.output_directory / Path(*parts)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.get_data_path(),
            self.get_temp_path(),
            self.get_output_path(),
            self.get_output_path(self.publishing.articles_subdir),
            self.get_output_path(self.publishing.glossary_subdir),
            self.get_output_path(self.publishing.taxonomies_subdir),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = PipelineConfig()
