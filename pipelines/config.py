"""
Pipeline Configuration Management

Centralized configuration for the content pipeline with environment variable support
and validation using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_knowledge"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False


class ScrapingConfig(BaseModel):
    """Web scraping configuration."""
    
    respect_robots_txt: bool = True
    user_agent: str = "AI-Knowledge-Bot/1.0 (+https://ai-knowledge.org/bot)"
    request_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    concurrent_requests: int = 5
    max_content_size: int = 10 * 1024 * 1024  # 10MB


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