"""
SQLAlchemy models for content pipeline state management.

Defines database schema for articles, sources, duplicates, and pipeline runs.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    
    INGEST = "ingest"
    NORMALIZE = "normalize"
    DEDUP = "dedup"
    ENRICH = "enrich"
    PUBLISH = "publish"


class ContentStatus(str, Enum):
    """Content processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ContentType(str, Enum):
    """Content type classification."""
    
    ARTICLE = "article"
    GLOSSARY_TERM = "glossary_term"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    NEWS = "news"


class Source(Base):
    """Content source configuration and metadata."""
    
    __tablename__ = "sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    base_url = Column(String(2048), nullable=False)
    source_type = Column(String(50), nullable=False)  # 'rss', 'sitemap', 'manual'
    
    # Configuration
    config = Column(JSON, nullable=True)  # Source-specific settings
    crawl_frequency = Column(Integer, default=3600)  # seconds
    max_articles_per_run = Column(Integer, default=100)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_crawl = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    articles = relationship("Article", back_populates="source")


class Article(Base):
    """Article content and processing metadata."""
    
    __tablename__ = "articles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id"), nullable=False)
    
    # Content identifiers
    url = Column(String(2048), nullable=False)
    title = Column(String(1024), nullable=True)
    slug = Column(String(255), nullable=True)
    
    # Raw content
    raw_html = Column(Text, nullable=True)
    raw_text = Column(Text, nullable=True)
    
    # Processed content
    cleaned_content = Column(Text, nullable=True)
    markdown_content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    
    # Content metadata
    content_type = Column(SQLEnum(ContentType), default=ContentType.ARTICLE)
    language = Column(String(10), default="en")
    word_count = Column(Integer, nullable=True)
    reading_time = Column(Integer, nullable=True)  # minutes
    
    # Deduplication
    simhash = Column(String(64), nullable=True)
    content_hash = Column(String(64), nullable=True)
    
    # Classification and tags
    tags = Column(JSON, default=list)  # List[str]
    categories = Column(JSON, default=list)  # List[str]
    topics = Column(JSON, default=list)  # List[str]
    
    # SEO and metadata
    meta_description = Column(Text, nullable=True)
    meta_keywords = Column(JSON, default=list)  # List[str]
    author = Column(String(255), nullable=True)
    publish_date = Column(DateTime, nullable=True)
    
    # Processing status
    current_stage = Column(SQLEnum(PipelineStage), default=PipelineStage.INGEST)
    status = Column(SQLEnum(ContentStatus), default=ContentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    
    # Timestamps
    discovered_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    source = relationship("Source", back_populates="articles")
    duplicates = relationship("ContentDuplicate", foreign_keys="[ContentDuplicate.article_id]")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint("source_id", "url", name="uq_article_source_url"),
        # Performance indexes
        Index("idx_article_status_stage", status, current_stage),
        Index("idx_article_simhash", simhash),
        Index("idx_article_content_hash", content_hash),
        Index("idx_article_publish_date", publish_date),
        Index("idx_article_created_at", created_at),
        Index("idx_article_quality_score", quality_score),
        # Composite indexes for common queries
        Index("idx_article_source_status", source_id, status),
        Index("idx_article_stage_created", current_stage, created_at),
        # Partial indexes for active content
        Index("idx_article_active_content", status, 
              postgresql_where=(status.in_(['completed', 'processing']))),
    )


class ContentDuplicate(Base):
    """Duplicate content relationships and similarity scores."""
    
    __tablename__ = "content_duplicates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False)
    duplicate_article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False)
    
    # Similarity metrics
    similarity_score = Column(Float, nullable=False)
    simhash_distance = Column(Integer, nullable=True)
    minhash_jaccard = Column(Float, nullable=True)
    
    # Detection method
    detection_method = Column(String(50), nullable=False)  # 'simhash', 'minhash', 'exact'
    confidence_score = Column(Float, nullable=True)
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolution_action = Column(String(50), nullable=True)  # 'keep_original', 'keep_duplicate', 'merge'
    resolved_by = Column(String(255), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("article_id", "duplicate_article_id", name="uq_duplicate_pair"),
    )


class PipelineRun(Base):
    """Pipeline execution runs and their status."""
    
    __tablename__ = "pipeline_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Run identification
    run_name = Column(String(255), nullable=True)
    trigger = Column(String(50), nullable=False)  # 'scheduled', 'manual', 'api'
    
    # Configuration
    config_snapshot = Column(JSON, nullable=True)
    source_filters = Column(JSON, default=list)  # List[str] of source names
    stage_filters = Column(JSON, default=list)  # List[PipelineStage]
    
    # Status
    status = Column(SQLEnum(ContentStatus), default=ContentStatus.PENDING)
    current_stage = Column(SQLEnum(PipelineStage), nullable=True)
    
    # Statistics
    articles_processed = Column(Integer, default=0)
    articles_success = Column(Integer, default=0)
    articles_failed = Column(Integer, default=0)
    articles_skipped = Column(Integer, default=0)
    duplicates_found = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class EnrichmentTask(Base):
    """Content enrichment tasks and their results."""
    
    __tablename__ = "enrichment_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False)
    
    # Task details
    task_type = Column(String(50), nullable=False)  # 'summarize', 'cross_link', 'classify'
    priority = Column(Integer, default=100)
    
    # Input/Output
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    
    # Status
    status = Column(SQLEnum(ContentStatus), default=ContentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Processing details
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    article = relationship("Article")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("article_id", "task_type", name="uq_enrichment_article_task"),
    )