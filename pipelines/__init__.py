"""
AI Knowledge Content Pipeline

A robust, scalable content pipeline for ingesting, processing, and publishing
AI/ML knowledge content with deduplication and enrichment capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Knowledge Team"

from .config import PipelineConfig
from .logging_config import setup_pipeline_logging, get_pipeline_logger
from .exceptions import PipelineException
from .monitoring import performance_monitor, health_checker

__all__ = [
    "PipelineConfig", 
    "setup_pipeline_logging", 
    "get_pipeline_logger",
    "PipelineException",
    "performance_monitor",
    "health_checker"
]