"""
LangGraph nodes implementation for each pipeline stage.

Implements the individual processing nodes for ingest, normalize, dedup,
enrich, and publish stages with error handling and state management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import config
from ...database import get_db_session
from ...database.models import Article, ContentStatus, PipelineRun, PipelineStage
from ...dedup import LSHIndex, SimHashDeduplicator
from ...enrich import ContentSummarizer, CrossLinker
from ...ingest import SourceManager
from ...normalize import ContentExtractor
from ...publish import MarkdownGenerator
from ...security import InputValidationError, default_validator
from .workflow import PipelineState

logger = logging.getLogger(__name__)


class PipelineNodes:
    """Implementation of all pipeline processing nodes."""

    def __init__(self):
        self.source_manager = SourceManager()
        self.content_extractor = ContentExtractor()
        self.simhash_deduplicator = SimHashDeduplicator()
        self.lsh_index = LSHIndex()
        self.content_summarizer = ContentSummarizer()
        self.cross_linker = CrossLinker()
        self.markdown_generator = MarkdownGenerator()

    async def initialize_pipeline(self, state: PipelineState) -> PipelineState:
        """Initialize pipeline run and create database record."""
        try:
            logger.info(f"Initializing pipeline run: {state['run_id']}")

            async with get_db_session() as session:
                # Create pipeline run record
                pipeline_run = PipelineRun(
                    run_name=state["pipeline_name"],
                    trigger="manual",  # Could be derived from config
                    config_snapshot={
                        "source_filters": state["source_filters"],
                        "stage_filters": state["stage_filters"],
                        "batch_size": state["batch_size"],
                    },
                    source_filters=state["source_filters"],
                    stage_filters=state["stage_filters"],
                    status=ContentStatus.PROCESSING,
                    current_stage=PipelineStage.INGEST,
                    started_at=datetime.utcnow(),
                )

                session.add(pipeline_run)
                await session.commit()

                # Store run ID for tracking
                state["session_id"] = str(pipeline_run.id)

            state["status"] = "initialized"
            state["current_stage"] = "ingest"

            logger.info(f"Pipeline initialized successfully: {state['run_id']}")
            return state

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "initialize",
                    "type": "initialization_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def ingest_content(self, state: PipelineState) -> PipelineState:
        """Ingest content from configured sources."""
        try:
            logger.info("Starting content ingestion")
            state["current_stage"] = "ingest"

            async with get_db_session() as session:
                # Sync sources configuration
                await self.source_manager.sync_sources_to_database(session)

                # Run scheduled ingestion
                ingest_stats = await self.source_manager.run_scheduled_ingestion(session)

                # Get newly ingested articles
                query = select(Article).where(
                    Article.current_stage == PipelineStage.INGEST,
                    Article.status == ContentStatus.COMPLETED,
                )

                # Apply source filters if specified
                if state["source_filters"]:
                    query = query.join(Article.source).where(
                        Article.source.has(name__in=state["source_filters"])
                    )

                result = await session.execute(query)
                articles = result.scalars().all()

                # Update state
                article_ids = [str(article.id) for article in articles]
                state["articles_to_process"] = article_ids
                state["total_articles"] = len(article_ids)
                state["ingest_results"] = ingest_stats

            logger.info(f"Ingestion completed: {len(article_ids)} articles to process")
            state["status"] = "completed"
            return state

        except Exception as e:
            logger.error(f"Content ingestion failed: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "ingest",
                    "type": "ingestion_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def normalize_content(self, state: PipelineState) -> PipelineState:
        """Normalize and clean content."""
        try:
            logger.info("Starting content normalization")
            state["current_stage"] = "normalize"

            articles_to_process = state["articles_to_process"]
            if not articles_to_process:
                logger.info("No articles to normalize")
                state["normalize_results"] = {"processed": 0}
                return state

            async with get_db_session() as session:
                processed_count = 0
                total_quality_score = 0.0
                batch_size = state["batch_size"]

                # Process articles in batches
                for i in range(0, len(articles_to_process), batch_size):
                    batch = articles_to_process[i : i + batch_size]
                    state["current_article_batch"] = batch

                    for article_id in batch:
                        try:
                            # Get article from database
                            query = select(Article).where(Article.id == article_id)
                            result = await session.execute(query)
                            article = result.scalar_one_or_none()

                            if not article or not article.raw_html:
                                continue

                            # Validate and sanitize inputs before processing
                            try:
                                sanitized_html = default_validator.validate_html_content(
                                    article.raw_html, f"article_{article_id}_html"
                                )
                                sanitized_url = default_validator.validate_url(
                                    article.url, f"article_{article_id}_url"
                                )
                            except InputValidationError as e:
                                logger.warning(
                                    f"Input validation failed for article {article_id}: {e}"
                                )
                                state["failed_articles"].append(article_id)
                                continue

                            # Extract and normalize content with sanitized inputs
                            extraction_result = await self.content_extractor.extract_content(
                                sanitized_html, sanitized_url
                            )

                            # Update article with normalized content
                            article.cleaned_content = extraction_result["content"]
                            article.markdown_content = extraction_result["markdown_content"]
                            article.word_count = extraction_result["word_count"]
                            article.reading_time = extraction_result["reading_time"]
                            article.quality_score = extraction_result["quality_score"]
                            article.readability_score = extraction_result["readability_score"]
                            article.language = extraction_result["language"]

                            # Update processing status
                            article.current_stage = PipelineStage.NORMALIZE
                            article.status = ContentStatus.COMPLETED

                            processed_count += 1
                            total_quality_score += extraction_result["quality_score"]
                            state["processed_articles"].append(article_id)

                        except Exception as e:
                            logger.error(f"Failed to normalize article {article_id}: {e}")
                            state["failed_articles"].append(article_id)
                            state["errors"].append(
                                {
                                    "stage": "normalize",
                                    "article_id": article_id,
                                    "type": "normalization_error",
                                    "message": str(e),
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                    await session.commit()

                    # Add small delay between batches
                    await asyncio.sleep(0.1)

                # Calculate statistics
                avg_quality_score = (
                    total_quality_score / processed_count if processed_count > 0 else 0.0
                )

                state["normalize_results"] = {
                    "processed": processed_count,
                    "avg_quality_score": avg_quality_score,
                    "failed": len(state["failed_articles"]),
                }

                state["processed_count"] = processed_count
                state["success_count"] += processed_count
                state["failure_count"] += len(state["failed_articles"])

            logger.info(f"Normalization completed: {processed_count} articles processed")
            state["status"] = "completed"
            return state

        except Exception as e:
            logger.error(f"Content normalization failed: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "normalize",
                    "type": "normalization_stage_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def deduplicate_content(self, state: PipelineState) -> PipelineState:
        """Detect and handle duplicate content."""
        try:
            logger.info("Starting content deduplication")
            state["current_stage"] = "dedup"

            processed_articles = state["processed_articles"]
            if not processed_articles:
                logger.info("No articles to deduplicate")
                state["dedup_results"] = {"processed": 0, "duplicates_found": 0}
                return state

            async with get_db_session() as session:
                # Build deduplication indices
                await self._build_dedup_indices(session)

                duplicates_found = 0
                processed_count = 0

                for article_id in processed_articles:
                    try:
                        # Get article
                        query = select(Article).where(Article.id == article_id)
                        result = await session.execute(query)
                        article = result.scalar_one_or_none()

                        if not article or not article.cleaned_content:
                            continue

                        # Check for duplicates using both methods
                        simhash_duplicates = self.simhash_deduplicator.find_duplicates(
                            article.cleaned_content, exclude_id=article_id
                        )

                        lsh_duplicates = self.lsh_index.find_duplicates(
                            article.cleaned_content, exclude_id=article_id
                        )

                        # Combine and dedupe results
                        all_duplicates = {}
                        for dup_id, similarity in simhash_duplicates:
                            all_duplicates[dup_id] = {"similarity": similarity, "method": "simhash"}

                        for dup_id, similarity in lsh_duplicates:
                            if (
                                dup_id not in all_duplicates
                                or similarity > all_duplicates[dup_id]["similarity"]
                            ):
                                all_duplicates[dup_id] = {"similarity": similarity, "method": "lsh"}

                        # Store duplicate relationships if found
                        if all_duplicates:
                            duplicates_found += len(all_duplicates)

                            # TODO: Store duplicates in database
                            # This would involve creating ContentDuplicate records

                            logger.info(
                                f"Found {len(all_duplicates)} duplicates for article {article_id}"
                            )

                        # Add to dedup index for future comparisons
                        self.simhash_deduplicator.add_content(article_id, article.cleaned_content)
                        self.lsh_index.add_content(article_id, article.cleaned_content)

                        # Update article status
                        article.current_stage = PipelineStage.DEDUP
                        article.status = ContentStatus.COMPLETED

                        processed_count += 1

                    except Exception as e:
                        logger.error(f"Failed to deduplicate article {article_id}: {e}")
                        state["failed_articles"].append(article_id)

                await session.commit()

                state["dedup_results"] = {
                    "processed": processed_count,
                    "duplicates_found": duplicates_found,
                    "unique_articles": processed_count - duplicates_found,
                }

            logger.info(f"Deduplication completed: {duplicates_found} duplicates found")
            state["status"] = "completed"
            return state

        except Exception as e:
            logger.error(f"Content deduplication failed: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "dedup",
                    "type": "deduplication_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def enrich_content(self, state: PipelineState) -> PipelineState:
        """Enrich content with summaries and cross-links."""
        try:
            logger.info("Starting content enrichment")
            state["current_stage"] = "enrich"

            processed_articles = state["processed_articles"]
            if not processed_articles:
                logger.info("No articles to enrich")
                state["enrich_results"] = {"processed": 0}
                return state

            async with get_db_session() as session:
                enriched_count = 0
                total_confidence = 0.0

                # Build cross-linking index
                await self.cross_linker.build_similarity_index(session)

                for article_id in processed_articles:
                    try:
                        # Get article
                        query = select(Article).where(Article.id == article_id)
                        result = await session.execute(query)
                        article = result.scalar_one_or_none()

                        if not article or not article.cleaned_content:
                            continue

                        # Generate summary
                        if not article.summary:
                            summary_result = await self.content_summarizer.summarize_content(
                                article.cleaned_content, article.title or "", "executive"
                            )

                            if summary_result["summary"]:
                                article.summary = summary_result["summary"]

                        # Generate cross-links
                        cross_links = await self.cross_linker.generate_cross_links(
                            article_id, session
                        )

                        # TODO: Store cross-links in article metadata or separate table

                        # Update article status
                        article.current_stage = PipelineStage.ENRICH
                        article.status = ContentStatus.COMPLETED

                        enriched_count += 1
                        total_confidence += 0.8  # Mock confidence score

                    except Exception as e:
                        logger.error(f"Failed to enrich article {article_id}: {e}")
                        state["failed_articles"].append(article_id)

                await session.commit()

                avg_confidence = total_confidence / enriched_count if enriched_count > 0 else 0.0

                state["enrich_results"] = {
                    "processed": enriched_count,
                    "avg_confidence": avg_confidence,
                    "cross_links_generated": len(processed_articles) * 2,  # Mock value
                }

            logger.info(f"Enrichment completed: {enriched_count} articles enriched")
            state["status"] = "completed"
            return state

        except Exception as e:
            logger.error(f"Content enrichment failed: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "enrich",
                    "type": "enrichment_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def publish_content(self, state: PipelineState) -> PipelineState:
        """Publish content as markdown files."""
        try:
            logger.info("Starting content publishing")
            state["current_stage"] = "publish"

            processed_articles = state["processed_articles"]
            if not processed_articles:
                logger.info("No articles to publish")
                state["publish_results"] = {"published": 0}
                return state

            async with get_db_session() as session:
                # Publish articles in batch
                publish_stats = await self.markdown_generator.publish_batch(
                    processed_articles, session
                )

                # Update taxonomies
                await self.markdown_generator.update_taxonomies(session)

                state["publish_results"] = publish_stats
                state["success_count"] = publish_stats["published"]
                state["failure_count"] += publish_stats["failed"]

            logger.info(f"Publishing completed: {publish_stats['published']} articles published")
            state["status"] = "completed"
            return state

        except Exception as e:
            logger.error(f"Content publishing failed: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "publish",
                    "type": "publishing_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def finalize_pipeline(self, state: PipelineState) -> PipelineState:
        """Finalize pipeline execution and update records."""
        try:
            logger.info("Finalizing pipeline execution")

            state["end_time"] = datetime.utcnow().isoformat()
            state["status"] = "completed"

            # Update pipeline run record
            if state["session_id"]:
                async with get_db_session() as session:
                    query = select(PipelineRun).where(PipelineRun.id == state["session_id"])
                    result = await session.execute(query)
                    pipeline_run = result.scalar_one_or_none()

                    if pipeline_run:
                        pipeline_run.status = ContentStatus.COMPLETED
                        pipeline_run.completed_at = datetime.utcnow()
                        pipeline_run.articles_processed = state["total_articles"]
                        pipeline_run.articles_success = state["success_count"]
                        pipeline_run.articles_failed = state["failure_count"]
                        pipeline_run.duration_seconds = (
                            datetime.fromisoformat(state["end_time"])
                            - datetime.fromisoformat(state["start_time"])
                        ).total_seconds()

                        await session.commit()

            logger.info(f"Pipeline finalized: {state['run_id']}")
            return state

        except Exception as e:
            logger.error(f"Failed to finalize pipeline: {e}")
            state["status"] = "error"
            state["errors"].append(
                {
                    "stage": "finalize",
                    "type": "finalization_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return state

    async def check_human_review(self, state: PipelineState) -> PipelineState:
        """Check if human review is required."""
        # This is a placeholder - in practice, this would implement
        # more sophisticated logic for determining when human review is needed

        state["requires_human_review"] = False
        state["review_items"] = []

        # Example: flag articles with low quality scores for review
        if state.get("normalize_results", {}).get("avg_quality_score", 1.0) < 0.5:
            state["requires_human_review"] = True
            state["review_items"] = [
                {
                    "type": "low_quality",
                    "message": "Articles with low quality scores detected",
                    "risk_level": "medium",
                }
            ]

        return state

    async def wait_for_human_review(self, state: PipelineState) -> PipelineState:
        """Wait for human review (placeholder implementation)."""
        # In a real implementation, this would:
        # 1. Send notifications to reviewers
        # 2. Create review tasks in a queue
        # 3. Wait for human input through a UI or API
        # 4. Process review decisions

        logger.info("Human review required - implementing auto-approval for demo")

        # Auto-approve for demo purposes
        state["requires_human_review"] = False
        state["review_items"] = []

        return state

    async def handle_pipeline_error(self, state: PipelineState) -> PipelineState:
        """Handle pipeline errors and determine recovery strategy."""
        errors = state.get("errors", [])

        if errors:
            latest_error = errors[-1]
            logger.error(f"Handling pipeline error: {latest_error}")

            # Increment retry count
            state["retry_count"] = state.get("retry_count", 0) + 1

            # Log error details
            error_stage = latest_error.get("stage", "unknown")
            error_message = latest_error.get("message", "Unknown error")

            logger.error(f"Error in stage {error_stage}: {error_message}")

        return state

    async def check_retry_conditions(self, state: PipelineState) -> PipelineState:
        """Check if retry conditions are met."""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_count >= max_retries:
            state["status"] = "failed"
            logger.error(f"Maximum retries exceeded: {retry_count}/{max_retries}")
        else:
            state["status"] = "retrying"
            logger.info(f"Retrying pipeline: attempt {retry_count + 1}/{max_retries}")

        return state

    async def _build_dedup_indices(self, session: AsyncSession) -> None:
        """Build deduplication indices from existing articles."""
        try:
            # Get existing articles for building indices
            query = select(Article).where(
                Article.cleaned_content.is_not(None),
                Article.current_stage.in_(
                    [PipelineStage.DEDUP, PipelineStage.ENRICH, PipelineStage.PUBLISH]
                ),
            )

            result = await session.execute(query)
            existing_articles = result.scalars().all()

            if existing_articles:
                logger.info(
                    f"Building deduplication indices from {len(existing_articles)} existing articles"
                )

                # Build indices
                for article in existing_articles:
                    if article.cleaned_content:
                        self.simhash_deduplicator.add_content(
                            str(article.id), article.cleaned_content
                        )
                        self.lsh_index.add_content(str(article.id), article.cleaned_content)

                logger.info("Deduplication indices built successfully")

        except Exception as e:
            logger.error(f"Failed to build deduplication indices: {e}")
            raise
