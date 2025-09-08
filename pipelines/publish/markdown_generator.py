"""
Markdown file generation with frontmatter validation and content formatting
for the Astro content collection system.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from slugify import slugify
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import config
from ..database.models import Article, ContentStatus, ContentType

logger = logging.getLogger(__name__)


class MarkdownGenerator:
    """Generates markdown files with proper frontmatter for Astro content collections."""

    def __init__(self):
        self.config = config.publishing
        self.output_base = Path(config.project_root) / self.config.output_directory

        # Ensure output directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all output directories exist."""
        directories = [
            self.output_base / self.config.articles_subdir,
            self.output_base / self.config.glossary_subdir,
            self.output_base / self.config.taxonomies_subdir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def publish_article(self, article_id: str, session: AsyncSession) -> bool:
        """
        Publish a single article as markdown file.

        Args:
            article_id: ID of the article to publish
            session: Database session

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get article from database
            query = select(Article).where(Article.id == article_id)
            result = await session.execute(query)
            article = result.scalar_one_or_none()

            if not article:
                logger.error(f"Article not found: {article_id}")
                return False

            # Validate article is ready for publishing
            if not self._validate_article_for_publishing(article):
                logger.warning(f"Article not ready for publishing: {article_id}")
                return False

            # Generate markdown content
            markdown_content = await self._generate_markdown_content(article)

            if not markdown_content:
                logger.error(f"Failed to generate markdown for article: {article_id}")
                return False

            # Determine output file path
            file_path = self._get_output_file_path(article)

            # Write markdown file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # Update article status
            article.status = ContentStatus.COMPLETED
            article.published_at = datetime.utcnow()
            await session.commit()

            logger.info(f"Published article: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error publishing article {article_id}: {e}")
            return False

    async def publish_batch(self, article_ids: List[str], session: AsyncSession) -> Dict:
        """
        Publish multiple articles in batch.

        Args:
            article_ids: List of article IDs to publish
            session: Database session

        Returns:
            Dict with publishing statistics
        """
        stats = {
            "total_articles": len(article_ids),
            "published": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }

        for article_id in article_ids:
            try:
                success = await self.publish_article(article_id, session)
                if success:
                    stats["published"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"{article_id}: {str(e)}")
                logger.error(f"Error publishing article {article_id}: {e}")

        logger.info(f"Batch publishing completed: {stats}")
        return stats

    async def _generate_markdown_content(self, article: Article) -> Optional[str]:
        """Generate complete markdown content with frontmatter."""
        try:
            # Generate frontmatter
            frontmatter = self._generate_frontmatter(article)

            # Get article content
            content = article.markdown_content or article.cleaned_content or ""

            if not content:
                logger.warning(f"No content available for article: {article.id}")
                return None

            # Process content
            processed_content = self._process_content(content, article)

            # Combine frontmatter and content
            frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)

            markdown_content = f"---\n{frontmatter_yaml}---\n\n{processed_content}"

            # Validate against schema if enabled
            if self.config.validate_frontmatter:
                if not self._validate_frontmatter(frontmatter, article.content_type):
                    logger.warning(f"Frontmatter validation failed for article: {article.id}")

            return markdown_content

        except Exception as e:
            logger.error(f"Error generating markdown content: {e}")
            return None

    def _generate_frontmatter(self, article: Article) -> Dict:
        """Generate frontmatter for the article."""
        frontmatter = {
            "title": article.title or "Untitled",
            "description": article.meta_description or article.summary or "",
            "pubDate": (
                article.publish_date.isoformat()
                if article.publish_date
                else datetime.utcnow().isoformat()
            ),
            "tags": article.tags or [],
            "categories": article.categories or [],
        }

        # Add content type specific fields
        if article.content_type == ContentType.ARTICLE:
            frontmatter.update(
                {
                    "type": "article",
                    "author": article.author,
                    "readingTime": article.reading_time or 0,
                    "wordCount": article.word_count or 0,
                }
            )
        elif article.content_type == ContentType.GLOSSARY_TERM:
            frontmatter.update(
                {
                    "type": "glossary",
                    "term": article.title,
                    "definition": article.summary,
                }
            )
        elif article.content_type == ContentType.TUTORIAL:
            frontmatter.update(
                {
                    "type": "tutorial",
                    "difficulty": self._determine_difficulty(article),
                    "estimatedTime": f"{article.reading_time or 5}m",
                }
            )

        # Add SEO fields
        if article.meta_description:
            frontmatter["seo"] = {
                "description": article.meta_description,
                "keywords": article.tags[:5] if article.tags else [],
            }

        # Add source information
        if article.source and article.url:
            frontmatter["source"] = {
                "name": article.source.name,
                "url": article.url,
                "canonical": article.url,
            }

        # Add quality metrics
        if hasattr(article, "quality_score") and article.quality_score:
            frontmatter["quality"] = {
                "score": round(article.quality_score, 2),
                "readabilityScore": round(article.readability_score or 0, 2),
            }

        # Clean up None values
        frontmatter = {k: v for k, v in frontmatter.items() if v is not None and v != ""}

        return frontmatter

    def _process_content(self, content: str, article: Article) -> str:
        """Process and enhance the content."""
        processed = content

        # Clean up markdown formatting
        processed = self._clean_markdown(processed)

        # Add cross-links if available
        # TODO: Integrate with cross-linking system

        # Add table of contents for long articles
        if article.word_count and article.word_count > 1500:
            toc = self._generate_table_of_contents(processed)
            if toc:
                processed = f"{toc}\n\n{processed}"

        # Add source attribution
        if article.url and article.source:
            source_attribution = self._generate_source_attribution(article)
            processed = f"{processed}\n\n{source_attribution}"

        return processed

    def _clean_markdown(self, content: str) -> str:
        """Clean and normalize markdown content."""
        # Fix excessive line breaks
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Fix heading spacing
        content = re.sub(r"\n(#{1,6}\s+[^\n]+)\n+", r"\n\n\1\n\n", content)

        # Fix list formatting
        content = re.sub(r"\n(\s*[-*+]\s+)", r"\n\1", content)
        content = re.sub(r"\n(\s*\d+\.\s+)", r"\n\1", content)

        # Fix code block formatting
        content = re.sub(r"\n```([^`]+)```\n", r"\n\n```\1```\n\n", content)

        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split("\n")]
        content = "\n".join(lines)

        return content.strip()

    def _generate_table_of_contents(self, content: str) -> Optional[str]:
        """Generate table of contents from headings."""
        try:
            # Extract headings
            headings = []
            heading_pattern = r"^(#{1,6})\s+(.+)$"

            for match in re.finditer(heading_pattern, content, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2).strip()
                slug = slugify(title)

                headings.append({"level": level, "title": title, "slug": slug})

            if len(headings) < 3:  # Don't add TOC for articles with few headings
                return None

            # Generate TOC markdown
            toc_lines = ["## Table of Contents"]

            for heading in headings:
                if heading["level"] <= 3:  # Only include h1-h3 in TOC
                    indent = "  " * (heading["level"] - 1)
                    toc_lines.append(f"{indent}- [{heading['title']}](#{heading['slug']})")

            return "\n".join(toc_lines)

        except Exception as e:
            logger.error(f"Error generating table of contents: {e}")
            return None

    def _generate_source_attribution(self, article: Article) -> str:
        """Generate source attribution footer."""
        attribution = "---\n\n"
        attribution += "*This article was originally published at "
        attribution += f"[{article.source.name}]({article.url})*"

        if article.author:
            attribution += f" by {article.author}"

        if article.publish_date:
            attribution += f" on {article.publish_date.strftime('%B %d, %Y')}"

        attribution += "."

        return attribution

    def _determine_difficulty(self, article: Article) -> str:
        """Determine tutorial difficulty based on content analysis."""
        if not article.cleaned_content:
            return "beginner"

        content = article.cleaned_content.lower()

        # Advanced indicators
        advanced_terms = [
            "algorithm",
            "optimization",
            "advanced",
            "complex",
            "sophisticated",
            "implementation",
            "architecture",
            "framework",
            "system design",
        ]

        # Intermediate indicators
        intermediate_terms = [
            "configure",
            "setup",
            "install",
            "deploy",
            "integrate",
            "customize",
            "extend",
            "modify",
        ]

        advanced_count = sum(1 for term in advanced_terms if term in content)
        intermediate_count = sum(1 for term in intermediate_terms if term in content)

        if advanced_count >= 3:
            return "advanced"
        elif intermediate_count >= 2 or advanced_count >= 1:
            return "intermediate"
        else:
            return "beginner"

    def _get_output_file_path(self, article: Article) -> Path:
        """Determine output file path for the article."""
        # Generate slug from title
        if article.slug:
            slug = article.slug
        elif article.title:
            slug = slugify(article.title)
        else:
            slug = slugify(f"article-{article.id}")

        # Determine subdirectory based on content type
        if article.content_type == ContentType.GLOSSARY_TERM:
            subdir = self.config.glossary_subdir
        else:
            subdir = self.config.articles_subdir

        # Ensure slug is unique by checking for existing files
        base_slug = slug
        counter = 1

        while True:
            file_path = self.output_base / subdir / f"{slug}.md"
            if not file_path.exists():
                break

            slug = f"{base_slug}-{counter}"
            counter += 1

        return file_path

    def _validate_article_for_publishing(self, article: Article) -> bool:
        """Validate that article is ready for publishing."""
        # Check required fields
        if not article.title:
            logger.warning(f"Article missing title: {article.id}")
            return False

        if not (article.markdown_content or article.cleaned_content):
            logger.warning(f"Article missing content: {article.id}")
            return False

        # Check minimum content length
        content = article.markdown_content or article.cleaned_content or ""
        if len(content.split()) < 100:
            logger.warning(f"Article content too short: {article.id}")
            return False

        # Check quality score if available
        if hasattr(article, "quality_score") and article.quality_score:
            if article.quality_score < 0.5:
                logger.warning(f"Article quality score too low: {article.id}")
                return False

        return True

    def _validate_frontmatter(self, frontmatter: Dict, content_type: ContentType) -> bool:
        """Validate frontmatter against schema."""
        try:
            # Basic validation - check required fields
            required_fields = ["title", "pubDate"]

            for field in required_fields:
                if field not in frontmatter or not frontmatter[field]:
                    logger.warning(f"Missing required frontmatter field: {field}")
                    return False

            # Type-specific validation
            if content_type == ContentType.GLOSSARY_TERM:
                if "term" not in frontmatter:
                    logger.warning("Glossary term missing 'term' field")
                    return False

            # Validate data types
            if "tags" in frontmatter and not isinstance(frontmatter["tags"], list):
                logger.warning("Tags field must be a list")
                return False

            if "categories" in frontmatter and not isinstance(frontmatter["categories"], list):
                logger.warning("Categories field must be a list")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating frontmatter: {e}")
            return False

    async def update_taxonomies(self, session: AsyncSession) -> bool:
        """Update taxonomy files with current tags and categories."""
        try:
            # Get all unique tags and categories from published articles
            query = select(Article).where(
                Article.status == ContentStatus.COMPLETED, Article.published_at.is_not(None)
            )
            result = await session.execute(query)
            articles = result.scalars().all()

            # Collect taxonomies
            all_tags = set()
            all_categories = set()
            tag_counts = {}
            category_counts = {}

            for article in articles:
                if article.tags:
                    for tag in article.tags:
                        all_tags.add(tag)
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                if article.categories:
                    for category in article.categories:
                        all_categories.add(category)
                        category_counts[category] = category_counts.get(category, 0) + 1

            # Generate taxonomy data
            tags_data = [
                {"name": tag, "count": tag_counts[tag], "slug": slugify(tag)}
                for tag in sorted(all_tags)
            ]

            categories_data = [
                {"name": category, "count": category_counts[category], "slug": slugify(category)}
                for category in sorted(all_categories)
            ]

            # Write taxonomy files
            taxonomies_dir = self.output_base / self.config.taxonomies_subdir

            with open(taxonomies_dir / "tags.json", "w", encoding="utf-8") as f:
                json.dump(tags_data, f, indent=2, ensure_ascii=False)

            with open(taxonomies_dir / "categories.json", "w", encoding="utf-8") as f:
                json.dump(categories_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Updated taxonomies: {len(tags_data)} tags, {len(categories_data)} categories"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating taxonomies: {e}")
            return False

    async def cleanup_orphaned_files(self, session: AsyncSession) -> int:
        """Remove markdown files for articles that no longer exist or are unpublished."""
        try:
            # Get all published article IDs
            query = select(Article.id).where(
                Article.status == ContentStatus.COMPLETED, Article.published_at.is_not(None)
            )
            result = await session.execute(query)
            published_ids = set(str(row[0]) for row in result.fetchall())

            # Check existing markdown files
            orphaned_count = 0

            for subdir in [self.config.articles_subdir, self.config.glossary_subdir]:
                subdir_path = self.output_base / subdir

                if subdir_path.exists():
                    for md_file in subdir_path.glob("*.md"):
                        # Try to extract article ID from file metadata
                        # This is a simplified approach - in practice you might
                        # want to store this mapping in the database
                        file_id = self._extract_article_id_from_file(md_file)

                        if file_id and file_id not in published_ids:
                            md_file.unlink()
                            orphaned_count += 1
                            logger.info(f"Removed orphaned file: {md_file}")

            logger.info(f"Cleaned up {orphaned_count} orphaned files")
            return orphaned_count

        except Exception as e:
            logger.error(f"Error cleaning up orphaned files: {e}")
            return 0

    def _extract_article_id_from_file(self, file_path: Path) -> Optional[str]:
        """Extract article ID from markdown file frontmatter."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract frontmatter
            if content.startswith("---\n"):
                end_idx = content.find("\n---\n", 4)
                if end_idx != -1:
                    frontmatter_text = content[4:end_idx]
                    frontmatter = yaml.safe_load(frontmatter_text)

                    # Look for ID in various places
                    return (
                        frontmatter.get("id")
                        or frontmatter.get("source", {}).get("id")
                        or frontmatter.get("articleId")
                    )

            return None

        except Exception:
            return None
