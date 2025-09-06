"""
Unit tests for content publishing components.

Tests Markdown generation, frontmatter creation, and content output formatting.
"""

import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch
import yaml

from pipelines.publish.markdown_generator import MarkdownGenerator


class TestMarkdownGenerator:
    """Test Markdown generation and content publishing."""
    
    @pytest.fixture
    def markdown_generator(self):
        """Create MarkdownGenerator instance for testing."""
        return MarkdownGenerator()
    
    @pytest.fixture
    def sample_article_data(self):
        """Sample article data for publishing tests."""
        return {
            'id': 'ai-fundamentals-2024',
            'title': 'Artificial Intelligence Fundamentals',
            'slug': 'ai-fundamentals-2024',
            'content': """
            # Introduction to AI
            
            Artificial intelligence represents a revolutionary approach to problem-solving in computer science.
            
            ## Core Concepts
            
            - **Machine Learning**: Algorithms that learn from data
            - **Neural Networks**: Computational models inspired by biological neurons
            - **Deep Learning**: Multi-layered neural networks for complex pattern recognition
            
            ## Applications
            
            AI finds applications across numerous domains:
            
            1. Healthcare diagnostics
            2. Financial analysis
            3. Autonomous vehicles
            4. Natural language processing
            
            > "The future of AI lies in augmenting human intelligence, not replacing it."
            
            For more information, visit [our ML guide](/articles/machine-learning-guide).
            """,
            'summary': 'A comprehensive introduction to artificial intelligence fundamentals, covering core concepts and real-world applications.',
            'author': 'Dr. Sarah Johnson',
            'published_date': datetime(2024, 1, 15, 10, 30, 0),
            'updated_date': datetime(2024, 1, 20, 14, 15, 0),
            'category': 'artificial-intelligence',
            'tags': ['ai', 'machine-learning', 'fundamentals', 'introduction'],
            'reading_time': 5,
            'word_count': 234,
            'difficulty': 'beginner',
            'language': 'en',
            'featured_image': '/images/ai-fundamentals.jpg',
            'related_articles': [
                {'id': 'ml-basics', 'title': 'Machine Learning Basics', 'url': '/articles/ml-basics'},
                {'id': 'neural-networks', 'title': 'Neural Networks Guide', 'url': '/articles/neural-networks'}
            ],
            'key_points': [
                'AI is transforming multiple industries',
                'Machine learning enables automated learning from data',
                'Applications span from healthcare to autonomous vehicles'
            ],
            'seo_metadata': {
                'meta_description': 'Learn AI fundamentals including machine learning, neural networks, and real-world applications in this comprehensive guide.',
                'keywords': ['artificial intelligence', 'machine learning', 'AI basics', 'neural networks'],
                'canonical_url': 'https://ai-knowledge.org/articles/ai-fundamentals-2024'
            }
        }
    
    @pytest.mark.unit
    def test_generate_article_markdown(self, markdown_generator, sample_article_data):
        """Test complete article Markdown generation."""
        markdown_content = markdown_generator.generate_article_markdown(sample_article_data)
        
        # Check frontmatter section
        assert markdown_content.startswith('---')
        frontmatter_end = markdown_content.find('---', 3)
        assert frontmatter_end > 0
        
        frontmatter_yaml = markdown_content[3:frontmatter_end].strip()
        frontmatter = yaml.safe_load(frontmatter_yaml)
        
        # Verify frontmatter content
        assert frontmatter['title'] == 'Artificial Intelligence Fundamentals'
        assert frontmatter['slug'] == 'ai-fundamentals-2024'
        assert frontmatter['category'] == 'artificial-intelligence'
        assert 'ai' in frontmatter['tags']
        assert frontmatter['author'] == 'Dr. Sarah Johnson'
        assert frontmatter['readingTime'] == 5
        assert frontmatter['wordCount'] == 234
        assert frontmatter['difficulty'] == 'beginner'
        
        # Check content section
        content_section = markdown_content[frontmatter_end + 3:].strip()
        assert '# Introduction to AI' in content_section
        assert 'Machine Learning' in content_section
        assert 'Neural Networks' in content_section
        assert 'Healthcare diagnostics' in content_section
        assert '[our ML guide](/articles/machine-learning-guide)' in content_section
    
    @pytest.mark.unit
    def test_generate_frontmatter(self, markdown_generator, sample_article_data):
        """Test frontmatter generation."""
        frontmatter = markdown_generator._generate_frontmatter(sample_article_data)
        
        # Check required fields
        assert frontmatter['title'] == 'Artificial Intelligence Fundamentals'
        assert frontmatter['slug'] == 'ai-fundamentals-2024'
        assert frontmatter['publishDate'] == '2024-01-15T10:30:00'
        assert frontmatter['updateDate'] == '2024-01-20T14:15:00'
        assert frontmatter['category'] == 'artificial-intelligence'
        assert frontmatter['tags'] == ['ai', 'machine-learning', 'fundamentals', 'introduction']
        
        # Check optional fields
        assert frontmatter['author'] == 'Dr. Sarah Johnson'
        assert frontmatter['summary'] == sample_article_data['summary']
        assert frontmatter['readingTime'] == 5
        assert frontmatter['wordCount'] == 234
        assert frontmatter['difficulty'] == 'beginner'
        assert frontmatter['language'] == 'en'
        assert frontmatter['featuredImage'] == '/images/ai-fundamentals.jpg'
        
        # Check SEO metadata
        assert frontmatter['seo']['metaDescription'] == sample_article_data['seo_metadata']['meta_description']
        assert frontmatter['seo']['keywords'] == sample_article_data['seo_metadata']['keywords']
        assert frontmatter['seo']['canonicalUrl'] == sample_article_data['seo_metadata']['canonical_url']
        
        # Check related articles
        assert len(frontmatter['relatedArticles']) == 2
        assert frontmatter['relatedArticles'][0]['title'] == 'Machine Learning Basics'
        
        # Check key points
        assert len(frontmatter['keyPoints']) == 3
        assert 'AI is transforming multiple industries' in frontmatter['keyPoints']
    
    @pytest.mark.unit
    def test_generate_minimal_frontmatter(self, markdown_generator):
        """Test frontmatter generation with minimal data."""
        minimal_data = {
            'title': 'Simple Article',
            'slug': 'simple-article',
            'content': 'Simple content',
            'category': 'general',
            'published_date': datetime(2024, 1, 1, 12, 0, 0)
        }
        
        frontmatter = markdown_generator._generate_frontmatter(minimal_data)
        
        # Check required fields are present
        assert frontmatter['title'] == 'Simple Article'
        assert frontmatter['slug'] == 'simple-article'
        assert frontmatter['publishDate'] == '2024-01-01T12:00:00'
        assert frontmatter['category'] == 'general'
        
        # Optional fields should have defaults or be omitted
        assert frontmatter.get('tags', []) == []
        assert frontmatter.get('language') == 'en'
        assert 'author' not in frontmatter or frontmatter['author'] is None
    
    @pytest.mark.unit
    def test_format_content_links(self, markdown_generator):
        """Test internal link formatting in content."""
        content_with_links = """
        Check out our [machine learning guide](ml-guide) for more details.
        Also see the [neural networks tutorial](neural-networks-tutorial).
        External links like [OpenAI](https://openai.com) should remain unchanged.
        """
        
        formatted_content = markdown_generator._format_content_links(content_with_links)
        
        # Internal links should be formatted with /articles/ prefix
        assert '[machine learning guide](/articles/ml-guide)' in formatted_content
        assert '[neural networks tutorial](/articles/neural-networks-tutorial)' in formatted_content
        
        # External links should remain unchanged
        assert '[OpenAI](https://openai.com)' in formatted_content
    
    @pytest.mark.unit
    def test_generate_glossary_entry(self, markdown_generator):
        """Test glossary entry Markdown generation."""
        glossary_data = {
            'term': 'Machine Learning',
            'slug': 'machine-learning',
            'definition': 'A subset of artificial intelligence that enables systems to learn from data.',
            'detailed_explanation': """
            Machine learning algorithms build mathematical models based on training data to make 
            predictions or decisions without being explicitly programmed for the task.
            """,
            'category': 'core-concepts',
            'tags': ['ml', 'ai', 'algorithms'],
            'related_terms': [
                {'term': 'Artificial Intelligence', 'slug': 'artificial-intelligence'},
                {'term': 'Deep Learning', 'slug': 'deep-learning'}
            ],
            'examples': [
                'Email spam detection',
                'Image recognition',
                'Recommendation systems'
            ],
            'created_date': datetime(2024, 1, 10, 9, 0, 0),
            'updated_date': datetime(2024, 1, 15, 16, 30, 0)
        }
        
        markdown_content = markdown_generator.generate_glossary_entry(glossary_data)
        
        # Check structure
        assert markdown_content.startswith('---')
        assert 'term: Machine Learning' in markdown_content
        assert 'slug: machine-learning' in markdown_content
        assert 'category: core-concepts' in markdown_content
        assert 'ml' in markdown_content
        
        # Check content sections
        assert '## Definition' in markdown_content
        assert 'subset of artificial intelligence' in markdown_content
        assert '## Examples' in markdown_content
        assert 'Email spam detection' in markdown_content
        assert '## Related Terms' in markdown_content
        assert '[Artificial Intelligence](/glossary/artificial-intelligence)' in markdown_content
    
    @pytest.mark.unit
    def test_generate_taxonomy_files(self, markdown_generator, temp_directory):
        """Test taxonomy file generation."""
        categories = {
            'artificial-intelligence': {
                'name': 'Artificial Intelligence',
                'description': 'Core AI concepts and techniques',
                'color': '#4F46E5',
                'icon': 'brain',
                'article_count': 15
            },
            'machine-learning': {
                'name': 'Machine Learning',
                'description': 'ML algorithms and applications',
                'color': '#059669',
                'icon': 'chart-line',
                'article_count': 23
            }
        }
        
        tags = {
            'ai': {
                'name': 'AI',
                'description': 'Artificial Intelligence related content',
                'article_count': 45
            },
            'neural-networks': {
                'name': 'Neural Networks',
                'description': 'Neural network architectures and applications',
                'article_count': 12
            }
        }
        
        # Generate taxonomy files
        categories_path = temp_directory / 'categories.json'
        tags_path = temp_directory / 'tags.json'
        
        markdown_generator.generate_taxonomy_files(categories, tags, str(temp_directory))
        
        # Check files were created
        assert categories_path.exists()
        assert tags_path.exists()
        
        # Check content
        import json
        with open(categories_path) as f:
            saved_categories = json.load(f)
        with open(tags_path) as f:
            saved_tags = json.load(f)
        
        assert saved_categories == categories
        assert saved_tags == tags
    
    @pytest.mark.unit
    def test_validate_markdown_output(self, markdown_generator, sample_article_data):
        """Test validation of generated Markdown."""
        markdown_content = markdown_generator.generate_article_markdown(sample_article_data)
        
        validation_result = markdown_generator.validate_markdown_output(markdown_content)
        
        assert validation_result['valid'] is True
        assert validation_result['errors'] == []
        assert validation_result['warnings'] == []
        
        # Check structure validation
        assert validation_result['has_frontmatter'] is True
        assert validation_result['has_content'] is True
        assert validation_result['word_count'] > 0
    
    @pytest.mark.unit
    def test_validate_invalid_markdown(self, markdown_generator):
        """Test validation of invalid Markdown."""
        invalid_markdown = """
        Invalid frontmatter without proper YAML
        title: Missing quotes and structure
        ---
        Some content here
        """
        
        validation_result = markdown_generator.validate_markdown_output(invalid_markdown)
        
        assert validation_result['valid'] is False
        assert len(validation_result['errors']) > 0
        assert 'frontmatter' in validation_result['errors'][0].lower()
    
    @pytest.mark.unit
    def test_generate_content_index(self, markdown_generator, temp_directory):
        """Test content index generation for navigation."""
        articles = [
            {
                'id': 'ai-basics',
                'title': 'AI Basics',
                'slug': 'ai-basics',
                'category': 'artificial-intelligence',
                'tags': ['ai', 'basics'],
                'published_date': datetime(2024, 1, 15),
                'summary': 'Introduction to AI concepts',
                'reading_time': 5,
                'difficulty': 'beginner'
            },
            {
                'id': 'ml-advanced',
                'title': 'Advanced ML',
                'slug': 'ml-advanced',
                'category': 'machine-learning',
                'tags': ['ml', 'advanced'],
                'published_date': datetime(2024, 1, 20),
                'summary': 'Advanced ML techniques',
                'reading_time': 15,
                'difficulty': 'advanced'
            }
        ]
        
        index_path = temp_directory / 'content-index.json'
        markdown_generator.generate_content_index(articles, str(index_path))
        
        assert index_path.exists()
        
        import json
        with open(index_path) as f:
            content_index = json.load(f)
        
        assert len(content_index['articles']) == 2
        assert content_index['metadata']['total_articles'] == 2
        assert 'artificial-intelligence' in content_index['metadata']['categories']
        assert 'machine-learning' in content_index['metadata']['categories']
        assert 'generated_at' in content_index['metadata']
    
    @pytest.mark.unit
    def test_clean_content_for_markdown(self, markdown_generator):
        """Test content cleaning for Markdown output."""
        dirty_content = """
        <div class="content">
            <p>This is a paragraph with <span style="color: red;">colored text</span>.</p>
            <script>alert('xss');</script>
            <a href="javascript:void(0)">Dangerous link</a>
            <img src="image.jpg" onerror="alert('xss')" alt="Image">
        </div>
        """
        
        cleaned_content = markdown_generator._clean_content_for_markdown(dirty_content)
        
        # Should remove dangerous elements
        assert '<script>' not in cleaned_content
        assert 'javascript:' not in cleaned_content
        assert 'onerror=' not in cleaned_content
        assert 'alert(' not in cleaned_content
        
        # Should preserve safe content
        assert 'This is a paragraph' in cleaned_content
        assert 'alt="Image"' in cleaned_content or 'Image' in cleaned_content
    
    @pytest.mark.unit
    def test_generate_sitemap_data(self, markdown_generator):
        """Test sitemap data generation."""
        articles = [
            {
                'slug': 'ai-basics',
                'published_date': datetime(2024, 1, 15),
                'updated_date': datetime(2024, 1, 20),
                'category': 'ai',
                'priority': 0.8
            },
            {
                'slug': 'ml-guide',
                'published_date': datetime(2024, 1, 10),
                'updated_date': datetime(2024, 1, 25),
                'category': 'ml',
                'priority': 0.9
            }
        ]
        
        sitemap_data = markdown_generator.generate_sitemap_data(
            articles,
            base_url='https://example.com'
        )
        
        assert len(sitemap_data) == 2
        
        # Check first entry
        first_entry = sitemap_data[0]
        assert first_entry['url'] == 'https://example.com/articles/ai-basics'
        assert first_entry['lastmod'] == '2024-01-20'
        assert first_entry['priority'] == 0.8
        assert first_entry['changefreq'] == 'monthly'
    
    @pytest.mark.unit
    def test_batch_markdown_generation(self, markdown_generator, temp_directory):
        """Test batch generation of multiple articles."""
        articles_data = []
        for i in range(5):
            articles_data.append({
                'id': f'article_{i}',
                'title': f'Article {i}',
                'slug': f'article-{i}',
                'content': f'Content for article {i}',
                'category': 'test',
                'published_date': datetime(2024, 1, i+1),
                'tags': [f'tag-{i}']
            })
        
        output_dir = temp_directory / 'articles'
        output_dir.mkdir()
        
        # Generate all articles
        results = markdown_generator.batch_generate_articles(
            articles_data,
            str(output_dir)
        )
        
        assert len(results) == 5
        assert all(result['success'] for result in results)
        
        # Check files were created
        for i in range(5):
            article_file = output_dir / f'article-{i}.md'
            assert article_file.exists()
            
            # Check content
            content = article_file.read_text()
            assert f'title: Article {i}' in content
            assert f'slug: article-{i}' in content
            assert f'Content for article {i}' in content
    
    @pytest.mark.unit
    def test_error_handling_invalid_data(self, markdown_generator):
        """Test error handling with invalid article data."""
        invalid_data = {
            'title': None,  # Invalid title
            'content': '',   # Empty content
            'published_date': 'invalid-date'  # Invalid date format
        }
        
        # Should handle errors gracefully
        try:
            markdown_content = markdown_generator.generate_article_markdown(invalid_data)
            # If no exception, check that defaults are applied
            assert 'title:' in markdown_content
        except Exception as e:
            # Should raise descriptive error
            assert 'title' in str(e).lower() or 'required' in str(e).lower()


class TestMarkdownValidation:
    """Test Markdown validation and quality checks."""
    
    @pytest.fixture
    def markdown_validator(self):
        """Create markdown validator for testing."""
        from pipelines.publish.markdown_generator import MarkdownGenerator
        return MarkdownGenerator()
    
    @pytest.mark.unit
    def test_frontmatter_validation(self, markdown_validator):
        """Test frontmatter validation."""
        valid_frontmatter = """---
title: Test Article
slug: test-article
category: test
publishDate: 2024-01-15T10:00:00
---

Content here."""
        
        result = markdown_validator.validate_markdown_output(valid_frontmatter)
        assert result['valid'] is True
        assert result['has_frontmatter'] is True
        
        # Invalid YAML
        invalid_frontmatter = """---
title: Test Article
invalid: yaml: structure:
---

Content here."""
        
        result = markdown_validator.validate_markdown_output(invalid_frontmatter)
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    @pytest.mark.unit
    def test_content_quality_checks(self, markdown_validator):
        """Test content quality validation."""
        # Good quality content
        good_content = """---
title: Quality Article
slug: quality-article
category: test
---

# Introduction

This is a well-structured article with multiple paragraphs, proper headings, and good content depth.

## Section 1

Content with sufficient detail and proper formatting.

## Section 2

More detailed content that provides value to readers."""
        
        result = markdown_validator.validate_markdown_output(good_content)
        quality_score = result.get('quality_score', 0)
        assert quality_score > 0.7
        
        # Poor quality content
        poor_content = """---
title: Poor Article
slug: poor-article
category: test
---

Short content."""
        
        result = markdown_validator.validate_markdown_output(poor_content)
        quality_score = result.get('quality_score', 1)
        assert quality_score < 0.5
    
    @pytest.mark.unit
    def test_link_validation(self, markdown_validator):
        """Test internal and external link validation."""
        content_with_links = """---
title: Article with Links
slug: article-links
category: test
---

Check out [our guide](/articles/guide) and [external site](https://example.com).
Also see [broken link](broken-url) and [missing internal](/articles/missing)."""
        
        validation_result = markdown_validator.validate_markdown_output(content_with_links)
        
        # Should identify potential link issues
        if 'link_issues' in validation_result:
            link_issues = validation_result['link_issues']
            assert len(link_issues) > 0  # Should find some issues with broken/missing links