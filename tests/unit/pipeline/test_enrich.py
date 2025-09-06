"""
Unit tests for content enrichment components.

Tests AI-powered summarization, cross-linking, and content enhancement.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

from pipelines.enrich.summarizer import ContentSummarizer
from pipelines.enrich.cross_linker import CrossLinker


class TestContentSummarizer:
    """Test AI-powered content summarization."""
    
    @pytest.fixture
    def content_summarizer(self):
        """Create ContentSummarizer instance for testing."""
        return ContentSummarizer()
    
    @pytest.fixture
    def long_article_content(self):
        """Long article content for summarization testing."""
        return """
        Artificial intelligence has emerged as one of the most transformative technologies of the 21st century, 
        fundamentally changing how we approach complex problems across numerous industries. From healthcare and 
        finance to transportation and entertainment, AI systems are demonstrating capabilities that were once 
        considered the exclusive domain of human intelligence.
        
        Machine learning, a subset of artificial intelligence, enables systems to learn and improve from 
        experience without being explicitly programmed for every task. This approach has proven particularly 
        effective in areas where creating explicit rules would be extremely difficult or impossible. Consider 
        image recognition: writing code to identify objects in photographs by manually specifying all possible 
        visual features would be an insurmountable challenge.
        
        Deep learning represents the next evolution in machine learning, using neural networks with multiple 
        layers to model complex patterns in data. These systems can process vast amounts of information, 
        identifying subtle relationships and patterns that would be invisible to traditional analytical methods. 
        The success of deep learning has been particularly evident in natural language processing, computer 
        vision, and speech recognition applications.
        
        However, the rapid advancement of AI technology also raises important ethical and societal questions. 
        Issues of bias, fairness, transparency, and accountability have become central concerns as AI systems 
        increasingly influence critical decisions in areas such as hiring, lending, criminal justice, and 
        healthcare. Ensuring that AI development proceeds responsibly requires careful consideration of these 
        factors throughout the design and deployment process.
        
        The future of artificial intelligence lies not in replacing human intelligence, but in augmenting it. 
        The most successful AI applications combine the computational power and pattern recognition capabilities 
        of machines with human creativity, intuition, and ethical judgment. This collaborative approach promises 
        to unlock new levels of productivity and innovation across all sectors of society.
        
        As we continue to advance AI technology, it is crucial that we maintain focus on developing systems 
        that are not only powerful and efficient, but also transparent, fair, and aligned with human values. 
        The decisions we make today about AI development and governance will shape the technology landscape 
        for generations to come.
        """
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_summary_openai(self, content_summarizer, long_article_content, mock_openai_client):
        """Test summary generation using OpenAI."""
        with patch.object(content_summarizer, 'openai_client', mock_openai_client):
            summary = await content_summarizer.generate_summary(
                content=long_article_content,
                max_length=200,
                provider='openai'
            )
            
            assert summary is not None
            assert len(summary) > 0
            assert len(summary) <= 250  # Should respect length limit with some tolerance
            assert 'artificial intelligence' in summary.lower() or 'ai' in summary.lower()
            
            # Check that OpenAI client was called
            mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_summary_anthropic(self, content_summarizer, long_article_content):
        """Test summary generation using Anthropic Claude."""
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.content = [AsyncMock(text="This is a test summary of the AI article.")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client
            
            content_summarizer.anthropic_client = mock_client
            
            summary = await content_summarizer.generate_summary(
                content=long_article_content,
                max_length=150,
                provider='anthropic'
            )
            
            assert summary == "This is a test summary of the AI article."
            mock_client.messages.create.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_summary_extractive(self, content_summarizer, long_article_content):
        """Test extractive summarization fallback."""
        summary = await content_summarizer.generate_summary(
            content=long_article_content,
            max_length=200,
            provider='extractive'
        )
        
        assert summary is not None
        assert len(summary) > 0
        assert len(summary) <= 250  # Should respect length limit
        
        # Extractive summary should contain sentences from original content
        original_sentences = [s.strip() for s in long_article_content.split('.') if s.strip()]
        summary_words = set(summary.lower().split())
        original_words = set(long_article_content.lower().split())
        
        # Should have significant word overlap
        word_overlap = len(summary_words.intersection(original_words))
        assert word_overlap > len(summary_words) * 0.7  # At least 70% word overlap
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_summary_short_content(self, content_summarizer):
        """Test summarization with content that's already short."""
        short_content = "This is a very short article about AI."
        
        summary = await content_summarizer.generate_summary(
            content=short_content,
            max_length=100,
            provider='openai'
        )
        
        # Short content should return the original or a simple summary
        assert summary is not None
        assert len(summary) > 0
        assert 'ai' in summary.lower() or 'artificial intelligence' in summary.lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_key_points(self, content_summarizer, long_article_content, mock_openai_client):
        """Test key points extraction."""
        with patch.object(content_summarizer, 'openai_client', mock_openai_client):
            # Mock response with key points
            mock_openai_client.chat.completions.create.return_value = AsyncMock(
                choices=[
                    AsyncMock(
                        message=AsyncMock(
                            content="• AI is transforming multiple industries\n• Machine learning enables automated learning\n• Deep learning uses neural networks\n• Ethical considerations are important\n• Future involves human-AI collaboration"
                        )
                    )
                ]
            )
            
            key_points = await content_summarizer.generate_key_points(
                content=long_article_content,
                max_points=5
            )
            
            assert isinstance(key_points, list)
            assert len(key_points) <= 5
            assert all(isinstance(point, str) for point in key_points)
            assert len(key_points) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enhance_metadata(self, content_summarizer, long_article_content, mock_openai_client):
        """Test metadata enhancement with AI."""
        original_metadata = {
            'title': 'AI Article',
            'category': 'technology',
            'tags': ['ai']
        }
        
        with patch.object(content_summarizer, 'openai_client', mock_openai_client):
            # Mock enhanced metadata response
            mock_openai_client.chat.completions.create.return_value = AsyncMock(
                choices=[
                    AsyncMock(
                        message=AsyncMock(
                            content='{"suggested_tags": ["artificial-intelligence", "machine-learning", "deep-learning", "ethics"], "category": "artificial-intelligence", "complexity_level": "intermediate", "target_audience": "technical"}'
                        )
                    )
                ]
            )
            
            enhanced_metadata = await content_summarizer.enhance_metadata(
                content=long_article_content,
                existing_metadata=original_metadata
            )
            
            assert 'suggested_tags' in enhanced_metadata
            assert 'category' in enhanced_metadata
            assert 'complexity_level' in enhanced_metadata
            assert 'target_audience' in enhanced_metadata
            
            # Should include AI-related tags
            suggested_tags = enhanced_metadata['suggested_tags']
            assert any('ai' in tag.lower() or 'intelligence' in tag.lower() for tag in suggested_tags)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, content_summarizer, long_article_content):
        """Test error handling when API calls fail."""
        with patch.object(content_summarizer, 'openai_client') as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            # Should fallback to extractive summarization
            summary = await content_summarizer.generate_summary(
                content=long_article_content,
                max_length=200,
                provider='openai'
            )
            
            assert summary is not None
            assert len(summary) > 0
            # Should be extractive summary (contains original content words)
    
    @pytest.mark.unit
    def test_extractive_summary_sentence_scoring(self, content_summarizer, long_article_content):
        """Test sentence scoring for extractive summarization."""
        sentences = content_summarizer._split_into_sentences(long_article_content)
        scores = content_summarizer._score_sentences(sentences)
        
        assert len(scores) == len(sentences)
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)
        
        # Sentences with important terms should have higher scores
        important_sentence_indices = [
            i for i, sentence in enumerate(sentences)
            if 'artificial intelligence' in sentence.lower() or 'machine learning' in sentence.lower()
        ]
        
        if important_sentence_indices:
            avg_important_score = sum(scores[i] for i in important_sentence_indices) / len(important_sentence_indices)
            avg_all_scores = sum(scores) / len(scores)
            
            # Important sentences should generally score higher than average
            assert avg_important_score >= avg_all_scores


class TestCrossLinker:
    """Test cross-linking and relationship detection."""
    
    @pytest.fixture
    def cross_linker(self):
        """Create CrossLinker instance for testing."""
        return CrossLinker()
    
    @pytest.fixture
    def article_database(self):
        """Mock article database for cross-linking tests."""
        return [
            {
                'id': 'ml_basics',
                'title': 'Machine Learning Fundamentals',
                'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
                'url': '/articles/ml-basics',
                'tags': ['machine-learning', 'ai', 'fundamentals'],
                'category': 'machine-learning'
            },
            {
                'id': 'deep_learning',
                'title': 'Introduction to Deep Learning',
                'content': 'Deep learning uses neural networks with multiple layers to process complex data.',
                'url': '/articles/deep-learning',
                'tags': ['deep-learning', 'neural-networks', 'ai'],
                'category': 'deep-learning'
            },
            {
                'id': 'nlp_guide',
                'title': 'Natural Language Processing Guide',
                'content': 'Natural language processing enables computers to understand and process human language.',
                'url': '/articles/nlp-guide',
                'tags': ['nlp', 'language', 'ai'],
                'category': 'nlp'
            },
            {
                'id': 'ai_ethics',
                'title': 'AI Ethics and Fairness',
                'content': 'Ethical considerations in artificial intelligence development and deployment.',
                'url': '/articles/ai-ethics',
                'tags': ['ethics', 'fairness', 'ai'],
                'category': 'ethics'
            }
        ]
    
    @pytest.mark.unit
    async def test_find_related_articles(self, cross_linker, article_database):
        """Test finding related articles based on content similarity."""
        current_article = {
            'id': 'new_ai_article',
            'title': 'Advanced AI Techniques',
            'content': 'This article discusses machine learning algorithms and neural networks for artificial intelligence applications.',
            'tags': ['ai', 'algorithms'],
            'category': 'machine-learning'
        }
        
        related_articles = await cross_linker.find_related_articles(
            current_article,
            article_database,
            max_related=3
        )
        
        assert len(related_articles) <= 3
        assert all('similarity_score' in article for article in related_articles)
        assert all(article['similarity_score'] > 0 for article in related_articles)
        
        # Should find ML and deep learning articles as most related
        related_ids = [article['id'] for article in related_articles]
        assert 'ml_basics' in related_ids
        assert 'deep_learning' in related_ids
        
        # Results should be sorted by similarity score (highest first)
        scores = [article['similarity_score'] for article in related_articles]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.unit
    async def test_generate_internal_links(self, cross_linker, article_database):
        """Test generation of internal links within content."""
        content = """
        Machine learning is a powerful technique in artificial intelligence.
        Neural networks are particularly effective for deep learning applications.
        Natural language processing helps computers understand human language.
        """
        
        linked_content = await cross_linker.generate_internal_links(
            content,
            article_database,
            max_links=5
        )
        
        # Should contain markdown links
        assert '[machine learning]' in linked_content or '[Machine learning]' in linked_content
        assert '(/articles/ml-basics)' in linked_content
        assert '[neural networks]' in linked_content or '[Neural networks]' in linked_content
        assert '(/articles/deep-learning)' in linked_content
        assert '[natural language processing]' in linked_content or '[Natural language processing]' in linked_content
        assert '(/articles/nlp-guide)' in linked_content
    
    @pytest.mark.unit
    def test_calculate_content_similarity(self, cross_linker):
        """Test content similarity calculation."""
        content1 = "Machine learning algorithms enable computers to learn from data without explicit programming."
        content2 = "Machine learning techniques allow systems to learn patterns from data automatically."
        content3 = "Natural language processing focuses on understanding human language with computers."
        
        # High similarity between content1 and content2
        similarity_high = cross_linker._calculate_content_similarity(content1, content2)
        assert similarity_high > 0.5
        
        # Lower similarity between content1 and content3
        similarity_low = cross_linker._calculate_content_similarity(content1, content3)
        assert similarity_low < similarity_high
        
        # Similarity should be symmetric
        similarity_reverse = cross_linker._calculate_content_similarity(content2, content1)
        assert abs(similarity_high - similarity_reverse) < 0.01
    
    @pytest.mark.unit
    def test_calculate_tag_similarity(self, cross_linker):
        """Test tag-based similarity calculation."""
        tags1 = ['machine-learning', 'ai', 'algorithms']
        tags2 = ['machine-learning', 'deep-learning', 'ai']
        tags3 = ['nlp', 'language', 'processing']
        
        # Moderate similarity (shared 'machine-learning' and 'ai')
        similarity_moderate = cross_linker._calculate_tag_similarity(tags1, tags2)
        assert 0.3 < similarity_moderate < 0.8
        
        # Low similarity (no shared tags)
        similarity_low = cross_linker._calculate_tag_similarity(tags1, tags3)
        assert similarity_low < 0.3
        
        # Perfect similarity (identical tags)
        similarity_perfect = cross_linker._calculate_tag_similarity(tags1, tags1)
        assert similarity_perfect == 1.0
    
    @pytest.mark.unit
    def test_extract_linkable_terms(self, cross_linker, article_database):
        """Test extraction of terms that can be linked."""
        content = """
        Machine learning is a subset of artificial intelligence.
        Deep learning neural networks process complex patterns.
        Natural language processing enables human-computer interaction.
        """
        
        linkable_terms = cross_linker._extract_linkable_terms(content, article_database)
        
        assert len(linkable_terms) > 0
        
        # Should find terms that match article titles/content
        term_texts = [term['text'].lower() for term in linkable_terms]
        assert any('machine learning' in term for term in term_texts)
        assert any('deep learning' in term for term in term_texts)
        assert any('natural language processing' in term for term in term_texts)
        
        # Each term should have position and target URL
        for term in linkable_terms:
            assert 'text' in term
            assert 'start_pos' in term
            assert 'end_pos' in term
            assert 'target_url' in term
            assert 'target_title' in term
    
    @pytest.mark.unit
    async def test_suggest_tags(self, cross_linker, article_database):
        """Test tag suggestions based on content and related articles."""
        content = """
        This article explores advanced neural network architectures for computer vision tasks.
        Convolutional neural networks excel at image recognition and processing.
        """
        
        suggested_tags = await cross_linker.suggest_tags(
            content,
            existing_tags=['neural-networks'],
            article_database=article_database,
            max_suggestions=5
        )
        
        assert len(suggested_tags) <= 5
        assert all(isinstance(tag, str) for tag in suggested_tags)
        
        # Should suggest relevant tags based on content
        suggested_lower = [tag.lower() for tag in suggested_tags]
        assert any('computer-vision' in tag or 'vision' in tag for tag in suggested_lower)
        assert any('cnn' in tag or 'convolutional' in tag for tag in suggested_lower)
    
    @pytest.mark.unit
    async def test_create_content_map(self, cross_linker, article_database):
        """Test creation of content relationship map."""
        content_map = await cross_linker.create_content_map(
            article_database,
            similarity_threshold=0.3
        )
        
        assert isinstance(content_map, dict)
        
        # Each article should have relationships
        for article_id in content_map:
            relationships = content_map[article_id]
            assert isinstance(relationships, list)
            
            # Each relationship should have required fields
            for rel in relationships:
                assert 'target_id' in rel
                assert 'similarity_score' in rel
                assert 'relationship_type' in rel
                assert rel['similarity_score'] >= 0.3
    
    @pytest.mark.unit
    def test_avoid_duplicate_links(self, cross_linker, article_database):
        """Test that duplicate internal links are avoided."""
        content = """
        Machine learning is important. Machine learning algorithms are powerful.
        Machine learning applications are everywhere in machine learning systems.
        """
        
        linked_content = cross_linker._generate_internal_links_sync(
            content,
            article_database,
            max_links=3
        )
        
        # Count occurrences of the same link
        ml_link_count = linked_content.count('[machine learning](/articles/ml-basics)')
        ml_link_count += linked_content.count('[Machine learning](/articles/ml-basics)')
        
        # Should not create more than 2-3 links to the same article
        assert ml_link_count <= 3
        
        # Should still contain some unlinked occurrences
        unlinked_ml_count = linked_content.count('machine learning') + linked_content.count('Machine learning')
        assert unlinked_ml_count > 0
    
    @pytest.mark.unit
    async def test_performance_with_large_database(self, cross_linker, performance_benchmarks):
        """Test cross-linking performance with large article database."""
        # Create large article database
        large_database = []
        for i in range(1000):
            large_database.append({
                'id': f'article_{i}',
                'title': f'Article {i} on AI Topic {i % 10}',
                'content': f'This is article {i} about artificial intelligence topic {i % 10}.',
                'url': f'/articles/article-{i}',
                'tags': [f'tag-{i % 5}', 'ai'],
                'category': f'category-{i % 3}'
            })
        
        test_article = {
            'id': 'test_article',
            'title': 'Test Article',
            'content': 'This is a test article about artificial intelligence and machine learning.',
            'tags': ['ai', 'test'],
            'category': 'test'
        }
        
        import time
        start_time = time.time()
        
        related_articles = await cross_linker.find_related_articles(
            test_article,
            large_database,
            max_related=10
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        max_time = performance_benchmarks.get("cross_linking_time_per_1000_articles", 2.0)
        assert duration <= max_time, f"Cross-linking too slow: {duration:.2f}s > {max_time}s"
        
        # Should return results
        assert len(related_articles) <= 10
        assert len(related_articles) > 0