"""
Unit tests for content normalization components.

Tests content extraction, HTML cleaning, and content quality analysis.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.normalize.html_cleaner import HTMLCleaner


class TestContentExtractor:
    """Test content extraction functionality."""
    
    @pytest.fixture
    def content_extractor(self):
        """Create ContentExtractor instance for testing."""
        return ContentExtractor()
    
    @pytest.fixture
    def complex_html_content(self):
        """Complex HTML content for testing extraction."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Advanced AI Techniques in Modern Applications</title>
            <meta name="description" content="Exploring cutting-edge AI techniques and their real-world applications in various industries.">
            <meta name="author" content="Dr. Jane Smith">
            <meta name="keywords" content="artificial intelligence, machine learning, deep learning, applications">
            <meta property="article:published_time" content="2024-01-15T10:30:00Z">
            <meta property="og:title" content="Advanced AI Techniques in Modern Applications">
            <meta property="og:description" content="A comprehensive guide to modern AI techniques">
            <link rel="canonical" href="https://example.com/ai-techniques">
        </head>
        <body>
            <nav>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/articles">Articles</a></li>
                </ul>
            </nav>
            <header>
                <h1>Advanced AI Techniques in Modern Applications</h1>
                <p class="meta">
                    Published on <time datetime="2024-01-15T10:30:00Z">January 15, 2024</time> 
                    by <span rel="author">Dr. Jane Smith</span>
                </p>
            </header>
            <main>
                <article>
                    <section>
                        <h2>Introduction</h2>
                        <p>Artificial intelligence has revolutionized numerous industries over the past decade. 
                        This comprehensive guide explores the most significant AI techniques currently transforming 
                        business operations and technological capabilities.</p>
                        
                        <p>From natural language processing to computer vision, modern AI systems demonstrate 
                        remarkable capabilities that were considered science fiction just years ago. Understanding 
                        these techniques is crucial for professionals across various fields.</p>
                    </section>
                    
                    <section>
                        <h2>Core AI Techniques</h2>
                        <p>The foundation of modern AI rests on several key techniques that work together to 
                        create intelligent systems. These include machine learning algorithms, neural networks, 
                        and statistical modeling approaches.</p>
                        
                        <h3>Machine Learning Fundamentals</h3>
                        <p>Machine learning enables systems to learn patterns from data without explicit 
                        programming. This approach has proven particularly effective in areas such as 
                        recommendation systems, fraud detection, and predictive analytics.</p>
                        
                        <p>Supervised learning algorithms use labeled datasets to train models that can make 
                        predictions on new data. Unsupervised learning discovers hidden patterns in unlabeled 
                        data, while reinforcement learning optimizes decision-making through trial and error.</p>
                    </section>
                    
                    <section>
                        <h2>Real-World Applications</h2>
                        <p>AI techniques find applications across numerous industries, from healthcare and 
                        finance to transportation and entertainment. Each application requires careful 
                        consideration of ethical implications and technical constraints.</p>
                        
                        <ul>
                            <li>Healthcare: Diagnostic imaging, drug discovery, personalized treatment</li>
                            <li>Finance: Risk assessment, algorithmic trading, fraud prevention</li>
                            <li>Transportation: Autonomous vehicles, route optimization, traffic management</li>
                            <li>Entertainment: Content recommendation, game AI, creative assistance</li>
                        </ul>
                    </section>
                    
                    <section>
                        <h2>Future Considerations</h2>
                        <p>As AI continues to evolve, practitioners must balance innovation with responsibility. 
                        Ethical AI development requires consideration of bias, fairness, transparency, and 
                        societal impact.</p>
                        
                        <p>The future of AI lies in creating systems that augment human capabilities rather 
                        than replace them entirely. This collaborative approach promises to unlock new levels 
                        of productivity and creativity across all sectors.</p>
                    </section>
                </article>
            </main>
            <footer>
                <p>© 2024 AI Research Journal. All rights reserved.</p>
            </footer>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "Advanced AI Techniques in Modern Applications",
                "author": {
                    "@type": "Person",
                    "name": "Dr. Jane Smith"
                },
                "datePublished": "2024-01-15T10:30:00Z"
            }
            </script>
        </body>
        </html>
        """
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_content_success(self, content_extractor, complex_html_content):
        """Test successful content extraction."""
        url = "https://example.com/ai-techniques"
        
        result = await content_extractor.extract_content(complex_html_content, url)
        
        # Check basic extraction
        assert result['title'] == "Advanced AI Techniques in Modern Applications"
        assert len(result['content']) > 500
        assert 'artificial intelligence' in result['content'].lower()
        
        # Check metadata extraction
        assert result['author'] == "Dr. Jane Smith"
        assert result['meta_description'] == "Exploring cutting-edge AI techniques and their real-world applications in various industries."
        assert result['canonical_url'] == "https://example.com/ai-techniques"
        assert 'artificial intelligence' in result['meta_keywords']
        
        # Check quality metrics
        assert result['word_count'] > 100
        assert result['quality_score'] > 0.5  # Should be high quality content
        assert result['readability_score'] > 0.3
        assert result['sentence_count'] > 5
        assert result['paragraph_count'] > 3
        
        # Check language detection
        assert result['language'] == 'en'
        
        # Check keywords extraction
        assert len(result['keywords']) > 0
        assert any('ai' in kw.lower() or 'intelligence' in kw.lower() for kw in result['keywords'])
        
        # Check structured data
        assert 'structured_data' in result
        assert result['structured_data']['@type'] == 'Article'
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_content_readability_fallback(self, content_extractor):
        """Test fallback to manual cleaning when readability fails."""
        # HTML with minimal content that might cause readability to fail
        minimal_html = """
        <html>
        <head><title>Short Article</title></head>
        <body>
            <h1>Short Article</h1>
            <p>This is very short content.</p>
        </body>
        </html>
        """
        
        with patch.object(content_extractor, '_extract_with_readability', return_value=None):
            result = await content_extractor.extract_content(minimal_html, "https://example.com/short")
            
            assert result['title'] == "Short Article"
            assert result['method'] == 'manual_cleaning'
            assert len(result['content']) > 0
    
    @pytest.mark.unit
    def test_extract_title_multiple_sources(self, content_extractor):
        """Test title extraction from various HTML sources."""
        # Test OpenGraph title
        html_og = '<html><head><meta property="og:title" content="OG Title"></head></html>'
        soup = BeautifulSoup(html_og, 'html.parser')
        title = content_extractor._extract_title(soup)
        assert title == "OG Title"
        
        # Test Twitter title
        html_twitter = '<html><head><meta name="twitter:title" content="Twitter Title"></head></html>'
        soup = BeautifulSoup(html_twitter, 'html.parser')
        title = content_extractor._extract_title(soup)
        assert title == "Twitter Title"
        
        # Test standard title tag
        html_title = '<html><head><title>Standard Title</title></head></html>'
        soup = BeautifulSoup(html_title, 'html.parser')
        title = content_extractor._extract_title(soup)
        assert title == "Standard Title"
        
        # Test H1 fallback
        html_h1 = '<html><body><h1>H1 Title</h1></body></html>'
        soup = BeautifulSoup(html_h1, 'html.parser')
        title = content_extractor._extract_title(soup)
        assert title == "H1 Title"
    
    @pytest.mark.unit
    def test_extract_metadata_comprehensive(self, content_extractor, complex_html_content):
        """Test comprehensive metadata extraction."""
        metadata = content_extractor._extract_metadata(complex_html_content, "https://example.com/test")
        
        # Check all metadata fields
        assert metadata['meta_description'] == "Exploring cutting-edge AI techniques and their real-world applications in various industries."
        assert metadata['author'] == "Dr. Jane Smith"
        assert metadata['canonical_url'] == "https://example.com/ai-techniques"
        assert 'artificial intelligence' in metadata['meta_keywords']
        
        # Check structured data
        assert 'structured_data' in metadata
        assert metadata['structured_data']['@type'] == 'Article'
        assert metadata['structured_data']['author']['name'] == 'Dr. Jane Smith'
        
        # Check publish date
        assert metadata['publish_date'] is not None
        assert metadata['publish_date'].year == 2024
        assert metadata['publish_date'].month == 1
        assert metadata['publish_date'].day == 15
    
    @pytest.mark.unit
    def test_analyze_content_quality_high_quality(self, content_extractor):
        """Test quality analysis for high-quality content."""
        high_quality_content = """
        Machine learning represents a paradigm shift in how we approach problem-solving in computer science. 
        Unlike traditional programming where we explicitly code rules and logic, machine learning allows 
        systems to learn patterns from data automatically.
        
        This approach has proven particularly effective in domains where creating explicit rules would be 
        extremely difficult or impossible. Consider image recognition: writing code to identify a cat in 
        a photograph by manually specifying all possible visual features would be an insurmountable task.
        
        Instead, machine learning algorithms can examine thousands of labeled images and learn to identify 
        the patterns that distinguish cats from other objects. This capability has revolutionized fields 
        ranging from healthcare diagnostics to autonomous vehicle navigation.
        
        The fundamental principle underlying machine learning is statistical inference. Algorithms analyze 
        training data to identify relationships between input features and desired outputs. Once trained, 
        these models can make predictions or classifications on new, unseen data.
        
        Modern machine learning encompasses several key approaches. Supervised learning uses labeled examples 
        to train predictive models. Unsupervised learning discovers hidden patterns in unlabeled data. 
        Reinforcement learning optimizes decision-making through interaction with an environment.
        """
        
        quality_metrics = content_extractor._analyze_content_quality(high_quality_content)
        
        assert quality_metrics['quality_score'] > 0.7  # Should be high quality
        assert quality_metrics['readability_score'] > 0.4  # Should be reasonably readable
        assert quality_metrics['sentence_count'] >= 10
        assert quality_metrics['paragraph_count'] >= 3
    
    @pytest.mark.unit
    def test_analyze_content_quality_low_quality(self, content_extractor):
        """Test quality analysis for low-quality content."""
        low_quality_content = "Short text. No detail. Bad quality."
        
        quality_metrics = content_extractor._analyze_content_quality(low_quality_content)
        
        assert quality_metrics['quality_score'] < 0.3  # Should be low quality
        assert quality_metrics['sentence_count'] < 5
        assert quality_metrics['paragraph_count'] <= 1
    
    @pytest.mark.unit
    def test_calculate_readability_score(self, content_extractor):
        """Test readability score calculation."""
        # Simple, readable text
        simple_text = "The cat sat on the mat. The dog ran in the park. Children love to play games."
        word_count = len(simple_text.split())
        sentence_count = 3
        
        score = content_extractor._calculate_readability_score(simple_text, word_count, sentence_count)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Simple text should be quite readable
        
        # Complex, less readable text
        complex_text = """
        The methodological framework employed in this investigation necessitates 
        comprehensive analysis utilizing sophisticated computational algorithms 
        integrated with multidimensional statistical modeling approaches.
        """
        word_count = len(complex_text.split())
        sentence_count = 1
        
        score = content_extractor._calculate_readability_score(complex_text, word_count, sentence_count)
        assert 0.0 <= score <= 1.0
        # Complex text should have lower readability
    
    @pytest.mark.unit
    def test_count_syllables(self, content_extractor):
        """Test syllable counting approximation."""
        # Test simple words
        assert content_extractor._count_syllables("cat dog") >= 2
        assert content_extractor._count_syllables("beautiful wonderful") >= 4
        
        # Test with common words
        test_text = "machine learning artificial intelligence"
        syllables = content_extractor._count_syllables(test_text)
        assert syllables >= 8  # Approximate expected minimum
    
    @pytest.mark.unit
    def test_detect_language(self, content_extractor):
        """Test language detection."""
        # English text
        english_text = """
        Machine learning is a subset of artificial intelligence that provides systems 
        with the ability to automatically learn and improve from experience without 
        being explicitly programmed.
        """
        
        with patch('pipelines.normalize.content_extractor.detect', return_value='en'):
            language = content_extractor._detect_language(english_text)
            assert language == 'en'
        
        # Short text should default to English
        short_text = "AI"
        language = content_extractor._detect_language(short_text)
        assert language == 'en'
        
        # Test error handling
        with patch('pipelines.normalize.content_extractor.detect', side_effect=Exception("Detection failed")):
            language = content_extractor._detect_language(english_text)
            assert language == 'en'  # Should fallback to English
    
    @pytest.mark.unit
    def test_extract_keywords(self, content_extractor):
        """Test keyword extraction using TF-IDF."""
        content = """
        Machine learning artificial intelligence data science algorithms neural networks
        deep learning supervised learning unsupervised learning reinforcement learning
        natural language processing computer vision pattern recognition statistical modeling
        Machine learning artificial intelligence data science algorithms neural networks
        """
        
        keywords = content_extractor._extract_keywords(content)
        
        assert len(keywords) > 0
        assert len(keywords) <= 20  # Should be limited
        # Should include important terms
        expected_terms = ['machine learning', 'artificial intelligence', 'neural networks']
        found_terms = [term for term in expected_terms if any(term in ' '.join(keywords).lower() for term in [term])]
        assert len(found_terms) > 0
    
    @pytest.mark.unit 
    def test_extract_keywords_short_content(self, content_extractor):
        """Test keyword extraction with insufficient content."""
        short_content = "AI ML"
        keywords = content_extractor._extract_keywords(short_content)
        assert keywords == []  # Should return empty list for short content
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_content_error_handling(self, content_extractor):
        """Test error handling in content extraction."""
        # Test with invalid HTML
        invalid_html = "<<<invalid html>>>"
        
        result = await content_extractor.extract_content(invalid_html, "https://example.com/invalid")
        
        # Should return empty result structure
        assert result['title'] == ''
        assert result['content'] == ''
        assert result['quality_score'] == 0.0
        assert result['method'] == 'failed'
        assert result['extracted_at'] is not None


class TestHTMLCleaner:
    """Test HTML cleaning and content formatting."""
    
    @pytest.fixture
    def html_cleaner(self):
        """Create HTMLCleaner instance for testing."""
        return HTMLCleaner()
    
    @pytest.fixture
    def messy_html(self):
        """HTML content with various cleaning challenges."""
        return """
        <html>
        <body>
            <div class="content">
                <h1>Article Title</h1>
                <p>This is a paragraph with <script>alert('xss')</script> a script tag.</p>
                <p>Here's some text with <a href="relative-link">relative links</a> and 
                <a href="https://external.com">external links</a>.</p>
                
                <blockquote>
                    This is a quote with <em>emphasis</em> and <strong>strong</strong> text.
                </blockquote>
                
                <ul>
                    <li>First item with <code>code</code></li>
                    <li>Second item</li>
                </ul>
                
                <div class="advertisement">
                    <p>This is an ad that should be removed</p>
                </div>
                
                <table>
                    <tr><th>Header 1</th><th>Header 2</th></tr>
                    <tr><td>Cell 1</td><td>Cell 2</td></tr>
                </table>
                
                <img src="image.jpg" alt="Test image" width="300" height="200">
                
                <p style="color: red; background: url('bg.jpg');">Styled paragraph</p>
            </div>
            
            <!-- This comment should be removed -->
            <footer>Footer content to remove</footer>
        </body>
        </html>
        """
    
    @pytest.mark.unit
    def test_clean_html_basic(self, html_cleaner, messy_html):
        """Test basic HTML cleaning functionality."""
        result = html_cleaner.clean(messy_html, base_url="https://example.com/")
        
        # Check that result contains expected fields
        assert 'cleaned_html' in result
        assert 'markdown' in result
        assert 'plain_text' in result
        assert 'word_count' in result
        assert 'reading_time' in result
        
        # Check content extraction
        assert 'Article Title' in result['plain_text']
        assert 'paragraph with' in result['plain_text']
        assert 'First item' in result['plain_text']
        
        # Check word count is reasonable
        assert result['word_count'] > 10
        assert result['reading_time'] > 0
    
    @pytest.mark.unit
    def test_remove_unwanted_elements(self, html_cleaner):
        """Test removal of unwanted HTML elements."""
        html_with_unwanted = """
        <div>
            <p>Good content</p>
            <script>alert('bad')</script>
            <style>body { color: red; }</style>
            <nav>Navigation</nav>
            <footer>Footer</footer>
            <div class="advertisement">Ad content</div>
            <aside class="sidebar">Sidebar</aside>
        </div>
        """
        
        result = html_cleaner.clean(html_with_unwanted)
        
        # Unwanted content should be removed
        assert 'alert' not in result['plain_text']
        assert 'Navigation' not in result['plain_text']
        assert 'Footer' not in result['plain_text']
        assert 'Ad content' not in result['plain_text']
        assert 'Sidebar' not in result['plain_text']
        
        # Good content should remain
        assert 'Good content' in result['plain_text']
    
    @pytest.mark.unit
    def test_convert_relative_links(self, html_cleaner):
        """Test conversion of relative links to absolute."""
        html_with_links = """
        <div>
            <p>Text with <a href="/relative/path">relative link</a> and 
            <a href="https://external.com/page">absolute link</a>.</p>
        </div>
        """
        
        result = html_cleaner.clean(html_with_links, base_url="https://example.com/article/")
        
        # Check that relative links are converted
        assert 'https://example.com/relative/path' in result['cleaned_html']
        assert 'https://external.com/page' in result['cleaned_html']  # Absolute links unchanged
    
    @pytest.mark.unit
    def test_sanitize_attributes(self, html_cleaner):
        """Test HTML attribute sanitization."""
        html_with_attributes = """
        <div onclick="alert('xss')" data-safe="ok">
            <p style="color: red; background: url('image.jpg');" class="safe-class">
                Text with <a href="javascript:alert('xss')">dangerous link</a>
                and <a href="https://safe.com" onclick="track()">safe link</a>.
            </p>
            <img src="image.jpg" onerror="alert('xss')" alt="Image" width="100">
        </div>
        """
        
        result = html_cleaner.clean(html_with_attributes)
        
        # Dangerous attributes should be removed
        assert 'onclick' not in result['cleaned_html']
        assert 'onerror' not in result['cleaned_html']
        assert 'javascript:' not in result['cleaned_html']
        
        # Safe attributes should remain
        assert 'data-safe' in result['cleaned_html']
        assert 'class="safe-class"' in result['cleaned_html']
        assert 'alt="Image"' in result['cleaned_html']
    
    @pytest.mark.unit
    def test_markdown_conversion(self, html_cleaner):
        """Test conversion of HTML to Markdown."""
        html_content = """
        <div>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
            <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
            <ul>
                <li>First item</li>
                <li>Second item with <code>inline code</code></li>
            </ul>
            <blockquote>This is a quote</blockquote>
            <a href="https://example.com">Link text</a>
        </div>
        """
        
        result = html_cleaner.clean(html_content)
        markdown = result['markdown']
        
        # Check Markdown formatting
        assert '# Main Title' in markdown
        assert '## Subtitle' in markdown
        assert '**bold**' in markdown
        assert '*italic*' in markdown
        assert '- First item' in markdown
        assert '`inline code`' in markdown
        assert '> This is a quote' in markdown
        assert '[Link text](https://example.com)' in markdown
    
    @pytest.mark.unit
    def test_calculate_reading_time(self, html_cleaner):
        """Test reading time calculation."""
        # Test with different word counts
        short_text = "This is a short text with ten words exactly here."
        medium_text = " ".join(["word"] * 250)  # 250 words
        long_text = " ".join(["word"] * 1000)   # 1000 words
        
        # Assuming average reading speed of ~200 words per minute
        short_time = html_cleaner._calculate_reading_time(short_text)
        medium_time = html_cleaner._calculate_reading_time(medium_text)
        long_time = html_cleaner._calculate_reading_time(long_text)
        
        assert short_time == 1  # Minimum 1 minute
        assert medium_time >= 1
        assert long_time >= 5
        assert long_time > medium_time > short_time or medium_time == short_time
    
    @pytest.mark.unit 
    def test_empty_html_handling(self, html_cleaner):
        """Test handling of empty or minimal HTML."""
        empty_html = ""
        minimal_html = "<html></html>"
        
        empty_result = html_cleaner.clean(empty_html)
        minimal_result = html_cleaner.clean(minimal_html)
        
        assert empty_result['word_count'] == 0
        assert empty_result['reading_time'] == 0
        assert empty_result['plain_text'] == ""
        
        assert minimal_result['word_count'] == 0
        assert minimal_result['reading_time'] == 0


# Performance tests for normalization
class TestNormalizationPerformance:
    """Performance tests for content normalization."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_content_extraction_performance(self, performance_benchmarks):
        """Test content extraction performance with large HTML."""
        # Create large HTML document
        large_html = self._create_large_html_document(5000)  # ~5000 words
        
        extractor = ContentExtractor()
        start_time = datetime.now()
        
        result = await extractor.extract_content(large_html, "https://example.com/large")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should process reasonably quickly
        max_time = performance_benchmarks.get("content_processing_time_per_1000_words", 2.0)
        expected_max_time = (result['word_count'] / 1000) * max_time
        
        assert duration <= expected_max_time, f"Content extraction too slow: {duration:.2f}s > {expected_max_time:.2f}s"
        assert result['word_count'] > 1000  # Should have extracted significant content
    
    def _create_large_html_document(self, target_words: int) -> str:
        """Create large HTML document for performance testing."""
        words_per_paragraph = 50
        paragraphs_needed = target_words // words_per_paragraph
        
        paragraphs = []
        for i in range(paragraphs_needed):
            # Create paragraph with varied content
            paragraph = f"""
            <p>This is paragraph {i+1} containing various artificial intelligence and machine learning 
            concepts that are commonly discussed in technical articles. The content includes detailed 
            explanations of algorithms, methodologies, applications, and theoretical foundations that 
            form the basis of modern AI systems. Each paragraph contributes to the overall understanding 
            of complex topics while maintaining readability and coherence throughout the document structure.</p>
            """
            paragraphs.append(paragraph)
        
        return f"""
        <html>
        <head>
            <title>Large Test Document</title>
            <meta name="description" content="Large document for performance testing">
        </head>
        <body>
            <h1>Performance Test Document</h1>
            <div class="content">
                {''.join(paragraphs)}
            </div>
        </body>
        </html>
        """