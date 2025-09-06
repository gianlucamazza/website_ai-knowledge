"""
Test data fixtures and sample content for comprehensive testing.

Provides realistic test data that mirrors production content structure.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid


class TestDataFactory:
    """Factory for creating consistent test data across test suites."""
    
    @staticmethod
    def create_sample_articles(count: int = 10) -> List[Dict[str, Any]]:
        """Create sample articles with realistic content."""
        articles = []
        categories = ['artificial-intelligence', 'machine-learning', 'deep-learning', 'nlp', 'computer-vision']
        authors = ['Dr. Sarah Johnson', 'Prof. Michael Chen', 'Dr. Emily Rodriguez', 'Alex Thompson', 'Dr. David Kim']
        
        for i in range(count):
            base_date = datetime(2024, 1, 1) + timedelta(days=i * 5)
            
            articles.append({
                'id': f'test-article-{i+1}',
                'slug': f'test-article-{i+1}',
                'url': f'https://example.com/articles/test-article-{i+1}',
                'title': f'Understanding {categories[i % len(categories)].replace("-", " ").title()}: Article {i+1}',
                'summary': f'This is a comprehensive guide to {categories[i % len(categories)].replace("-", " ")} concepts and applications in modern technology.',
                'content': TestDataFactory._generate_article_content(i+1, categories[i % len(categories)]),
                'raw_html': TestDataFactory._generate_html_content(i+1, categories[i % len(categories)]),
                'markdown_content': TestDataFactory._generate_markdown_content(i+1, categories[i % len(categories)]),
                'author': authors[i % len(authors)],
                'published_date': base_date,
                'updated_date': base_date + timedelta(days=2),
                'source_id': f'test-source-{(i % 3) + 1}',
                'category': categories[i % len(categories)],
                'tags': TestDataFactory._generate_tags(categories[i % len(categories)]),
                'word_count': 800 + (i * 100),
                'reading_time': 5 + (i // 2),
                'difficulty': ['beginner', 'intermediate', 'advanced'][i % 3],
                'language': 'en',
                'quality_score': 0.7 + (i * 0.02),
                'readability_score': 0.6 + (i * 0.03),
                'featured_image': f'https://example.com/images/article-{i+1}.jpg',
                'meta_description': f'Learn about {categories[i % len(categories)].replace("-", " ")} in this comprehensive article covering key concepts and applications.',
                'keywords': TestDataFactory._generate_keywords(categories[i % len(categories)]),
                'related_articles': [],
                'key_points': TestDataFactory._generate_key_points(categories[i % len(categories)]),
                'created_at': base_date,
                'processing_status': 'published'
            })
        
        # Add related articles references
        for i, article in enumerate(articles):
            related_count = min(3, len(articles) - 1)
            related_indices = [(i + j + 1) % len(articles) for j in range(related_count)]
            article['related_articles'] = [
                {
                    'id': articles[idx]['id'],
                    'title': articles[idx]['title'],
                    'url': articles[idx]['url'],
                    'similarity_score': 0.8 - (j * 0.1)
                }
                for j, idx in enumerate(related_indices)
            ]
        
        return articles
    
    @staticmethod
    def create_sample_glossary_entries(count: int = 20) -> List[Dict[str, Any]]:
        """Create sample glossary entries with realistic terms."""
        terms_data = [
            ('Artificial Intelligence', 'The simulation of human intelligence processes by machines, especially computer systems.'),
            ('Machine Learning', 'A subset of AI that provides systems the ability to automatically learn and improve from experience.'),
            ('Deep Learning', 'A subset of machine learning based on artificial neural networks with representation learning.'),
            ('Neural Network', 'A computing system inspired by biological neural networks that constitute animal brains.'),
            ('Natural Language Processing', 'A branch of AI that helps computers understand, interpret and manipulate human language.'),
            ('Computer Vision', 'A field of AI that trains computers to interpret and understand the visual world.'),
            ('Reinforcement Learning', 'An area of ML where agents learn to make decisions by performing actions in an environment.'),
            ('Supervised Learning', 'A type of ML where the algorithm learns from labeled training data.'),
            ('Unsupervised Learning', 'A type of ML that finds hidden patterns in data without labeled examples.'),
            ('Feature Engineering', 'The process of selecting and transforming variables for your machine learning model.'),
            ('Gradient Descent', 'An optimization algorithm used to minimize the cost function in machine learning.'),
            ('Overfitting', 'A modeling error that occurs when a model learns the training data too well.'),
            ('Cross Validation', 'A statistical method used to estimate the performance of machine learning models.'),
            ('Hyperparameter', 'A configuration setting used to control the learning process of a machine learning algorithm.'),
            ('Ensemble Learning', 'A technique that combines multiple learning algorithms to improve predictive performance.'),
            ('Convolutional Neural Network', 'A deep learning algorithm particularly effective for image recognition tasks.'),
            ('Recurrent Neural Network', 'A type of neural network designed for processing sequences of data.'),
            ('Transformer', 'A neural network architecture that has revolutionized natural language processing.'),
            ('Attention Mechanism', 'A technique that allows models to focus on relevant parts of the input data.'),
            ('Transfer Learning', 'A technique where a model developed for one task is reused for a related task.'),
        ]
        
        entries = []
        categories = ['core-concepts', 'algorithms', 'techniques', 'applications', 'tools']
        
        for i in range(min(count, len(terms_data))):
            term, definition = terms_data[i]
            slug = term.lower().replace(' ', '-')
            
            entries.append({
                'id': f'glossary-{i+1}',
                'slug': slug,
                'term': term,
                'definition': definition,
                'detailed_explanation': TestDataFactory._generate_detailed_explanation(term, definition),
                'category': categories[i % len(categories)],
                'tags': TestDataFactory._generate_glossary_tags(term),
                'examples': TestDataFactory._generate_examples(term),
                'related_terms': [],
                'created_date': datetime(2024, 1, 1) + timedelta(days=i),
                'updated_date': datetime(2024, 1, 1) + timedelta(days=i + 10),
                'difficulty': ['beginner', 'intermediate', 'advanced'][i % 3],
                'usage_count': 100 + (i * 10)
            })
        
        # Add related terms references
        for i, entry in enumerate(entries):
            related_count = min(3, len(entries) - 1)
            related_indices = [(i + j + 1) % len(entries) for j in range(related_count)]
            entry['related_terms'] = [
                {
                    'term': entries[idx]['term'],
                    'slug': entries[idx]['slug'],
                    'definition': entries[idx]['definition'][:100] + '...'
                }
                for idx in related_indices
            ]
        
        return entries
    
    @staticmethod
    def create_sample_sources(count: int = 5) -> List[Dict[str, Any]]:
        """Create sample content sources for testing."""
        sources = []
        
        source_configs = [
            {
                'name': 'AI Research Journal',
                'type': 'rss',
                'url': 'https://ai-research.example.com/feed.xml',
                'categories': ['artificial-intelligence', 'research'],
                'tags': ['research', 'academic', 'ai'],
                'quality_score': 0.9
            },
            {
                'name': 'Machine Learning Blog',
                'type': 'rss', 
                'url': 'https://ml-blog.example.com/rss',
                'categories': ['machine-learning'],
                'tags': ['ml', 'tutorials', 'practical'],
                'quality_score': 0.8
            },
            {
                'name': 'Deep Learning News',
                'type': 'rss',
                'url': 'https://dl-news.example.com/feed',
                'categories': ['deep-learning'],
                'tags': ['deep-learning', 'news', 'industry'],
                'quality_score': 0.85
            },
            {
                'name': 'NLP Research Hub',
                'type': 'rss',
                'url': 'https://nlp-hub.example.com/feed.xml',
                'categories': ['nlp'],
                'tags': ['nlp', 'language', 'research'],
                'quality_score': 0.88
            },
            {
                'name': 'Computer Vision Weekly',
                'type': 'rss',
                'url': 'https://cv-weekly.example.com/rss.xml',
                'categories': ['computer-vision'],
                'tags': ['computer-vision', 'weekly', 'updates'],
                'quality_score': 0.82
            }
        ]
        
        for i in range(min(count, len(source_configs))):
            config = source_configs[i]
            
            sources.append({
                'id': f'test-source-{i+1}',
                'name': config['name'],
                'type': config['type'],
                'url': config['url'],
                'enabled': True,
                'categories': config['categories'],
                'tags': config['tags'],
                'max_articles_per_run': 50 + (i * 25),
                'crawl_frequency': 3600 + (i * 1800),  # seconds
                'last_crawled_at': datetime.now() - timedelta(hours=i+1),
                'articles_found': 150 + (i * 50),
                'articles_processed': 120 + (i * 40),
                'quality_score': config['quality_score'],
                'error_count': i,
                'config': {
                    'respect_robots_txt': True,
                    'request_delay': 1.0,
                    'timeout': 30,
                    'user_agent': 'AI-Knowledge-Bot/1.0'
                },
                'created_at': datetime(2024, 1, 1) + timedelta(days=i),
                'updated_at': datetime.now() - timedelta(minutes=i*30)
            })
        
        return sources
    
    @staticmethod
    def create_processing_jobs(count: int = 10) -> List[Dict[str, Any]]:
        """Create sample processing jobs for testing."""
        jobs = []
        statuses = ['running', 'completed', 'failed', 'completed_with_errors']
        
        for i in range(count):
            start_time = datetime.now() - timedelta(hours=i*2)
            
            job = {
                'id': f'job-{uuid.uuid4().hex[:8]}',
                'source_ids': [f'test-source-{(i % 3) + 1}'],
                'status': statuses[i % len(statuses)],
                'articles_found': 20 + (i * 5),
                'articles_processed': 15 + (i * 4),
                'articles_published': 12 + (i * 3),
                'errors_count': max(0, i - 5),
                'started_at': start_time,
                'created_at': start_time - timedelta(minutes=5),
                'updated_at': start_time + timedelta(minutes=30 + i*5)
            }
            
            if job['status'] in ['completed', 'failed', 'completed_with_errors']:
                job['completed_at'] = start_time + timedelta(minutes=45 + i*10)
            
            if job['status'] in ['failed', 'completed_with_errors']:
                job['error_log'] = TestDataFactory._generate_error_log(job['status'])
            
            jobs.append(job)
        
        return jobs
    
    @staticmethod
    def create_duplicate_relations(articles: List[Dict], count: int = 5) -> List[Dict[str, Any]]:
        """Create sample duplicate relations for testing."""
        relations = []
        
        for i in range(min(count, len(articles) - 1)):
            article1 = articles[i]
            article2 = articles[i + 1]
            
            relations.append({
                'id': f'dup-relation-{i+1}',
                'article1_id': article1['id'],
                'article2_id': article2['id'],
                'similarity_score': 0.85 + (i * 0.02),
                'similarity_type': 'content',
                'detection_method': 'simhash',
                'status': 'confirmed' if i % 2 == 0 else 'pending',
                'detected_at': datetime.now() - timedelta(hours=i),
                'reviewed_at': datetime.now() - timedelta(minutes=i*30) if i % 2 == 0 else None,
                'action_taken': 'merged' if i % 3 == 0 else 'kept_separate'
            })
        
        return relations
    
    @staticmethod
    def _generate_article_content(index: int, category: str) -> str:
        """Generate realistic article content."""
        category_content = {
            'artificial-intelligence': f"""
            Artificial Intelligence (AI) has emerged as one of the most transformative technologies of our time. 
            This article explores the fundamental concepts, applications, and future implications of AI systems.
            
            ## Introduction to AI
            
            AI represents a paradigm shift in how we approach problem-solving and automation. Unlike traditional 
            programming approaches, AI systems can learn from data and make decisions based on patterns they discover.
            
            ## Key Components of AI
            
            The foundation of modern AI rests on several key components:
            - Machine Learning algorithms that enable pattern recognition
            - Data processing pipelines that handle large-scale information
            - Neural networks that mimic human brain function
            - Decision-making frameworks that evaluate options and outcomes
            
            ## Applications in Industry
            
            AI is transforming numerous industries including healthcare, finance, transportation, and entertainment.
            Each application brings unique challenges and opportunities for innovation.
            
            ## Future Considerations
            
            As AI continues to evolve, we must consider ethical implications, job displacement concerns, and the 
            need for responsible development practices. The future of AI depends on how well we address these challenges.
            """,
            'machine-learning': f"""
            Machine Learning (ML) is a subset of artificial intelligence that focuses on the development of 
            algorithms that can learn and improve from experience without being explicitly programmed.
            
            ## Fundamental Concepts
            
            ML operates on the principle that systems can automatically learn and improve from experience. 
            This learning process involves analyzing data patterns and making predictions or decisions.
            
            ## Types of Machine Learning
            
            There are three primary categories of machine learning:
            1. **Supervised Learning**: Uses labeled datasets to train algorithms
            2. **Unsupervised Learning**: Finds hidden patterns in unlabeled data
            3. **Reinforcement Learning**: Learns through trial and error interactions
            
            ## Popular Algorithms
            
            Common ML algorithms include linear regression, decision trees, random forests, support vector 
            machines, and various neural network architectures.
            
            ## Real-World Applications
            
            Machine learning powers recommendation systems, fraud detection, image recognition, natural 
            language processing, and autonomous vehicle navigation systems.
            """,
            'deep-learning': f"""
            Deep Learning is a specialized subset of machine learning that uses multi-layered artificial 
            neural networks to model and understand complex patterns in data.
            
            ## Neural Network Architecture
            
            Deep learning networks consist of multiple layers of interconnected nodes (neurons) that process 
            information hierarchically. Each layer learns increasingly complex features from the data.
            
            ## Key Advantages
            
            Deep learning excels at:
            - Automatic feature extraction from raw data
            - Handling complex, high-dimensional data
            - Learning hierarchical representations
            - Achieving state-of-the-art performance in many domains
            
            ## Common Architectures
            
            Popular deep learning architectures include:
            - Convolutional Neural Networks (CNNs) for image processing
            - Recurrent Neural Networks (RNNs) for sequential data
            - Transformer models for natural language processing
            - Generative Adversarial Networks (GANs) for data generation
            
            ## Implementation Challenges
            
            Deep learning requires significant computational resources, large datasets, and careful 
            hyperparameter tuning to achieve optimal performance.
            """,
            'nlp': f"""
            Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
            understand, interpret, and manipulate human language in a valuable way.
            
            ## Core NLP Tasks
            
            NLP encompasses various tasks including:
            - Text classification and sentiment analysis
            - Named entity recognition and extraction
            - Language translation and summarization
            - Question answering and dialogue systems
            
            ## Processing Pipeline
            
            A typical NLP pipeline involves:
            1. Text preprocessing and tokenization
            2. Feature extraction and representation
            3. Model training and validation
            4. Post-processing and result interpretation
            
            ## Modern Approaches
            
            Contemporary NLP leverages transformer architectures, pre-trained language models, and 
            transfer learning to achieve remarkable performance across diverse language tasks.
            
            ## Applications and Impact
            
            NLP powers chatbots, search engines, translation services, content moderation, and 
            voice assistants that millions of people use daily.
            """,
            'computer-vision': f"""
            Computer Vision is a field of artificial intelligence that trains computers to interpret 
            and understand the visual world through digital images and videos.
            
            ## Visual Understanding
            
            Computer vision systems process visual data to:
            - Identify and classify objects in images
            - Detect and track movement in videos
            - Recognize faces and expressions
            - Understand spatial relationships and depth
            
            ## Technical Foundations
            
            Core computer vision techniques include:
            - Image preprocessing and enhancement
            - Feature detection and matching
            - Object segmentation and recognition
            - 3D reconstruction and depth estimation
            
            ## Deep Learning Integration
            
            Modern computer vision heavily relies on deep learning, particularly convolutional neural 
            networks, to achieve human-level performance in many visual recognition tasks.
            
            ## Industry Applications
            
            Computer vision enables autonomous vehicles, medical image analysis, augmented reality, 
            security systems, and quality control in manufacturing.
            """
        }
        
        base_content = category_content.get(category, category_content['artificial-intelligence'])
        return base_content.replace('Article explores', f'Article {index} explores')
    
    @staticmethod
    def _generate_html_content(index: int, category: str) -> str:
        """Generate HTML version of article content."""
        plain_content = TestDataFactory._generate_article_content(index, category)
        
        # Convert to HTML (simplified)
        html_content = plain_content.replace('\n\n## ', '\n\n<h2>').replace('\n\n', '</p>\n\n<p>')
        html_content = html_content.replace('</p>\n\n<h2>', '</p>\n\n<h2>')
        html_content = html_content.replace('<h2>', '<h2>').replace('\n\n<p>', '<p>')
        
        # Add closing tags
        html_content = html_content.replace('</h2>\n<p>', '</h2>\n<p>').replace('</p>', '</p>')
        
        return f"""
        <html>
        <head>
            <title>Test Article {index}: {category.replace('-', ' ').title()}</title>
            <meta name="description" content="Test article about {category.replace('-', ' ')}">
        </head>
        <body>
            <article>
                <h1>Test Article {index}: {category.replace('-', ' ').title()}</h1>
                {html_content}
            </article>
        </body>
        </html>
        """
    
    @staticmethod
    def _generate_markdown_content(index: int, category: str) -> str:
        """Generate Markdown version of article content."""
        return TestDataFactory._generate_article_content(index, category)
    
    @staticmethod
    def _generate_tags(category: str) -> List[str]:
        """Generate relevant tags for a category."""
        tag_mapping = {
            'artificial-intelligence': ['ai', 'artificial-intelligence', 'technology', 'automation'],
            'machine-learning': ['ml', 'machine-learning', 'algorithms', 'data-science'],
            'deep-learning': ['deep-learning', 'neural-networks', 'ai', 'ml'],
            'nlp': ['nlp', 'natural-language-processing', 'text-analysis', 'language'],
            'computer-vision': ['computer-vision', 'image-processing', 'visual-ai', 'recognition']
        }
        
        base_tags = tag_mapping.get(category, ['ai', 'technology'])
        return base_tags + ['test', 'sample']
    
    @staticmethod
    def _generate_keywords(category: str) -> List[str]:
        """Generate SEO keywords for a category."""
        keyword_mapping = {
            'artificial-intelligence': ['artificial intelligence', 'AI systems', 'machine intelligence', 'AI applications'],
            'machine-learning': ['machine learning', 'ML algorithms', 'predictive modeling', 'data science'],
            'deep-learning': ['deep learning', 'neural networks', 'deep neural networks', 'AI models'],
            'nlp': ['natural language processing', 'NLP', 'text analysis', 'language models'],
            'computer-vision': ['computer vision', 'image recognition', 'visual AI', 'image analysis']
        }
        
        return keyword_mapping.get(category, ['artificial intelligence', 'technology'])
    
    @staticmethod
    def _generate_key_points(category: str) -> List[str]:
        """Generate key points for a category."""
        point_mapping = {
            'artificial-intelligence': [
                'AI systems can learn and adapt from experience',
                'Applications span across multiple industries',
                'Ethical considerations are crucial for AI development'
            ],
            'machine-learning': [
                'ML enables automatic pattern recognition from data',
                'Three main types: supervised, unsupervised, and reinforcement learning',
                'Requires quality data and proper algorithm selection'
            ],
            'deep-learning': [
                'Uses multi-layered neural networks for complex pattern recognition',
                'Excels at processing high-dimensional data like images and text',
                'Requires significant computational resources for training'
            ],
            'nlp': [
                'Enables computers to understand and process human language',
                'Powers applications like chatbots and translation services',
                'Modern approaches use transformer architectures'
            ],
            'computer-vision': [
                'Trains computers to interpret visual information',
                'Applications include autonomous vehicles and medical imaging',
                'Relies heavily on convolutional neural networks'
            ]
        }
        
        return point_mapping.get(category, [
            'Advanced technology with broad applications',
            'Requires careful implementation and ethical consideration',
            'Continues to evolve with new research breakthroughs'
        ])
    
    @staticmethod
    def _generate_detailed_explanation(term: str, definition: str) -> str:
        """Generate detailed explanation for glossary terms."""
        return f"""
        {definition}
        
        The concept of {term} has evolved significantly over the years, incorporating advances in 
        computational power, algorithmic sophistication, and data availability. Modern implementations 
        of {term} leverage cutting-edge techniques and methodologies to achieve performance levels 
        that were previously thought impossible.
        
        Key characteristics include scalability, efficiency, and adaptability to various problem domains. 
        Understanding {term} is essential for practitioners working in related fields, as it forms the 
        foundation for more advanced concepts and applications.
        
        Current research continues to push the boundaries of what's possible with {term}, exploring 
        new applications and improving existing implementations through novel approaches and optimizations.
        """
    
    @staticmethod
    def _generate_glossary_tags(term: str) -> List[str]:
        """Generate tags for glossary terms."""
        term_lower = term.lower()
        tags = ['glossary']
        
        if 'learning' in term_lower:
            tags.extend(['learning', 'algorithms'])
        if 'neural' in term_lower or 'network' in term_lower:
            tags.extend(['neural-networks', 'deep-learning'])
        if 'language' in term_lower:
            tags.extend(['nlp', 'language'])
        if 'vision' in term_lower or 'image' in term_lower:
            tags.extend(['computer-vision', 'visual'])
        if 'intelligence' in term_lower:
            tags.extend(['ai', 'intelligence'])
        
        return tags
    
    @staticmethod
    def _generate_examples(term: str) -> List[str]:
        """Generate examples for glossary terms."""
        term_lower = term.lower()
        
        if 'machine learning' in term_lower:
            return ['Email spam detection', 'Product recommendation systems', 'Credit fraud detection']
        elif 'neural network' in term_lower:
            return ['Image classification models', 'Language translation systems', 'Speech recognition']
        elif 'natural language' in term_lower:
            return ['Chatbots and virtual assistants', 'Document summarization', 'Sentiment analysis']
        elif 'computer vision' in term_lower:
            return ['Facial recognition systems', 'Medical image analysis', 'Autonomous vehicle navigation']
        elif 'deep learning' in term_lower:
            return ['Image generation with GANs', 'Language models like GPT', 'Game-playing AI systems']
        else:
            return [f'{term} in healthcare applications', f'{term} in financial services', f'{term} in autonomous systems']
    
    @staticmethod
    def _generate_error_log(status: str) -> str:
        """Generate realistic error logs for processing jobs."""
        if status == 'failed':
            return """
            ERROR: Failed to process batch at 2024-01-15 10:30:45
            - HTTP 404: Source RSS feed not found at https://example.com/feed.xml
            - Connection timeout after 30 seconds
            - Retry attempts exceeded (3/3)
            - Job terminated with exit code 1
            """
        elif status == 'completed_with_errors':
            return """
            WARNING: Job completed with errors at 2024-01-15 11:45:23
            - Failed to extract content from 2 out of 25 articles
            - Duplicate detection service temporarily unavailable
            - 3 articles skipped due to quality score below threshold (0.5)
            - All other articles processed successfully
            """
        else:
            return ""


# Convenience functions for common test data needs
def get_sample_articles(count: int = 5) -> List[Dict[str, Any]]:
    """Get sample articles for testing."""
    return TestDataFactory.create_sample_articles(count)


def get_sample_glossary_entries(count: int = 10) -> List[Dict[str, Any]]:
    """Get sample glossary entries for testing."""
    return TestDataFactory.create_sample_glossary_entries(count)


def get_sample_sources(count: int = 3) -> List[Dict[str, Any]]:
    """Get sample sources for testing."""
    return TestDataFactory.create_sample_sources(count)


def get_sample_processing_jobs(count: int = 5) -> List[Dict[str, Any]]:
    """Get sample processing jobs for testing."""
    return TestDataFactory.create_processing_jobs(count)


# Content for testing different scenarios
MALICIOUS_HTML_SAMPLES = [
    '<script>alert("XSS")</script>',
    '<img src="x" onerror="alert(\'XSS\')">',
    '<iframe src="javascript:alert(\'XSS\')"></iframe>',
    '<div onclick="alert(\'XSS\')">Click me</div>',
    '<style>body{background:url("javascript:alert(\'XSS\')")}</style>',
    '<object data="data:text/html,<script>alert(\'XSS\')</script>"></object>',
    '<svg onload="alert(\'XSS\')">',
    '<meta http-equiv="refresh" content="0;url=javascript:alert(\'XSS\')">'
]

PERFORMANCE_TEST_CONTENT = {
    'large_html': """
    <html>
    <head><title>Large Performance Test Article</title></head>
    <body>
        <h1>Performance Testing Article</h1>
        {}
    </body>
    </html>
    """.format('\n'.join([
        f'<p>This is paragraph {i+1} with detailed content about artificial intelligence and machine learning concepts. ' +
        'It contains technical information and explanations that would be typical in a comprehensive AI article. ' +
        'The content includes various keywords and phrases that would be processed during content extraction and analysis.</p>'
        for i in range(100)
    ])),
    
    'large_rss_feed': lambda count: f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Performance Test Feed</title>
            <description>Large feed for performance testing</description>
            <link>https://example.com</link>
            {chr(10).join([
                f'''<item>
                    <title>Performance Test Article {i+1}</title>
                    <link>https://example.com/perf-article-{i+1}</link>
                    <description>Performance testing article {i+1} about AI and ML concepts</description>
                    <pubDate>Mon, {(i % 28) + 1:02d} Jan 2024 10:00:00 GMT</pubDate>
                    <category>performance-test</category>
                </item>'''
                for i in range(count)
            ])}
        </channel>
    </rss>"""
}

EDGE_CASE_DATA = {
    'empty_content': {
        'title': 'Empty Article',
        'content': '',
        'summary': '',
        'author': '',
        'tags': []
    },
    'special_characters': {
        'title': 'Article with Special Characters: áéíóú ñ ü ç 中文 日本語 العربية',
        'content': 'Content with émojis 🤖🧠💡 and spëcial chars ñ ü ç',
        'author': 'José María Gutiérrez-Rodríguez',
        'tags': ['spëcial', 'chärs', 'tëst']
    },
    'very_long_content': {
        'title': 'Very Long Article Title ' + 'x' * 200,
        'content': 'Very long content. ' * 10000,  # ~170KB of text
        'summary': 'Very long summary. ' * 100,
        'tags': [f'tag-{i}' for i in range(50)]
    },
    'unicode_content': {
        'title': '🤖 AI and Machine Learning in 中文',
        'content': 'Content with various Unicode characters: 中文, العربية, Русский, 日本語, हिन्दी',
        'author': '张三',
        'tags': ['中文', 'العربية', 'Русский']
    }
}