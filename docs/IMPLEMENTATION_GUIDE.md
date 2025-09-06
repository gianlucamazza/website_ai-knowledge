# Implementation Guide - AI Knowledge Website

This guide provides concrete next steps and implementation details for the AI Knowledge Website architecture.

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose
```

### Development Environment Setup

```bash
# 1. Clone and setup project structure
mkdir -p {apps/site,pipelines/{ingest,normalize,dedup,enrich,publish},orchestrators/langgraph,scripts,tests,data/{sources,curated},docs}

# 2. Install dependencies
npm install astro @astrojs/tailwind zod
pip install langgraph fastapi celery psycopg2-binary redis datasketch

# 3. Start development services
docker-compose up -d postgres redis

# 4. Initialize database
python scripts/init_db.py

# 5. Start development servers
make dev  # Astro dev server
python pipelines/api/main.py  # Pipeline API
```

## Phase 1: Foundation (Weeks 1-4)

### 1.1 Core Schemas and Validation

**File**: `apps/site/src/content/config.ts`

```typescript
import { z, defineCollection } from 'astro:content';

// Content schemas
const glossarySchema = z.object({
  title: z.string().max(100),
  slug: z.string().regex(/^[a-z0-9-]+$/),
  summary: z.string().min(120).max(160),
  aliases: z.array(z.string()).optional(),
  tags: z.array(z.string().min(1)),
  related: z.array(z.string()).optional(),
  updated: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  sources: z.array(z.object({
    source_url: z.string().url(),
    source_title: z.string(),
    license: z.enum(['cc-by', 'cc-by-sa', 'mit', 'proprietary']),
    accessed_date: z.string().optional()
  })).optional()
});

const articleSchema = z.object({
  title: z.string().max(100),
  slug: z.string().regex(/^[a-z0-9-]+$/),
  description: z.string().max(200),
  category: z.enum(['tutorial', 'guide', 'analysis', 'news']),
  tags: z.array(z.string()),
  author: z.string(),
  published: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  updated: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  featured: z.boolean().optional(),
  sources: z.array(z.object({
    source_url: z.string().url(),
    source_title: z.string(),
    license: z.string()
  })).optional()
});

// Export collections
export const collections = {
  glossary: defineCollection({
    type: 'content',
    schema: glossarySchema
  }),
  articles: defineCollection({
    type: 'content', 
    schema: articleSchema
  })
};

// Type exports for use in pipeline
export type GlossaryEntry = z.infer<typeof glossarySchema>;
export type Article = z.infer<typeof articleSchema>;
```

**File**: `pipelines/schemas/content.py`

```python
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
from datetime import date

class SourceReference(BaseModel):
    source_url: HttpUrl
    source_title: str
    license: Literal['cc-by', 'cc-by-sa', 'mit', 'proprietary']
    accessed_date: Optional[date] = None

class GlossaryEntry(BaseModel):
    title: str = Field(..., max_length=100)
    slug: str = Field(..., regex=r'^[a-z0-9-]+$')
    summary: str = Field(..., min_length=120, max_length=160)
    aliases: Optional[List[str]] = None
    tags: List[str] = Field(..., min_items=1)
    related: Optional[List[str]] = None
    updated: date
    sources: Optional[List[SourceReference]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Artificial General Intelligence",
                "slug": "artificial-general-intelligence",
                "summary": "AGI refers to AI systems that match or exceed human cognitive abilities across all domains, representing a theoretical future milestone in AI development.",
                "tags": ["agi", "ai-theory", "future-tech"],
                "updated": "2024-09-06"
            }
        }

class RawContent(BaseModel):
    """Raw content from external sources"""
    source_id: str
    url: HttpUrl
    title: str
    content: str
    content_type: str
    metadata: dict
    fetched_at: date
    content_hash: str
```

### 1.2 Database Schema

**File**: `scripts/init_db.py`

```python
import asyncio
import asyncpg
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func

async def create_database_schema():
    # Connection setup
    DATABASE_URL = "postgresql://user:pass@localhost/ai_knowledge"
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    
    # Sources table
    sources = Table('sources', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String(255), unique=True, nullable=False),
        Column('url', Text, nullable=False),
        Column('type', String(50), nullable=False),
        Column('config', JSON),
        Column('last_crawled', DateTime),
        Column('status', String(50), default='active'),
        Column('created_at', DateTime, server_default=func.now()),
        Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
    )
    
    # Content items table
    content_items = Table('content_items', metadata,
        Column('id', Integer, primary_key=True),
        Column('source_id', Integer, ForeignKey('sources.id')),
        Column('original_url', Text, nullable=False),
        Column('title', Text, nullable=False),
        Column('content_hash', String(64), unique=True),
        Column('simhash', String(16)),  # 64-bit as hex string
        Column('status', String(50), default='processing'),
        Column('metadata', JSON),
        Column('created_at', DateTime, server_default=func.now()),
        Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
    )
    
    # Duplicates table
    duplicates = Table('duplicates', metadata,
        Column('id', Integer, primary_key=True),
        Column('primary_item_id', Integer, ForeignKey('content_items.id')),
        Column('duplicate_item_id', Integer, ForeignKey('content_items.id')),
        Column('similarity_score', String(10)),  # Store as decimal string
        Column('detection_method', String(50)),
        Column('created_at', DateTime, server_default=func.now())
    )
    
    # Workflow state table
    workflow_state = Table('workflow_state', metadata,
        Column('workflow_id', String(36), primary_key=True),  # UUID
        Column('current_stage', String(50)),
        Column('state_data', JSON),
        Column('created_at', DateTime, server_default=func.now()),
        Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
    )
    
    # Create all tables
    metadata.create_all(engine)
    print("Database schema created successfully")

if __name__ == "__main__":
    asyncio.run(create_database_schema())
```

### 1.3 Basic Pipeline Structure

**File**: `pipelines/core/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel
import logging
import time

logger = logging.getLogger(__name__)

class ProcessingResult(BaseModel):
    success: bool
    stage: str
    input_count: int
    output_count: int
    duration_ms: int
    errors: List[str] = []
    metadata: Dict[str, Any] = {}

class PipelineStage(ABC):
    """Base class for all pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    @abstractmethod
    async def process(self, input_data: List[Any]) -> List[Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: List[Any]) -> bool:
        """Validate input data format"""
        pass
    
    async def execute(self, input_data: List[Any]) -> ProcessingResult:
        """Execute the stage with error handling and metrics"""
        start_time = time.time()
        errors = []
        
        try:
            # Validate input
            if not await self.validate_input(input_data):
                raise ValueError("Input validation failed")
            
            # Process data
            self.logger.info(f"Processing {len(input_data)} items in stage {self.name}")
            output_data = await self.process(input_data)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return ProcessingResult(
                success=True,
                stage=self.name,
                input_count=len(input_data),
                output_count=len(output_data),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)
            
            return ProcessingResult(
                success=False,
                stage=self.name,
                input_count=len(input_data) if input_data else 0,
                output_count=0,
                duration_ms=duration_ms,
                errors=[str(e)]
            )
```

**File**: `pipelines/ingest/base_scraper.py`

```python
import aiohttp
import asyncio
from typing import Dict, List, Optional
from urllib.robotparser import RobotFileParser
from pipelines.core.base import PipelineStage
from pipelines.schemas.content import RawContent
import hashlib

class EthicalScraper(PipelineStage):
    """Base scraper with ethical constraints"""
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        super().__init__(name)
        self.rate_limit = rate_limit  # requests per second
        self.session: Optional[aiohttp.ClientSession] = None
        self.robots_cache: Dict[str, RobotFileParser] = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AI-Knowledge-Bot/1.0 (+https://ai-knowledge.example.com/about)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        from urllib.parse import urlparse, urljoin
        
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        if base_url not in self.robots_cache:
            try:
                robots_url = urljoin(base_url, '/robots.txt')
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.read_robots(robots_content)
                        self.robots_cache[base_url] = rp
                    else:
                        # If robots.txt doesn't exist, assume crawling is allowed
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        self.robots_cache[base_url] = rp
            except Exception as e:
                self.logger.warning(f"Could not fetch robots.txt for {base_url}: {e}")
                # Default to allowing crawling if robots.txt check fails
                return True
        
        robots = self.robots_cache.get(base_url)
        if robots:
            return robots.can_fetch('AI-Knowledge-Bot', url)
        return True
    
    async def fetch_content(self, url: str) -> Optional[RawContent]:
        """Fetch content from URL with rate limiting and robots.txt check"""
        
        # Check robots.txt
        if not await self.check_robots_txt(url):
            self.logger.warning(f"Robots.txt disallows crawling {url}")
            return None
        
        # Rate limiting
        await asyncio.sleep(1.0 / self.rate_limit)
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                    
                    return RawContent(
                        source_id=self.name,
                        url=url,
                        title=await self.extract_title(content),
                        content=content,
                        content_type=response.content_type,
                        metadata={
                            'status_code': response.status,
                            'headers': dict(response.headers),
                            'encoding': response.charset
                        },
                        fetched_at=datetime.utcnow().date(),
                        content_hash=content_hash
                    )
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    async def extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else "Untitled"
```

## Phase 2: Content Pipeline (Weeks 5-8)

### 2.1 Normalization Stage

**File**: `pipelines/normalize/html_cleaner.py`

```python
from pipelines.core.base import PipelineStage
from pipelines.schemas.content import RawContent, NormalizedContent
from bs4 import BeautifulSoup, Comment
import re
from typing import List

class HTMLNormalizer(PipelineStage):
    """Clean and normalize HTML content"""
    
    def __init__(self):
        super().__init__("html_normalizer")
        self.allowed_tags = {
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'div', 'span', 'br',
            'strong', 'b', 'em', 'i', 'u',
            'ul', 'ol', 'li',
            'blockquote', 'pre', 'code',
            'a', 'img',
            'table', 'tr', 'td', 'th', 'thead', 'tbody'
        }
        self.allowed_attrs = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title'],
            'blockquote': ['cite']
        }
    
    async def process(self, input_data: List[RawContent]) -> List[NormalizedContent]:
        results = []
        
        for item in input_data:
            try:
                cleaned_content = await self.clean_html(item.content)
                normalized_item = NormalizedContent(
                    source_id=item.source_id,
                    original_url=str(item.url),
                    title=self.clean_title(item.title),
                    content=cleaned_content,
                    content_type="text/html",
                    metadata=item.metadata,
                    processed_at=datetime.utcnow(),
                    content_hash=item.content_hash
                )
                results.append(normalized_item)
                
            except Exception as e:
                self.logger.error(f"Failed to normalize content from {item.url}: {e}")
                continue
        
        return results
    
    async def clean_html(self, html_content: str) -> str:
        """Clean HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove or clean unwanted tags
        for tag in soup.find_all(True):
            if tag.name not in self.allowed_tags:
                tag.unwrap()  # Keep content, remove tag
            else:
                # Clean attributes
                allowed_attrs = self.allowed_attrs.get(tag.name, [])
                attrs_to_remove = [attr for attr in tag.attrs if attr not in allowed_attrs]
                for attr in attrs_to_remove:
                    del tag[attr]
        
        # Convert to text and clean whitespace
        cleaned_html = str(soup)
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
        cleaned_html = re.sub(r'\n\s*\n', '\n', cleaned_html)
        
        return cleaned_html.strip()
    
    def clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common suffixes
        suffixes = [' - Blog', ' | Company Name', ' - Website']
        for suffix in suffixes:
            if title.endswith(suffix):
                title = title[:-len(suffix)]
        
        return title[:100]  # Truncate if too long
    
    async def validate_input(self, input_data: List[RawContent]) -> bool:
        return all(isinstance(item, RawContent) for item in input_data)
```

### 2.2 Duplicate Detection Implementation

**File**: `pipelines/dedup/simhash.py`

```python
import hashlib
import re
from typing import List, Set, Tuple
from collections import defaultdict
from pipelines.core.base import PipelineStage
from pipelines.schemas.content import NormalizedContent, DuplicateMatch

class SimHashDuplicateDetector(PipelineStage):
    """SimHash-based duplicate detection"""
    
    def __init__(self):
        super().__init__("simhash_detector")
        self.index: dict[int, str] = {}  # simhash -> content_id
        self.threshold = 3  # Hamming distance threshold
    
    async def process(self, input_data: List[NormalizedContent]) -> List[DuplicateMatch]:
        duplicates = []
        
        for item in input_data:
            simhash_value = self.generate_simhash(item.content)
            matches = await self.find_similar(simhash_value, item.content_hash)
            
            for match_hash, similarity in matches:
                duplicates.append(DuplicateMatch(
                    primary_id=item.content_hash,
                    duplicate_id=match_hash,
                    similarity_score=similarity,
                    detection_method='simhash'
                ))
            
            # Add to index
            self.index[simhash_value] = item.content_hash
        
        return duplicates
    
    def generate_simhash(self, text: str) -> int:
        """Generate 64-bit SimHash"""
        # Preprocess text
        text = self.preprocess_text(text)
        tokens = self.tokenize(text)
        
        # Initialize bit vector
        bit_vector = [0] * 64
        
        # Process each token
        for token in tokens:
            # Get hash of token
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # Update bit vector
            for i in range(64):
                if token_hash & (1 << i):
                    bit_vector[i] += 1
                else:
                    bit_vector[i] -= 1
        
        # Generate final fingerprint
        fingerprint = 0
        for i, bit in enumerate(bit_vector):
            if bit > 0:
                fingerprint |= 1 << i
        
        return fingerprint
    
    def preprocess_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Generate tokens for hashing"""
        words = text.split()
        
        # Word-level tokens
        tokens = []
        for word in words:
            if len(word) > 2:  # Skip very short words
                tokens.append(word)
        
        # Character n-grams for better fuzzy matching
        char_ngrams = []
        for n in [3, 4]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if not ngram.isspace():
                    char_ngrams.append(f"_{ngram}_")
        
        return tokens + char_ngrams
    
    async def find_similar(self, target_hash: int, exclude_id: str) -> List[Tuple[str, float]]:
        """Find similar content using Hamming distance"""
        matches = []
        
        for stored_hash, content_id in self.index.items():
            if content_id == exclude_id:
                continue
                
            # Calculate Hamming distance
            distance = bin(target_hash ^ stored_hash).count('1')
            
            if distance <= self.threshold:
                similarity = 1 - (distance / 64)  # Convert to similarity score
                matches.append((content_id, similarity))
        
        return matches
    
    async def validate_input(self, input_data: List[NormalizedContent]) -> bool:
        return all(isinstance(item, NormalizedContent) for item in input_data)
```

### 2.3 LangGraph Workflow Implementation

**File**: `orchestrators/langgraph/content_workflow.py`

```python
from langgraph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class ContentPipelineState(TypedDict):
    # Input
    source_configs: List[dict]
    
    # Processing data
    raw_content: List[dict]
    normalized_content: List[dict]
    duplicate_matches: List[dict]
    enriched_content: List[dict]
    
    # State management
    current_stage: str
    workflow_id: str
    errors: List[str]
    stage_results: Dict[str, Any]
    
    # Quality control
    quality_score: Optional[float]
    review_required: bool
    human_decision: Optional[str]

async def ingest_content(state: ContentPipelineState) -> ContentPipelineState:
    """Ingest content from configured sources"""
    try:
        from pipelines.ingest.web_scraper import WebScraper
        
        logger.info(f"Starting ingestion for {len(state['source_configs'])} sources")
        
        scraper = WebScraper()
        raw_items = []
        
        async with scraper:
            for source_config in state['source_configs']:
                try:
                    items = await scraper.fetch_from_source(source_config)
                    raw_items.extend(items)
                    logger.info(f"Fetched {len(items)} items from {source_config.get('name')}")
                except Exception as e:
                    logger.error(f"Failed to fetch from {source_config.get('name')}: {e}")
                    state['errors'].append(f"Ingestion failed for {source_config.get('name')}: {str(e)}")
        
        state['raw_content'] = [item.dict() for item in raw_items]
        state['current_stage'] = 'ingest'
        state['stage_results']['ingest'] = {
            'items_fetched': len(raw_items),
            'sources_processed': len(state['source_configs']),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Ingestion stage failed: {e}")
        state['errors'].append(f"Ingestion stage failed: {str(e)}")
        state['stage_results']['ingest'] = {'success': False, 'error': str(e)}
    
    return state

async def normalize_content(state: ContentPipelineState) -> ContentPipelineState:
    """Normalize and clean content"""
    try:
        from pipelines.normalize.html_cleaner import HTMLNormalizer
        from pipelines.schemas.content import RawContent
        
        logger.info(f"Normalizing {len(state['raw_content'])} items")
        
        normalizer = HTMLNormalizer()
        raw_items = [RawContent(**item) for item in state['raw_content']]
        normalized_items = await normalizer.process(raw_items)
        
        state['normalized_content'] = [item.dict() for item in normalized_items]
        state['current_stage'] = 'normalize'
        state['stage_results']['normalize'] = {
            'items_processed': len(normalized_items),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Normalization stage failed: {e}")
        state['errors'].append(f"Normalization failed: {str(e)}")
        state['stage_results']['normalize'] = {'success': False, 'error': str(e)}
    
    return state

async def check_duplicates(state: ContentPipelineState) -> ContentPipelineState:
    """Detect duplicate content"""
    try:
        from pipelines.dedup.simhash import SimHashDuplicateDetector
        from pipelines.schemas.content import NormalizedContent
        
        logger.info(f"Checking duplicates for {len(state['normalized_content'])} items")
        
        detector = SimHashDuplicateDetector()
        normalized_items = [NormalizedContent(**item) for item in state['normalized_content']]
        duplicate_matches = await detector.process(normalized_items)
        
        state['duplicate_matches'] = [match.dict() for match in duplicate_matches]
        state['current_stage'] = 'dedup'
        state['stage_results']['dedup'] = {
            'duplicates_found': len(duplicate_matches),
            'success': True
        }
        
        # Set review flag if high-similarity matches found
        high_similarity_matches = [
            match for match in duplicate_matches 
            if match.similarity_score > 0.8
        ]
        state['review_required'] = len(high_similarity_matches) > 0
        
    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        state['errors'].append(f"Duplicate detection failed: {str(e)}")
        state['stage_results']['dedup'] = {'success': False, 'error': str(e)}
    
    return state

def should_review(state: ContentPipelineState) -> str:
    """Determine if human review is needed"""
    if state.get('review_required', False):
        return 'human_review'
    return 'enrich'

async def human_review(state: ContentPipelineState) -> ContentPipelineState:
    """Pause for human review"""
    # In a real implementation, this would create a review task
    # and wait for human input via an external API call
    logger.info("Content requires human review - pausing workflow")
    
    state['current_stage'] = 'human_review'
    state['stage_results']['human_review'] = {
        'review_requested': True,
        'duplicate_count': len(state.get('duplicate_matches', [])),
        'success': True
    }
    
    return state

async def enrich_content(state: ContentPipelineState) -> ContentPipelineState:
    """Enrich content with metadata and cross-references"""
    try:
        # Placeholder for content enrichment logic
        enriched_items = state['normalized_content'].copy()
        
        # Add enrichment metadata
        for item in enriched_items:
            item['enriched'] = True
            item['tags'] = []  # Auto-generated tags would go here
            item['related'] = []  # Cross-references would go here
        
        state['enriched_content'] = enriched_items
        state['current_stage'] = 'enrich'
        state['stage_results']['enrich'] = {
            'items_enriched': len(enriched_items),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        state['errors'].append(f"Enrichment failed: {str(e)}")
        state['stage_results']['enrich'] = {'success': False, 'error': str(e)}
    
    return state

async def publish_content(state: ContentPipelineState) -> ContentPipelineState:
    """Generate final markdown files"""
    try:
        # Placeholder for markdown generation
        published_count = len(state.get('enriched_content', []))
        
        state['current_stage'] = 'publish'
        state['stage_results']['publish'] = {
            'items_published': published_count,
            'success': True
        }
        
        logger.info(f"Published {published_count} items")
        
    except Exception as e:
        logger.error(f"Publishing failed: {e}")
        state['errors'].append(f"Publishing failed: {str(e)}")
        state['stage_results']['publish'] = {'success': False, 'error': str(e)}
    
    return state

def create_content_workflow() -> StateGraph:
    """Create the content processing workflow"""
    workflow = StateGraph(ContentPipelineState)
    
    # Add nodes
    workflow.add_node("ingest", ingest_content)
    workflow.add_node("normalize", normalize_content)
    workflow.add_node("dedup", check_duplicates)
    workflow.add_node("human_review", human_review)
    workflow.add_node("enrich", enrich_content)
    workflow.add_node("publish", publish_content)
    
    # Define flow
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "normalize")
    workflow.add_edge("normalize", "dedup")
    
    # Conditional branching for review
    workflow.add_conditional_edges(
        "dedup",
        should_review,
        {
            "human_review": "human_review",
            "enrich": "enrich"
        }
    )
    
    workflow.add_edge("human_review", "enrich")
    workflow.add_edge("enrich", "publish")
    workflow.add_edge("publish", END)
    
    return workflow

# Usage example
async def run_content_pipeline():
    """Run the complete content pipeline"""
    
    # Setup checkpointer for state persistence
    checkpointer = PostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/ai_knowledge"
    )
    
    # Create workflow
    workflow = create_content_workflow()
    app = workflow.compile(checkpointer=checkpointer)
    
    # Initial state
    initial_state = ContentPipelineState(
        source_configs=[
            {
                'name': 'example_source',
                'url': 'https://example.com',
                'type': 'web_scraper'
            }
        ],
        raw_content=[],
        normalized_content=[],
        duplicate_matches=[],
        enriched_content=[],
        current_stage='start',
        workflow_id='test-workflow-1',
        errors=[],
        stage_results={},
        quality_score=None,
        review_required=False,
        human_decision=None
    )
    
    # Run workflow
    config = {"configurable": {"thread_id": "content-pipeline-1"}}
    result = await app.ainvoke(initial_state, config)
    
    print(f"Workflow completed with status: {result['current_stage']}")
    print(f"Errors: {result['errors']}")
    print(f"Stage results: {result['stage_results']}")

if __name__ == "__main__":
    asyncio.run(run_content_pipeline())
```

## Phase 3: Quality & Security (Weeks 9-12)

### 3.1 Comprehensive Testing

**File**: `tests/pipelines/test_duplicate_detection.py`

```python
import pytest
from pipelines.dedup.simhash import SimHashDuplicateDetector
from pipelines.schemas.content import NormalizedContent
from datetime import datetime

@pytest.fixture
def detector():
    return SimHashDuplicateDetector()

@pytest.fixture
def sample_content():
    return [
        NormalizedContent(
            source_id="test_source",
            original_url="http://example.com/1",
            title="Understanding Machine Learning",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash="hash1"
        ),
        NormalizedContent(
            source_id="test_source",
            original_url="http://example.com/2", 
            title="Machine Learning Basics",
            content="ML is a subset of AI that allows computers to learn without explicit programming.",
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash="hash2"
        ),
        NormalizedContent(
            source_id="test_source",
            original_url="http://example.com/3",
            title="Deep Learning Fundamentals", 
            content="Deep learning is a specialized area of machine learning that uses neural networks with multiple layers.",
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash="hash3"
        )
    ]

@pytest.mark.asyncio
async def test_simhash_generation(detector):
    """Test SimHash generation produces consistent results"""
    text = "This is a test document for SimHash generation"
    hash1 = detector.generate_simhash(text)
    hash2 = detector.generate_simhash(text)
    
    assert hash1 == hash2, "SimHash should be deterministic"
    assert isinstance(hash1, int), "SimHash should be an integer"
    assert hash1.bit_length() <= 64, "SimHash should be 64-bit"

@pytest.mark.asyncio
async def test_duplicate_detection_accuracy(detector, sample_content):
    """Test duplicate detection with known similar content"""
    duplicates = await detector.process(sample_content)
    
    # Should detect similarity between first two items (similar ML content)
    similar_pairs = [
        (d.primary_id, d.duplicate_id) for d in duplicates 
        if d.similarity_score > 0.7
    ]
    
    assert len(similar_pairs) > 0, "Should detect similar content"
    
    # Check that dissimilar content (deep learning vs ML basics) has lower similarity
    dissimilar_found = any(
        d.similarity_score < 0.5 for d in duplicates
        if (d.primary_id == "hash3" or d.duplicate_id == "hash3")
    )

@pytest.mark.asyncio
async def test_false_positive_rate(detector):
    """Test that dissimilar content doesn't trigger false positives"""
    dissimilar_content = [
        NormalizedContent(
            source_id="test",
            original_url="http://test.com/1",
            title="Cooking Recipes",
            content="Here are some delicious pasta recipes for dinner",
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash="recipe1"
        ),
        NormalizedContent(
            source_id="test",
            original_url="http://test.com/2",
            title="Space Exploration",
            content="NASA's latest mission to Mars involves advanced rover technology",
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash="space1"
        )
    ]
    
    duplicates = await detector.process(dissimilar_content)
    
    # Should not detect any duplicates between completely different topics
    high_similarity = [d for d in duplicates if d.similarity_score > 0.7]
    assert len(high_similarity) == 0, "Should not have false positives for dissimilar content"

@pytest.mark.asyncio
async def test_performance_large_dataset(detector):
    """Test performance with larger dataset"""
    import time
    
    # Generate 1000 items with some duplicates
    large_dataset = []
    base_content = "This is test content number"
    
    for i in range(1000):
        # Every 10th item is a near-duplicate
        if i % 10 == 0 and i > 0:
            content = f"{base_content} {i-10} with minor changes"
        else:
            content = f"{base_content} {i} with unique information"
        
        large_dataset.append(NormalizedContent(
            source_id="perf_test",
            original_url=f"http://test.com/{i}",
            title=f"Test Item {i}",
            content=content,
            content_type="text/html",
            metadata={},
            processed_at=datetime.utcnow(),
            content_hash=f"hash_{i}"
        ))
    
    start_time = time.time()
    duplicates = await detector.process(large_dataset)
    duration = time.time() - start_time
    
    # Should complete in reasonable time (under 30 seconds)
    assert duration < 30, f"Processing took too long: {duration}s"
    
    # Should detect the intentional near-duplicates
    assert len(duplicates) >= 90, "Should detect near-duplicate patterns"
```

### 3.2 Security Implementation

**File**: `pipelines/security/input_validation.py`

```python
import re
import html
from typing import Dict, Any, List
from pydantic import BaseModel, validator
from urllib.parse import urlparse

class SecurityValidator:
    """Input validation and sanitization for content pipeline"""
    
    ALLOWED_SCHEMES = {'http', 'https'}
    BLOCKED_DOMAINS = {
        'malware-site.com',
        'suspicious-content.net'
    }
    
    # Regex patterns for dangerous content
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>'
    ]
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL security"""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in cls.ALLOWED_SCHEMES:
                return False
            
            # Check for blocked domains
            if parsed.netloc.lower() in cls.BLOCKED_DOMAINS:
                return False
            
            # Check for suspicious patterns
            if any(char in url for char in ['<', '>', '"', "'"]):
                return False
            
            return True
            
        except Exception:
            return False
    
    @classmethod
    def sanitize_html_content(cls, content: str) -> str:
        """Sanitize HTML content"""
        # HTML escape to prevent XSS
        content = html.escape(content)
        
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove null bytes and control characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
        
        return content
    
    @classmethod
    def validate_content_length(cls, content: str, max_length: int = 1000000) -> bool:
        """Validate content length to prevent DoS"""
        return len(content) <= max_length
    
    @classmethod
    def check_content_safety(cls, content: str) -> Dict[str, Any]:
        """Comprehensive content safety check"""
        issues = []
        
        # Check for malicious patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                issues.append(f"Dangerous pattern detected: {pattern[:20]}...")
        
        # Check for suspicious URLs
        url_pattern = r'https?://[^\s<>"\'()]+'
        urls = re.findall(url_pattern, content)
        for url in urls:
            if not cls.validate_url(url):
                issues.append(f"Suspicious URL: {url}")
        
        # Check content length
        if not cls.validate_content_length(content):
            issues.append("Content exceeds maximum length")
        
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'urls_found': len(urls)
        }

class SecureContentModel(BaseModel):
    """Pydantic model with security validation"""
    
    title: str
    content: str
    url: str
    
    @validator('url')
    def validate_url_security(cls, v):
        if not SecurityValidator.validate_url(v):
            raise ValueError('Invalid or unsafe URL')
        return v
    
    @validator('content')
    def validate_content_safety(cls, v):
        safety_check = SecurityValidator.check_content_safety(v)
        if not safety_check['safe']:
            raise ValueError(f"Unsafe content detected: {safety_check['issues']}")
        return SecurityValidator.sanitize_html_content(v)
    
    @validator('title')
    def validate_title(cls, v):
        # Sanitize title
        v = html.escape(v)
        # Length check
        if len(v) > 200:
            raise ValueError('Title too long')
        return v
```

### 3.3 API Security

**File**: `pipelines/api/security.py`

```python
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import jwt
import hashlib
import secrets
import time
from typing import Dict, Optional

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.security_scheme = HTTPBearer()
        # In-memory token blacklist (use Redis in production)
        self.token_blacklist: set[str] = set()
        # Request tracking for anomaly detection
        self.request_tracking: Dict[str, list] = {}
    
    def create_api_token(self, user_id: str, permissions: list[str]) -> str:
        """Create JWT API token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,  # 1 hour expiry
            'jti': secrets.token_urlsafe(16)  # JWT ID for blacklisting
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            # Check blacklist
            if token in self.token_blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Decode and verify
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check expiry
            if payload.get('exp', 0) < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        self.token_blacklist.add(token)
    
    def check_permissions(self, user_permissions: list[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or 'admin' in user_permissions
    
    def detect_anomalous_requests(self, client_ip: str, endpoint: str) -> bool:
        """Simple anomaly detection for suspicious request patterns"""
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        if client_ip in self.request_tracking:
            self.request_tracking[client_ip] = [
                (timestamp, ep) for timestamp, ep in self.request_tracking[client_ip]
                if current_time - timestamp < 3600
            ]
        else:
            self.request_tracking[client_ip] = []
        
        # Add current request
        self.request_tracking[client_ip].append((current_time, endpoint))
        
        # Check for suspicious patterns
        recent_requests = [
            (timestamp, ep) for timestamp, ep in self.request_tracking[client_ip]
            if current_time - timestamp < 300  # Last 5 minutes
        ]
        
        # Too many requests in short time
        if len(recent_requests) > 100:
            return True
        
        # Too many requests to sensitive endpoints
        sensitive_endpoints = ['/api/v1/ingest', '/api/v1/admin']
        sensitive_count = sum(1 for _, ep in recent_requests if ep in sensitive_endpoints)
        if sensitive_count > 10:
            return True
        
        return False

# Global security manager instance
security_manager = SecurityManager("your-secret-key-here")

# Dependency for authenticated endpoints
async def authenticate_user(credentials: HTTPAuthorizationCredentials = security_manager.security_scheme):
    """Authentication dependency"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    return payload

# Dependency for permission checking
def require_permission(permission: str):
    """Create permission requirement dependency"""
    async def check_permission(user = authenticate_user):
        if not security_manager.check_permissions(user.get('permissions', []), permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return user
    return check_permission

# Security middleware
async def security_middleware(request: Request, call_next):
    """Security middleware for all requests"""
    client_ip = get_remote_address(request)
    endpoint = request.url.path
    
    # Anomaly detection
    if security_manager.detect_anomalous_requests(client_ip, endpoint):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Suspicious activity detected"
        )
    
    # Security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'none'"
    
    return response

# Example usage in FastAPI
from fastapi import FastAPI, Depends

app = FastAPI()

@app.middleware("http")
async def add_security_middleware(request: Request, call_next):
    return await security_middleware(request, call_next)

@app.post("/api/v1/ingest")
@limiter.limit("10/minute")  # Rate limiting
async def trigger_ingestion(
    request: Request,
    source_configs: list[dict],
    user = Depends(require_permission("pipeline:ingest"))
):
    """Secure ingestion endpoint"""
    # Implementation here
    pass
```

## Next Steps and Action Items

### Immediate Actions (Week 1)
1. **Set up development environment**
   ```bash
   # Run these commands
   docker-compose up -d postgres redis
   python scripts/init_db.py
   npm install
   pip install -r requirements.txt
   ```

2. **Implement core schemas**
   - Create `apps/site/src/content/config.ts` with Zod schemas
   - Implement `pipelines/schemas/content.py` with Pydantic models

3. **Create basic project structure**
   ```bash
   mkdir -p pipelines/{ingest,normalize,dedup,enrich,publish}
   mkdir -p orchestrators/langgraph
   mkdir -p tests/{pipelines,integration}
   ```

### Week 2-4 Focus Areas
1. **Pipeline Foundation**
   - Implement `PipelineStage` base class
   - Create `EthicalScraper` with robots.txt compliance
   - Build `HTMLNormalizer` for content cleaning

2. **Duplicate Detection**
   - Implement SimHash algorithm
   - Add MinHash LSH for scalability
   - Create comprehensive test suite

3. **LangGraph Integration**
   - Set up PostgreSQL checkpointing
   - Implement basic workflow nodes
   - Add error handling and retry logic

### Success Metrics for Phase 1
- [ ] Zod schemas validate all content correctly
- [ ] Database schema supports all required operations
- [ ] Basic pipeline stages process content end-to-end
- [ ] Duplicate detection achieves >95% accuracy on test dataset
- [ ] LangGraph workflow executes without errors
- [ ] All security validations pass

### Production Readiness Checklist
- [ ] Comprehensive test coverage (>90%)
- [ ] Security audit completed
- [ ] Performance benchmarks met
- [ ] Monitoring and alerting configured
- [ ] Documentation complete
- [ ] Deployment pipeline tested

This implementation guide provides a solid foundation for building the AI Knowledge Website with enterprise-grade quality and security standards. The modular architecture ensures maintainability and scalability while the comprehensive testing and security measures provide confidence for production deployment.