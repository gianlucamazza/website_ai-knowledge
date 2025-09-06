# Contributing Guide

Welcome to the AI Knowledge Website project! This guide provides everything you need to know about contributing to the project, from submitting bug reports to developing new features.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Submission Guidelines](#submission-guidelines)
5. [Code Standards](#code-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Standards](#documentation-standards)
8. [Review Process](#review-process)
9. [Community Guidelines](#community-guidelines)
10. [Recognition](#recognition)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@ai-knowledge.org.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Read and understood this contributing guide
- Set up your local development environment (see [Development Guide](DEVELOPMENT_GUIDE.md))
- Familiarized yourself with the project structure and architecture
- Joined our community channels (Discord/Slack)

### Types of Contributions

We welcome various types of contributions:

**Code Contributions:**
- Bug fixes
- New features
- Performance improvements
- Security enhancements
- Test improvements

**Documentation:**
- API documentation improvements
- Tutorial creation
- Code comments and docstrings
- Translation efforts

**Community:**
- Bug reporting
- Feature requests
- Community support
- Code reviews

### Finding Work

1. **Good First Issues**: Look for issues labeled `good first issue` for newcomers
2. **Help Wanted**: Issues labeled `help wanted` are ready for community contribution
3. **Bug Reports**: Check open bug reports that match your interests
4. **Feature Requests**: Contribute to requested features
5. **Documentation**: Always room for documentation improvements

## Development Process

### Workflow Overview

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Develop**: Make your changes following our standards
4. **Test**: Ensure all tests pass and add new tests
5. **Document**: Update documentation as needed
6. **Submit**: Create a pull request for review

### Detailed Steps

1. **Fork the Repository**:

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-knowledge-website.git
cd ai-knowledge-website

# Add upstream remote
git remote add upstream https://github.com/original-org/ai-knowledge-website.git
```

2. **Set Up Development Environment**:

```bash
# Follow the development guide
make install
make dev

# Verify setup
make health-check
```

3. **Create Feature Branch**:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

4. **Make Changes**:

```bash
# Make your changes
# Follow code standards
# Add tests
# Update documentation
```

5. **Test Your Changes**:

```bash
# Run full test suite
make test

# Run specific tests
make test-unit
make test-integration

# Check code quality
make lint
make type-check
```

6. **Commit Changes**:

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add content quality scoring algorithm

- Implement SimHash-based duplicate detection
- Add quality metrics to content model
- Include unit tests for scoring logic
- Update API documentation

Fixes #123"
```

7. **Push and Create PR**:

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request through GitHub interface
```

### Branch Naming Conventions

- **Features**: `feature/brief-description`
- **Bug Fixes**: `fix/issue-description`
- **Documentation**: `docs/what-you-are-documenting`
- **Performance**: `perf/performance-improvement`
- **Refactoring**: `refactor/component-being-refactored`
- **Security**: `security/security-improvement`

## Submission Guidelines

### Issue Reporting

When reporting bugs or requesting features, please:

**For Bug Reports:**

```markdown
## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Environment
- OS: [e.g. macOS 12.0]
- Browser: [e.g. Chrome 96]
- Version: [e.g. 1.2.3]

## Additional Context
Add any other context about the problem here.

## Logs
```
Include relevant log output here
```
```

**For Feature Requests:**

```markdown
## Feature Description
A clear and concise description of the feature you'd like to see.

## Problem Statement
What problem does this feature solve?

## Proposed Solution
Describe your proposed solution in detail.

## Alternatives Considered
Describe any alternative solutions you've considered.

## Additional Context
Add any other context, mockups, or examples.
```

### Pull Request Guidelines

**Before Submitting:**
- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Self-review completed
- [ ] Linked to relevant issues

**PR Title Format:**

Use conventional commit format:
- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update API documentation`
- `perf: improve query performance`
- `refactor: restructure component`
- `test: add missing test coverage`

**PR Description Template:**

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance testing completed (if applicable)

## Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] README updated (if needed)

## Breaking Changes
List any breaking changes and migration steps needed.

## Additional Notes
Any additional information reviewers should know.

## Related Issues
Fixes #123
Closes #456
Related to #789
```

## Code Standards

### General Principles

1. **Clarity Over Cleverness**: Write code that is easy to read and understand
2. **Consistency**: Follow established patterns in the codebase
3. **Documentation**: Document complex logic and public interfaces
4. **Testing**: Write tests for all new functionality
5. **Security**: Consider security implications of all changes

### Python Code Standards

**Style Guide**: Follow PEP 8 with these additions:

```python
# Good: Clear variable names
def calculate_content_similarity(content1: str, content2: str) -> float:
    """Calculate similarity score between two content items.
    
    Args:
        content1: First content string
        content2: Second content string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Implementation here
    pass

# Bad: Unclear variable names
def calc_sim(c1, c2):
    # No docstring, unclear purpose
    pass
```

**Error Handling**:

```python
# Good: Specific exceptions with context
try:
    content = fetch_content_from_api(url)
except APIRateLimitError as e:
    logger.warning(f"Rate limited for {url}: {e}")
    raise ContentIngestionError(f"Failed to fetch {url} due to rate limiting") from e
except APIError as e:
    logger.error(f"API error for {url}: {e}")
    raise ContentIngestionError(f"Failed to fetch {url}: {e}") from e

# Bad: Generic exception handling
try:
    content = fetch_content_from_api(url)
except Exception as e:
    print(f"Error: {e}")
    return None
```

**Type Hints**:

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# Good: Complete type annotations
def process_content_batch(
    items: List[ContentItem],
    config: ProcessingConfig,
    batch_size: int = 100
) -> List[ProcessedContent]:
    """Process a batch of content items."""
    processed_items = []
    for item in items:
        processed_items.append(process_single_item(item, config))
    return processed_items

# Use Pydantic for data validation
class ContentItem(BaseModel):
    title: str
    content: str
    source_url: str
    metadata: Dict[str, Any] = {}
    quality_score: Optional[float] = None
```

### TypeScript/JavaScript Standards

**Astro Components**:

```typescript
// Good: Well-structured component
---
import type { ContentCollectionEntry } from 'astro:content';
import { formatDate } from '../utils/date';

interface Props {
  article: ContentCollectionEntry<'articles'>;
  featured?: boolean;
}

const { article, featured = false } = Astro.props;
const { title, description, publishDate, author } = article.data;
---

<article class:list={['article-card', { featured }]}>
  <header>
    <h2>{title}</h2>
    <p class="meta">
      By {author} on {formatDate(publishDate)}
    </p>
  </header>
  <div class="content">
    <p>{description}</p>
  </div>
</article>

<style>
  .article-card {
    /* Scoped styles */
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
  }
  
  .featured {
    border-color: var(--color-primary);
    box-shadow: var(--shadow-lg);
  }
</style>
```

**Configuration and Schemas**:

```typescript
// Good: Well-defined schemas
import { z, defineCollection } from 'astro:content';

const articleSchema = z.object({
  title: z.string().min(1).max(200),
  description: z.string().min(10).max(500),
  publishDate: z.date(),
  author: z.string(),
  category: z.enum(['machine-learning', 'nlp', 'computer-vision']),
  tags: z.array(z.string()).min(1).max(10),
  featured: z.boolean().default(false),
  draft: z.boolean().default(false)
});

export const collections = {
  articles: defineCollection({
    type: 'content',
    schema: articleSchema
  })
};
```

### Database Standards

**Migration Files**:

```python
# Good: Descriptive migration with proper rollback
def upgrade():
    """Add quality_score column to content_items table."""
    op.add_column('content_items', 
        sa.Column('quality_score', sa.Float, nullable=True, default=0.0))
    
    # Add index for performance
    op.create_index('idx_content_items_quality_score', 'content_items', 
        ['quality_score'], postgresql_using='btree')
    
    # Add check constraint for valid range
    op.create_check_constraint(
        'ck_content_items_quality_score_range',
        'content_items',
        'quality_score >= 0.0 AND quality_score <= 1.0'
    )

def downgrade():
    """Remove quality_score column and related constraints."""
    op.drop_constraint('ck_content_items_quality_score_range', 'content_items')
    op.drop_index('idx_content_items_quality_score')
    op.drop_column('content_items', 'quality_score')
```

## Testing Requirements

### Test Coverage

- **Minimum Coverage**: 85% overall, 90% for new code
- **Critical Paths**: 100% coverage for security and data integrity code
- **Integration Tests**: Must cover all API endpoints and pipeline flows

### Test Types Required

1. **Unit Tests**: For all new functions and classes
2. **Integration Tests**: For API endpoints and database operations
3. **Performance Tests**: For performance-critical code
4. **Security Tests**: For authentication and data handling

### Test Examples

```python
# Good: Comprehensive test with multiple scenarios
class TestContentDeduplication:
    
    @pytest.fixture
    def duplicate_detector(self):
        return SimHashDuplicateDetector(threshold=3)
    
    def test_identical_content_detected_as_duplicate(self, duplicate_detector):
        """Test that identical content is detected as duplicate."""
        content = "This is a test article about machine learning."
        
        hash1 = duplicate_detector.compute_hash(content)
        hash2 = duplicate_detector.compute_hash(content)
        
        assert duplicate_detector.are_duplicates(hash1, hash2)
    
    def test_similar_content_detected_within_threshold(self, duplicate_detector):
        """Test that similar content is detected within threshold."""
        content1 = "This is a test article about machine learning."
        content2 = "This is a test article about machine learning algorithms."
        
        hash1 = duplicate_detector.compute_hash(content1)
        hash2 = duplicate_detector.compute_hash(content2)
        
        # Should be detected as duplicates (within threshold)
        assert duplicate_detector.are_duplicates(hash1, hash2)
    
    def test_different_content_not_detected_as_duplicate(self, duplicate_detector):
        """Test that different content is not detected as duplicate."""
        content1 = "This is about machine learning."
        content2 = "This discusses quantum computing principles."
        
        hash1 = duplicate_detector.compute_hash(content1)
        hash2 = duplicate_detector.compute_hash(content2)
        
        assert not duplicate_detector.are_duplicates(hash1, hash2)
    
    @pytest.mark.parametrize("threshold,expected", [
        (1, True),   # Very strict
        (3, True),   # Default
        (5, False),  # More lenient
    ])
    def test_threshold_affects_detection(self, threshold, expected):
        """Test that threshold parameter affects duplicate detection."""
        detector = SimHashDuplicateDetector(threshold=threshold)
        
        content1 = "Machine learning is a subset of AI."
        content2 = "Machine learning is a branch of AI."
        
        hash1 = detector.compute_hash(content1)
        hash2 = detector.compute_hash(content2)
        
        assert detector.are_duplicates(hash1, hash2) == expected
```

## Documentation Standards

### Code Documentation

**Python Docstrings** (Google Style):

```python
def calculate_similarity_score(content1: str, content2: str, 
                             algorithm: str = "simhash") -> float:
    """Calculate similarity score between two content items.
    
    This function supports multiple similarity algorithms and returns
    a normalized score between 0.0 and 1.0, where 1.0 indicates
    identical content.
    
    Args:
        content1: The first content string to compare.
        content2: The second content string to compare.
        algorithm: The similarity algorithm to use. Supported values
            are 'simhash', 'minhash', and 'cosine'. Defaults to 'simhash'.
    
    Returns:
        A float between 0.0 and 1.0 representing content similarity.
        Higher values indicate more similar content.
    
    Raises:
        ValueError: If an unsupported algorithm is specified.
        ContentError: If content cannot be processed.
        
    Example:
        >>> score = calculate_similarity_score(
        ...     "This is about AI",
        ...     "This discusses artificial intelligence",
        ...     algorithm="cosine"
        ... )
        >>> print(f"Similarity: {score:.2f}")
        Similarity: 0.85
    """
```

**TypeScript JSDoc**:

```typescript
/**
 * Formats a date for display in article metadata.
 * 
 * @param date - The date to format
 * @param locale - The locale to use for formatting (defaults to 'en-US')
 * @param options - Intl.DateTimeFormat options for customization
 * @returns A formatted date string
 * 
 * @example
 * ```typescript
 * const formatted = formatDate(new Date('2024-01-01'), 'en-US');
 * console.log(formatted); // "January 1, 2024"
 * ```
 */
export function formatDate(
  date: Date,
  locale: string = 'en-US',
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }
): string {
  return new Intl.DateTimeFormat(locale, options).format(date);
}
```

### API Documentation

**OpenAPI/Swagger Annotations**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

class ContentItem(BaseModel):
    """Content item model for API responses."""
    id: str
    title: str
    content: str
    quality_score: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "art_20240101_001",
                "title": "Introduction to Transformers",
                "content": "Transformers are a revolutionary...",
                "quality_score": 0.92
            }
        }

@app.get(
    "/api/v1/content/{content_id}",
    response_model=ContentItem,
    summary="Get content item by ID",
    description="""Retrieve a specific content item by its unique identifier.
    
    Returns the complete content item including metadata and quality score
    if available. Content must be in 'published' status to be accessible
    through this endpoint.
    """,
    responses={
        200: {
            "description": "Content item found and returned successfully",
            "model": ContentItem
        },
        404: {
            "description": "Content item not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Content item not found"}
                }
            }
        }
    }
)
async def get_content_item(content_id: str) -> ContentItem:
    """Get a content item by ID."""
    # Implementation here
    pass
```

## Review Process

### Review Requirements

All pull requests require:
- **Code Review**: At least one maintainer approval
- **Automated Checks**: All CI checks must pass
- **Documentation**: Updated documentation for new features
- **Testing**: Adequate test coverage

### Review Checklist

Reviewers should check:

**Code Quality:**
- [ ] Code follows project style guidelines
- [ ] Logic is clear and well-commented
- [ ] Error handling is appropriate
- [ ] No obvious security issues
- [ ] Performance implications considered

**Testing:**
- [ ] Adequate test coverage
- [ ] Tests are meaningful and robust
- [ ] Edge cases are covered
- [ ] Tests follow naming conventions

**Documentation:**
- [ ] Code is properly documented
- [ ] API changes are documented
- [ ] Breaking changes are highlighted
- [ ] Examples are provided where appropriate

**Architecture:**
- [ ] Changes fit within existing architecture
- [ ] Proper separation of concerns
- [ ] Dependencies are justified
- [ ] Database changes are backward compatible

### Review Timeline

- **Initial Response**: Within 2 business days
- **Full Review**: Within 5 business days for standard PRs
- **Complex PRs**: May require longer review period
- **Security PRs**: Priority review within 1 business day

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, and discussions
- **Discord**: Real-time community chat and support
- **Discussions**: Long-form technical discussions
- **Email**: Maintainer contact and security reports

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Search Issues**: Look for similar issues or questions
3. **Community**: Ask in Discord or GitHub Discussions
4. **Support**: Create a GitHub issue for specific problems

### Mentorship

We offer mentorship for new contributors:
- **Good First Issues**: Specially marked issues for beginners
- **Pair Programming**: Available for complex features
- **Code Review**: Detailed feedback to help you learn
- **Office Hours**: Regular community office hours

## Recognition

### Contributor Recognition

We recognize contributions through:
- **Contributors File**: All contributors listed in CONTRIBUTORS.md
- **Release Notes**: Significant contributions mentioned in releases
- **Community Spotlight**: Featured contributors in community channels
- **Maintainer Path**: Path to becoming a project maintainer

### Becoming a Maintainer

Maintainers are active community members who:
- Have made significant code contributions
- Participate in code reviews and community discussions
- Demonstrate understanding of project goals and architecture
- Show commitment to the project's long-term success

**Maintainer Responsibilities:**
- Review and merge pull requests
- Triage issues and guide discussions
- Mentor new contributors
- Participate in architectural decisions
- Maintain project quality standards

---

**Questions?** Join our [Discord](https://discord.gg/ai-knowledge) or create a [GitHub Discussion](https://github.com/org/ai-knowledge-website/discussions).

**Thank you for contributing to AI Knowledge Website!** 🚀

**Last Updated**: January 2024  
**Version**: 1.0.0