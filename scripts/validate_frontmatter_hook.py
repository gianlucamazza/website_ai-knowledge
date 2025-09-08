#!/usr/bin/env python3
"""
Pre-commit Frontmatter Validation Hook
Validates frontmatter structure and content for AI Knowledge Website
"""

import sys
import yaml
import frontmatter
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FrontmatterValidator:
    """Enterprise frontmatter validator with AI-specific rules"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
        # Load controlled taxonomies
        self.load_taxonomies()
        
    def load_taxonomies(self) -> None:
        """Load controlled vocabularies for tags and categories"""
        try:
            # Try to load from the actual taxonomy files
            tags_file = Path('apps/site/src/content/taxonomies/tags.json')
            categories_file = Path('apps/site/src/content/taxonomies/categories.json')
            
            if tags_file.exists():
                import json
                with open(tags_file) as f:
                    self.valid_tags = set(json.load(f))
            else:
                # Fallback to common AI tags
                self.valid_tags = {
                    'machine-learning', 'deep-learning', 'neural-networks',
                    'nlp', 'computer-vision', 'reinforcement-learning',
                    'ai-engineering', 'transformers', 'llm', 'agents',
                    'supervised-learning', 'unsupervised-learning',
                    'optimization', 'embeddings', 'attention',
                    'fine-tuning', 'prompt-engineering', 'rag',
                    'multimodal', 'generative-ai', 'classification',
                    'regression', 'clustering', 'dimensionality-reduction'
                }
            
            if categories_file.exists():
                import json
                with open(categories_file) as f:
                    self.valid_categories = set(json.load(f))
            else:
                # Fallback categories
                self.valid_categories = {
                    'fundamentals', 'algorithms', 'architectures',
                    'applications', 'tools', 'ethics', 'research'
                }
                
        except Exception as e:
            logger.warning(f"Could not load taxonomies: {e}")
            # Use fallback taxonomies
            self.valid_tags = {'machine-learning', 'deep-learning', 'nlp'}
            self.valid_categories = {'fundamentals', 'algorithms'}
    
    def validate_required_fields(self, metadata: Dict[str, Any], file_path: str) -> None:
        """Validate required frontmatter fields"""
        required_fields = ['title', 'slug', 'summary', 'tags', 'updated']
        
        for field in required_fields:
            if field not in metadata:
                self.errors.append(f"{file_path}: Missing required field '{field}'")
            elif not metadata[field]:
                self.errors.append(f"{file_path}: Field '{field}' is empty")
    
    def validate_title(self, title: str, file_path: str) -> None:
        """Validate title field"""
        if not title:
            return
            
        if len(title) < 3:
            self.errors.append(f"{file_path}: Title too short (minimum 3 characters)")
        elif len(title) > 100:
            self.errors.append(f"{file_path}: Title too long (maximum 100 characters)")
            
        # Check for proper capitalization
        if title != title.strip():
            self.errors.append(f"{file_path}: Title has leading/trailing whitespace")
    
    def validate_slug(self, slug: str, file_path: str) -> None:
        """Validate slug field"""
        if not slug:
            return
            
        # Slug should be kebab-case
        if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', slug):
            self.errors.append(f"{file_path}: Slug should be kebab-case (lowercase, hyphens)")
            
        if len(slug) < 3:
            self.errors.append(f"{file_path}: Slug too short (minimum 3 characters)")
        elif len(slug) > 50:
            self.errors.append(f"{file_path}: Slug too long (maximum 50 characters)")
    
    def validate_summary(self, summary: str, file_path: str) -> None:
        """Validate summary field"""
        if not summary:
            return
            
        word_count = len(summary.split())
        
        # AI content summaries should be comprehensive but concise
        if word_count < 20:
            self.errors.append(f"{file_path}: Summary too short (minimum 20 words, got {word_count})")
        elif word_count > 200:
            self.warnings.append(f"{file_path}: Summary quite long ({word_count} words, consider 20-200)")
            
        # Check for proper sentence structure
        if not summary.endswith('.'):
            self.warnings.append(f"{file_path}: Summary should end with a period")
            
        # Check for AI terminology consistency
        ai_terms = {
            'AI': 'artificial intelligence',
            'ML': 'machine learning', 
            'NLP': 'natural language processing',
            'CNN': 'convolutional neural network',
            'RNN': 'recurrent neural network',
            'LSTM': 'long short-term memory',
            'GAN': 'generative adversarial network',
        }
        
        for abbrev, full_term in ai_terms.items():
            if abbrev in summary and full_term not in summary.lower():
                self.warnings.append(f"{file_path}: Consider defining '{abbrev}' on first use")
    
    def validate_tags(self, tags: List[str], file_path: str) -> None:
        """Validate tags field"""
        if not tags:
            self.errors.append(f"{file_path}: At least one tag is required")
            return
            
        if len(tags) > 10:
            self.warnings.append(f"{file_path}: Many tags ({len(tags)}), consider reducing for better categorization")
            
        for tag in tags:
            if not isinstance(tag, str):
                self.errors.append(f"{file_path}: Tag should be string, got {type(tag)}")
                continue
                
            # Check tag format
            if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', tag):
                self.errors.append(f"{file_path}: Tag '{tag}' should be kebab-case")
                
            # Check against controlled vocabulary (with some flexibility)
            if hasattr(self, 'valid_tags') and tag not in self.valid_tags:
                # Only warn for unknown tags to allow vocabulary growth
                self.warnings.append(f"{file_path}: Tag '{tag}' not in controlled vocabulary")
    
    def validate_updated(self, updated: str, file_path: str) -> None:
        """Validate updated date field"""
        if not updated:
            return
            
        # Check date format (YYYY-MM-DD)
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(updated)):
            self.errors.append(f"{file_path}: 'updated' should be in YYYY-MM-DD format")
            return
            
        # Validate actual date
        try:
            from datetime import datetime
            date_obj = datetime.strptime(str(updated), '%Y-%m-%d')
            
            # Check if date is not in the future
            if date_obj.date() > datetime.now().date():
                self.warnings.append(f"{file_path}: 'updated' date is in the future")
        except ValueError:
            self.errors.append(f"{file_path}: Invalid date format in 'updated' field")
    
    def validate_aliases(self, aliases: List[str], file_path: str) -> None:
        """Validate optional aliases field"""
        if not aliases:
            return
            
        if not isinstance(aliases, list):
            self.errors.append(f"{file_path}: 'aliases' should be a list")
            return
            
        if len(aliases) > 10:
            self.warnings.append(f"{file_path}: Many aliases ({len(aliases)}), consider reducing")
            
        for alias in aliases:
            if not isinstance(alias, str):
                self.errors.append(f"{file_path}: Alias should be string, got {type(alias)}")
                continue
                
            if len(alias) < 2:
                self.errors.append(f"{file_path}: Alias too short: '{alias}'")
    
    def validate_related(self, related: List[str], file_path: str) -> None:
        """Validate optional related field"""
        if not related:
            return
            
        if not isinstance(related, list):
            self.errors.append(f"{file_path}: 'related' should be a list")
            return
            
        if len(related) > 15:
            self.warnings.append(f"{file_path}: Many related entries ({len(related)}), consider reducing")
            
        for entry in related:
            if not isinstance(entry, str):
                self.errors.append(f"{file_path}: Related entry should be string, got {type(entry)}")
                continue
                
            # Check slug format
            if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', entry):
                self.errors.append(f"{file_path}: Related entry '{entry}' should be kebab-case slug")
    
    def validate_sources(self, sources: List[Dict], file_path: str) -> None:
        """Validate optional sources field"""
        if not sources:
            return
            
        if not isinstance(sources, list):
            self.errors.append(f"{file_path}: 'sources' should be a list")
            return
            
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                self.errors.append(f"{file_path}: Source {i+1} should be an object")
                continue
                
            # Required source fields
            required_source_fields = ['source_url', 'source_title', 'license']
            for field in required_source_fields:
                if field not in source:
                    self.errors.append(f"{file_path}: Source {i+1} missing required field '{field}'")
                    
            # Validate URL format
            if 'source_url' in source and source['source_url']:
                url = source['source_url']
                if not re.match(r'^https?://', url):
                    self.errors.append(f"{file_path}: Source {i+1} URL should start with http:// or https://")
                    
            # Validate license
            valid_licenses = {
                'mit', 'apache-2.0', 'bsd-3-clause', 'gpl-3.0', 'lgpl-3.0',
                'cc-by-4.0', 'cc-by-sa-4.0', 'cc0-1.0', 'proprietary', 'unknown'
            }
            if 'license' in source and source['license']:
                license_val = source['license'].lower()
                if license_val not in valid_licenses:
                    self.warnings.append(f"{file_path}: Source {i+1} license '{license_val}' not recognized")
    
    def validate_file(self, file_path: str) -> bool:
        """Validate a single markdown file's frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                
            metadata = post.metadata
            
            # Core field validation
            self.validate_required_fields(metadata, file_path)
            
            # Individual field validation
            if 'title' in metadata:
                self.validate_title(metadata['title'], file_path)
            if 'slug' in metadata:
                self.validate_slug(metadata['slug'], file_path)
            if 'summary' in metadata:
                self.validate_summary(metadata['summary'], file_path)
            if 'tags' in metadata:
                self.validate_tags(metadata['tags'], file_path)
            if 'updated' in metadata:
                self.validate_updated(metadata['updated'], file_path)
                
            # Optional field validation
            if 'aliases' in metadata:
                self.validate_aliases(metadata['aliases'], file_path)
            if 'related' in metadata:
                self.validate_related(metadata['related'], file_path)
            if 'sources' in metadata:
                self.validate_sources(metadata['sources'], file_path)
                
            return len(self.errors) == 0
            
        except yaml.YAMLError as e:
            self.errors.append(f"{file_path}: Invalid YAML frontmatter: {e}")
            return False
        except Exception as e:
            self.errors.append(f"{file_path}: Error reading file: {e}")
            return False
    
    def validate_files(self, file_paths: List[str]) -> bool:
        """Validate multiple files"""
        all_valid = True
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            if not self.validate_file(file_path):
                all_valid = False
                
        return all_valid
    
    def print_results(self) -> None:
        """Print validation results"""
        if self.errors:
            logger.error("❌ Frontmatter validation errors:")
            for error in self.errors:
                logger.error(f"  {error}")
                
        if self.warnings:
            logger.warning("⚠️  Frontmatter validation warnings:")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
                
        if not self.errors and not self.warnings:
            logger.info("✅ All frontmatter validation passed")
        elif not self.errors:
            logger.info("✅ Frontmatter validation passed (with warnings)")


def main():
    parser = argparse.ArgumentParser(
        description='Validate frontmatter in markdown files for AI Knowledge Website'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Markdown files to validate'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )
    
    args = parser.parse_args()
    
    if not args.files:
        logger.info("No files to validate")
        return 0
        
    # Filter for markdown files
    md_files = [f for f in args.files if f.endswith('.md')]
    
    if not md_files:
        logger.info("No markdown files to validate")
        return 0
        
    validator = FrontmatterValidator()
    is_valid = validator.validate_files(md_files)
    
    validator.print_results()
    
    # Determine exit code
    if not is_valid:
        return 1
    elif args.strict and validator.warnings:
        logger.error("❌ Validation failed due to warnings (strict mode)")
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())