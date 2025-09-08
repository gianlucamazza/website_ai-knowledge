#!/usr/bin/env python3
"""
Content Validation Script for AI Knowledge Website

This script validates content files against schemas, checks for required metadata,
and ensures content quality standards are met.
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime

import yaml
# Use safe loader to prevent arbitrary code execution
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


@dataclass
class ValidationError:
    file_path: str
    error_type: str
    message: str
    severity: str  # 'error', 'warning', 'info'


@dataclass
class ContentStats:
    total_files: int = 0
    valid_files: int = 0
    errors: int = 0
    warnings: int = 0
    word_count: int = 0
    categories: Set[str] = None
    tags: Set[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = set()
        if self.tags is None:
            self.tags = set()


class ContentValidator:
    def __init__(self, content_dir: Path, verbose: bool = False):
        self.content_dir = Path(content_dir)
        self.verbose = verbose
        self.errors: List[ValidationError] = []
        self.stats = ContentStats()
        
        # Content type configurations
        self.content_configs = {
            'articles': {
                'required_fields': ['title', 'description', 'pubDate', 'author'],
                'optional_fields': ['tags', 'category', 'image', 'draft'],
                'content_min_words': 100,
                'title_max_length': 100,
                'description_max_length': 200,
            },
            'glossary': {
                'required_fields': ['title', 'description'],
                'optional_fields': ['tags', 'category', 'related'],
                'content_min_words': 20,
                'title_max_length': 80,
                'description_max_length': 150,
            }
        }

    def log(self, message: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level.upper()}] {message}")

    def add_error(self, file_path: str, error_type: str, message: str, severity: str = 'error'):
        error = ValidationError(file_path, error_type, message, severity)
        self.errors.append(error)
        
        if severity == 'error':
            self.stats.errors += 1
        elif severity == 'warning':
            self.stats.warnings += 1

    def extract_frontmatter(self, content: str) -> tuple[Optional[Dict], str]:
        """Extract YAML frontmatter from markdown content."""
        if not content.strip().startswith('---'):
            return None, content
        
        try:
            # Split content by frontmatter delimiters
            parts = content.split('---', 2)
            if len(parts) < 3:
                return None, content
            
            frontmatter = yaml.load(parts[1], Loader=SafeLoader)
            body = parts[2].strip()
            
            return frontmatter, body
            
        except yaml.YAMLError as e:
            return None, content

    def count_words(self, text: str) -> int:
        """Count words in text, excluding frontmatter."""
        # Remove markdown formatting
        text = re.sub(r'[#*_`\[\]()]', '', text)
        # Split by whitespace and filter empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)

    def validate_frontmatter(self, file_path: Path, frontmatter: Dict, content_type: str) -> bool:
        """Validate frontmatter against content type requirements."""
        config = self.content_configs.get(content_type, {})
        required_fields = config.get('required_fields', [])
        
        is_valid = True
        
        # Check required fields
        for field in required_fields:
            if field not in frontmatter:
                self.add_error(
                    str(file_path),
                    'missing_required_field',
                    f"Missing required field: {field}",
                    'error'
                )
                is_valid = False
            elif not frontmatter[field]:
                self.add_error(
                    str(file_path),
                    'empty_required_field',
                    f"Empty required field: {field}",
                    'error'
                )
                is_valid = False

        # Validate field lengths
        title_max = config.get('title_max_length', 200)
        desc_max = config.get('description_max_length', 300)
        
        if 'title' in frontmatter and isinstance(frontmatter['title'], str):
            if len(frontmatter['title']) > title_max:
                self.add_error(
                    str(file_path),
                    'title_too_long',
                    f"Title exceeds {title_max} characters ({len(frontmatter['title'])} chars)",
                    'warning'
                )
        
        if 'description' in frontmatter and isinstance(frontmatter['description'], str):
            if len(frontmatter['description']) > desc_max:
                self.add_error(
                    str(file_path),
                    'description_too_long',
                    f"Description exceeds {desc_max} characters ({len(frontmatter['description'])} chars)",
                    'warning'
                )

        # Validate date format for articles
        if content_type == 'articles' and 'pubDate' in frontmatter:
            try:
                if isinstance(frontmatter['pubDate'], str):
                    datetime.fromisoformat(frontmatter['pubDate'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                self.add_error(
                    str(file_path),
                    'invalid_date_format',
                    f"Invalid date format: {frontmatter['pubDate']}",
                    'error'
                )
                is_valid = False

        # Collect stats
        if 'category' in frontmatter:
            self.stats.categories.add(str(frontmatter['category']))
        
        if 'tags' in frontmatter:
            if isinstance(frontmatter['tags'], list):
                self.stats.tags.update(frontmatter['tags'])
            elif isinstance(frontmatter['tags'], str):
                self.stats.tags.add(frontmatter['tags'])

        return is_valid

    def validate_content_body(self, file_path: Path, body: str, content_type: str) -> bool:
        """Validate the content body."""
        config = self.content_configs.get(content_type, {})
        min_words = config.get('content_min_words', 50)
        
        is_valid = True
        word_count = self.count_words(body)
        self.stats.word_count += word_count
        
        if word_count < min_words:
            self.add_error(
                str(file_path),
                'insufficient_content',
                f"Content too short: {word_count} words (minimum: {min_words})",
                'warning'
            )
        
        # Check for common markdown issues
        if '](http' in body and not re.search(r'\]\(https?://[^\s\)]+\)', body):
            self.add_error(
                str(file_path),
                'malformed_links',
                "Potentially malformed markdown links detected",
                'warning'
            )
        
        # Check for empty headers
        empty_headers = re.findall(r'^#+\s*$', body, re.MULTILINE)
        if empty_headers:
            self.add_error(
                str(file_path),
                'empty_headers',
                f"Found {len(empty_headers)} empty headers",
                'warning'
            )
        
        # Check for duplicate headers
        headers = re.findall(r'^#+\s+(.+)', body, re.MULTILINE)
        if len(headers) != len(set(headers)):
            self.add_error(
                str(file_path),
                'duplicate_headers',
                "Duplicate headers found in content",
                'warning'
            )

        return is_valid

    def validate_file(self, file_path: Path) -> bool:
        """Validate a single markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            self.add_error(
                str(file_path),
                'file_read_error',
                f"Could not read file: {e}",
                'error'
            )
            return False

        # Determine content type from path
        content_type = 'articles' if 'articles' in file_path.parts else 'glossary'
        
        # Extract and validate frontmatter
        frontmatter, body = self.extract_frontmatter(content)
        
        is_valid = True
        
        if frontmatter is None:
            self.add_error(
                str(file_path),
                'no_frontmatter',
                "No valid YAML frontmatter found",
                'error'
            )
            is_valid = False
        else:
            if not self.validate_frontmatter(file_path, frontmatter, content_type):
                is_valid = False
        
        # Validate content body
        if not self.validate_content_body(file_path, body, content_type):
            is_valid = False
        
        if is_valid:
            self.stats.valid_files += 1
            self.log(f"✓ {file_path.name}", 'info')
        else:
            self.log(f"✗ {file_path.name}", 'error')
        
        return is_valid

    def validate_schema_files(self) -> bool:
        """Validate schema configuration files."""
        schema_files = [
            self.content_dir / 'config.ts',
            self.content_dir / 'taxonomies' / 'categories.json',
            self.content_dir / 'taxonomies' / 'tags.json'
        ]
        
        is_valid = True
        
        for schema_file in schema_files:
            if schema_file.exists():
                self.log(f"Found schema file: {schema_file.name}", 'info')
                
                # Basic validation for JSON files
                if schema_file.suffix == '.json':
                    try:
                        with open(schema_file, 'r') as f:
                            json.load(f)
                        self.log(f"✓ {schema_file.name} is valid JSON", 'info')
                    except json.JSONDecodeError as e:
                        self.add_error(
                            str(schema_file),
                            'invalid_json',
                            f"Invalid JSON: {e}",
                            'error'
                        )
                        is_valid = False
            else:
                self.add_error(
                    str(schema_file),
                    'missing_schema_file',
                    f"Schema file not found: {schema_file.name}",
                    'warning'
                )
        
        return is_valid

    def check_duplicate_titles(self) -> bool:
        """Check for duplicate titles across all content."""
        titles = {}
        
        for file_path in self.content_dir.rglob('*.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = self.extract_frontmatter(content)
                if frontmatter and 'title' in frontmatter:
                    title = frontmatter['title'].lower().strip()
                    if title in titles:
                        self.add_error(
                            str(file_path),
                            'duplicate_title',
                            f"Duplicate title found: '{frontmatter['title']}' (also in {titles[title]})",
                            'warning'
                        )
                    else:
                        titles[title] = file_path.name
            except Exception as e:
                self.log(f"Error checking duplicates in {file_path}: {e}", 'error')
        
        return len([e for e in self.errors if e.error_type == 'duplicate_title']) == 0

    def validate_all(self) -> bool:
        """Validate all content files."""
        self.log("Starting content validation...", 'info')
        
        # Find all markdown files
        md_files = list(self.content_dir.rglob('*.md'))
        self.stats.total_files = len(md_files)
        
        if not md_files:
            self.log("No markdown files found for validation", 'warning')
            return False
        
        self.log(f"Found {len(md_files)} markdown files to validate", 'info')
        
        # Validate individual files
        all_valid = True
        for file_path in md_files:
            if not self.validate_file(file_path):
                all_valid = False
        
        # Validate schema files
        if not self.validate_schema_files():
            all_valid = False
        
        # Check for duplicates
        if not self.check_duplicate_titles():
            all_valid = False
        
        return all_valid

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*50)
        print("CONTENT VALIDATION SUMMARY")
        print("="*50)
        
        print(f"Total files processed: {self.stats.total_files}")
        print(f"Valid files: {self.stats.valid_files}")
        print(f"Files with errors: {self.stats.total_files - self.stats.valid_files}")
        print(f"Total errors: {self.stats.errors}")
        print(f"Total warnings: {self.stats.warnings}")
        print(f"Total word count: {self.stats.word_count:,}")
        
        if self.stats.categories:
            print(f"Categories found: {len(self.stats.categories)}")
            if self.verbose:
                print(f"  {', '.join(sorted(self.stats.categories))}")
        
        if self.stats.tags:
            print(f"Tags found: {len(self.stats.tags)}")
            if self.verbose and len(self.stats.tags) <= 20:
                print(f"  {', '.join(sorted(self.stats.tags))}")
        
        # Print errors by type
        if self.errors:
            print(f"\nERRORS AND WARNINGS:")
            error_types = {}
            for error in self.errors:
                if error.error_type not in error_types:
                    error_types[error.error_type] = {'error': 0, 'warning': 0}
                error_types[error.error_type][error.severity] += 1
            
            for error_type, counts in sorted(error_types.items()):
                total = counts['error'] + counts['warning']
                print(f"  {error_type}: {total} ({counts['error']} errors, {counts['warning']} warnings)")
        
        # Print individual errors if verbose
        if self.verbose and self.errors:
            print(f"\nDETAILED ERRORS:")
            for error in sorted(self.errors, key=lambda x: (x.severity, x.file_path)):
                icon = "❌" if error.severity == 'error' else "⚠️"
                print(f"  {icon} {error.file_path}: {error.message}")

    def save_report(self, output_file: Path):
        """Save validation report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'total_files': self.stats.total_files,
                'valid_files': self.stats.valid_files,
                'errors': self.stats.errors,
                'warnings': self.stats.warnings,
                'word_count': self.stats.word_count,
                'categories': sorted(list(self.stats.categories)),
                'tags': sorted(list(self.stats.tags))
            },
            'errors': [
                {
                    'file_path': error.file_path,
                    'error_type': error.error_type,
                    'message': error.message,
                    'severity': error.severity
                }
                for error in self.errors
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Validation report saved to {output_file}", 'info')


def main():
    parser = argparse.ArgumentParser(description='Validate AI Knowledge Website content')
    parser.add_argument('--content-dir', 
                       type=str, 
                       default='apps/site/src/content',
                       help='Path to content directory')
    parser.add_argument('--verbose', '-v', 
                       action='store_true', 
                       help='Verbose output')
    parser.add_argument('--output-report', 
                       type=str, 
                       help='Save detailed report to JSON file')
    parser.add_argument('--fail-on-warnings', 
                       action='store_true', 
                       help='Exit with error code if warnings are found')
    
    args = parser.parse_args()
    
    # Resolve content directory path
    content_dir = Path(args.content_dir)
    if not content_dir.exists():
        print(f"Error: Content directory not found: {content_dir}")
        sys.exit(1)
    
    # Run validation
    validator = ContentValidator(content_dir, verbose=args.verbose)
    is_valid = validator.validate_all()
    
    # Print summary
    validator.print_summary()
    
    # Save report if requested
    if args.output_report:
        validator.save_report(Path(args.output_report))
    
    # Determine exit code
    if not is_valid:
        print(f"\n❌ Content validation FAILED with {validator.stats.errors} errors")
        sys.exit(1)
    elif validator.stats.warnings > 0 and args.fail_on_warnings:
        print(f"\n⚠️ Content validation completed with {validator.stats.warnings} warnings (--fail-on-warnings enabled)")
        sys.exit(1)
    else:
        print(f"\n✅ Content validation PASSED")
        if validator.stats.warnings > 0:
            print(f"   ({validator.stats.warnings} warnings found)")
        sys.exit(0)


if __name__ == '__main__':
    main()