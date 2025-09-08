#!/usr/bin/env python3
"""
Link Checker Script for AI Knowledge Website

This script validates all internal and external links in content files,
checks for broken references, and generates a comprehensive link report.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlparse
import time

import aiohttp
import yaml
# Use safe loader to prevent arbitrary code execution
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


@dataclass
class LinkResult:
    url: str
    status_code: Optional[int]
    is_valid: bool
    error_message: str = ""
    response_time: float = 0.0
    redirect_url: str = ""


@dataclass
class LinkReference:
    source_file: str
    line_number: int
    url: str
    link_text: str
    link_type: str  # 'internal', 'external', 'anchor'


class LinkChecker:
    def __init__(self, content_dir: Path, base_url: str = "", timeout: int = 10, max_concurrent: int = 10):
        self.content_dir = Path(content_dir)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.verbose = False
        
        # Results storage
        self.all_links: List[LinkReference] = []
        self.link_results: Dict[str, LinkResult] = {}
        self.internal_files: Set[str] = set()
        self.stats = {
            'total_links': 0,
            'internal_links': 0,
            'external_links': 0,
            'anchor_links': 0,
            'valid_links': 0,
            'invalid_links': 0,
            'broken_external': 0,
            'broken_internal': 0,
            'redirects': 0,
            'files_processed': 0
        }

    def log(self, message: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level.upper()}] {message}")

    def extract_frontmatter(self, content: str) -> tuple[Optional[Dict], str]:
        """Extract YAML frontmatter from markdown content."""
        if not content.strip().startswith('---'):
            return None, content
        
        try:
            parts = content.split('---', 2)
            if len(parts) < 3:
                return None, content
            
            frontmatter = yaml.load(parts[1], Loader=SafeLoader)
            body = parts[2].strip()
            
            return frontmatter, body
            
        except yaml.YAMLError:
            return None, content

    def build_internal_file_index(self):
        """Build an index of all internal content files."""
        self.log("Building internal file index...", 'info')
        
        # Find all markdown files
        md_files = list(self.content_dir.rglob('*.md'))
        
        for file_path in md_files:
            # Convert file path to URL path
            relative_path = file_path.relative_to(self.content_dir)
            
            # Handle different content types
            if 'articles' in relative_path.parts:
                # Articles: /articles/filename (without .md)
                url_path = f"/articles/{relative_path.stem}"
            elif 'glossary' in relative_path.parts:
                # Glossary: /glossary/filename (without .md)
                url_path = f"/glossary/{relative_path.stem}"
            else:
                # Generic: maintain path structure
                url_path = f"/{relative_path.with_suffix('')}"
            
            self.internal_files.add(url_path)
            self.internal_files.add(url_path + '/')  # Also add with trailing slash
        
        # Add common pages
        self.internal_files.update([
            '/', '/about', '/articles', '/glossary',
            '/about/', '/articles/', '/glossary/'
        ])
        
        self.log(f"Indexed {len(self.internal_files)} internal paths", 'info')

    def extract_links_from_content(self, file_path: Path, content: str) -> List[LinkReference]:
        """Extract all links from markdown content."""
        links = []
        
        # Extract frontmatter and body
        frontmatter, body = self.extract_frontmatter(content)
        
        # Find markdown links: [text](url)
        markdown_links = re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', body, re.MULTILINE)
        for match in markdown_links:
            link_text = match.group(1)
            url = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            
            link_type = self.classify_link(url)
            links.append(LinkReference(
                source_file=str(file_path.relative_to(self.content_dir)),
                line_number=line_number,
                url=url,
                link_text=link_text,
                link_type=link_type
            ))
        
        # Find HTML links: <a href="url">text</a>
        html_links = re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', body, re.MULTILINE)
        for match in html_links:
            url = match.group(1).strip()
            link_text = match.group(2)
            line_number = content[:match.start()].count('\n') + 1
            
            link_type = self.classify_link(url)
            links.append(LinkReference(
                source_file=str(file_path.relative_to(self.content_dir)),
                line_number=line_number,
                url=url,
                link_text=link_text,
                link_type=link_type
            ))
        
        # Find reference-style links: [text]: url
        ref_links = re.finditer(r'^\[([^\]]+)\]:\s*(.+)$', body, re.MULTILINE)
        for match in ref_links:
            link_text = match.group(1)
            url = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            
            link_type = self.classify_link(url)
            links.append(LinkReference(
                source_file=str(file_path.relative_to(self.content_dir)),
                line_number=line_number,
                url=url,
                link_text=link_text,
                link_type=link_type
            ))
        
        return links

    def classify_link(self, url: str) -> str:
        """Classify link as internal, external, or anchor."""
        url = url.strip()
        
        # Skip empty or invalid URLs
        if not url or url in ['#', 'javascript:', 'mailto:']:
            return 'anchor'
        
        # Anchor links
        if url.startswith('#'):
            return 'anchor'
        
        # Absolute URLs
        if url.startswith(('http://', 'https://')):
            if self.base_url and url.startswith(self.base_url):
                return 'internal'
            return 'external'
        
        # Protocol-relative URLs
        if url.startswith('//'):
            return 'external'
        
        # Email links
        if url.startswith('mailto:'):
            return 'external'
        
        # Relative URLs (internal)
        return 'internal'

    def validate_internal_link(self, url: str) -> LinkResult:
        """Validate an internal link."""
        # Clean the URL
        clean_url = url.split('#')[0]  # Remove anchor
        clean_url = clean_url.split('?')[0]  # Remove query params
        
        # Normalize path
        if not clean_url.startswith('/'):
            clean_url = '/' + clean_url
        
        # Check if file exists in our index
        is_valid = (clean_url in self.internal_files or 
                   clean_url.rstrip('/') in self.internal_files or
                   clean_url + '/' in self.internal_files)
        
        return LinkResult(
            url=url,
            status_code=200 if is_valid else 404,
            is_valid=is_valid,
            error_message="" if is_valid else "Internal link target not found"
        )

    async def validate_external_link(self, session: aiohttp.ClientSession, url: str) -> LinkResult:
        """Validate an external link."""
        start_time = time.time()
        
        try:
            async with session.head(url, timeout=self.timeout, allow_redirects=True) as response:
                response_time = time.time() - start_time
                
                # Consider 2xx and 3xx as valid
                is_valid = response.status < 400
                redirect_url = str(response.url) if response.url != url else ""
                
                return LinkResult(
                    url=url,
                    status_code=response.status,
                    is_valid=is_valid,
                    response_time=response_time,
                    redirect_url=redirect_url
                )
        
        except asyncio.TimeoutError:
            return LinkResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message="Request timeout",
                response_time=time.time() - start_time
            )
        
        except aiohttp.ClientError as e:
            return LinkResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )
        
        except Exception as e:
            return LinkResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                response_time=time.time() - start_time
            )

    async def validate_external_links(self, external_urls: Set[str]):
        """Validate all external links concurrently."""
        if not external_urls:
            return
        
        self.log(f"Validating {len(external_urls)} external links...", 'info')
        
        # Configure session with reasonable defaults
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'AI-Knowledge-Website-Link-Checker/1.0'
            }
        ) as session:
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def validate_with_semaphore(url):
                async with semaphore:
                    return await self.validate_external_link(session, url)
            
            # Run validations concurrently
            tasks = [validate_with_semaphore(url) for url in external_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for url, result in zip(external_urls, results):
                if isinstance(result, LinkResult):
                    self.link_results[url] = result
                    if result.is_valid:
                        self.log(f"✓ {url} ({result.status_code})", 'info')
                    else:
                        self.log(f"✗ {url} - {result.error_message}", 'warning')
                else:
                    # Handle exceptions
                    self.link_results[url] = LinkResult(
                        url=url,
                        status_code=None,
                        is_valid=False,
                        error_message=f"Validation failed: {str(result)}"
                    )
                    self.log(f"✗ {url} - Exception: {str(result)}", 'error')

    def process_file(self, file_path: Path):
        """Process a single markdown file for links."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            links = self.extract_links_from_content(file_path, content)
            self.all_links.extend(links)
            
            self.stats['files_processed'] += 1
            self.log(f"Processed {file_path.name}: {len(links)} links found", 'info')
            
        except Exception as e:
            self.log(f"Error processing {file_path}: {str(e)}", 'error')

    async def check_all_links(self):
        """Check all links in the content directory."""
        self.log("Starting link validation...", 'info')
        
        # Build internal file index
        self.build_internal_file_index()
        
        # Process all markdown files
        md_files = list(self.content_dir.rglob('*.md'))
        for file_path in md_files:
            self.process_file(file_path)
        
        # Collect unique URLs by type
        internal_urls = set()
        external_urls = set()
        anchor_urls = set()
        
        for link in self.all_links:
            if link.link_type == 'internal':
                internal_urls.add(link.url)
            elif link.link_type == 'external':
                external_urls.add(link.url)
            elif link.link_type == 'anchor':
                anchor_urls.add(link.url)
        
        # Update stats
        self.stats['total_links'] = len(self.all_links)
        self.stats['internal_links'] = len(internal_urls)
        self.stats['external_links'] = len(external_urls)
        self.stats['anchor_links'] = len(anchor_urls)
        
        # Validate internal links
        self.log(f"Validating {len(internal_urls)} internal links...", 'info')
        for url in internal_urls:
            result = self.validate_internal_link(url)
            self.link_results[url] = result
            
            if result.is_valid:
                self.log(f"✓ {url}", 'info')
            else:
                self.log(f"✗ {url} - {result.error_message}", 'warning')
        
        # Validate external links
        await self.validate_external_links(external_urls)
        
        # Calculate final stats
        valid_links = sum(1 for result in self.link_results.values() if result.is_valid)
        invalid_links = len(self.link_results) - valid_links
        
        self.stats['valid_links'] = valid_links
        self.stats['invalid_links'] = invalid_links
        self.stats['broken_internal'] = sum(1 for url, result in self.link_results.items() 
                                          if not result.is_valid and url in internal_urls)
        self.stats['broken_external'] = sum(1 for url, result in self.link_results.items() 
                                          if not result.is_valid and url in external_urls)
        self.stats['redirects'] = sum(1 for result in self.link_results.values() 
                                    if result.redirect_url)

    def print_summary(self):
        """Print link checking summary."""
        print("\n" + "="*60)
        print("LINK VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total links found: {self.stats['total_links']}")
        print(f"  Internal links: {self.stats['internal_links']}")
        print(f"  External links: {self.stats['external_links']}")
        print(f"  Anchor links: {self.stats['anchor_links']}")
        print()
        print(f"Validation results:")
        print(f"  Valid links: {self.stats['valid_links']}")
        print(f"  Invalid links: {self.stats['invalid_links']}")
        print(f"  Broken internal: {self.stats['broken_internal']}")
        print(f"  Broken external: {self.stats['broken_external']}")
        print(f"  Redirects: {self.stats['redirects']}")
        
        # Show broken links
        broken_links = [(url, result) for url, result in self.link_results.items() 
                       if not result.is_valid]
        
        if broken_links:
            print(f"\nBROKEN LINKS ({len(broken_links)}):")
            print("-" * 40)
            
            # Group by source file
            broken_by_file = {}
            for link in self.all_links:
                if link.url in self.link_results and not self.link_results[link.url].is_valid:
                    if link.source_file not in broken_by_file:
                        broken_by_file[link.source_file] = []
                    broken_by_file[link.source_file].append(link)
            
            for file_path, file_links in sorted(broken_by_file.items()):
                print(f"\n📄 {file_path}:")
                for link in file_links:
                    result = self.link_results[link.url]
                    status = f"({result.status_code})" if result.status_code else ""
                    print(f"  ❌ Line {link.line_number}: {link.url} {status}")
                    print(f"     Text: '{link.link_text}'")
                    if result.error_message:
                        print(f"     Error: {result.error_message}")
        
        # Show redirects
        redirects = [(url, result) for url, result in self.link_results.items() 
                    if result.redirect_url and result.redirect_url != url]
        
        if redirects and self.verbose:
            print(f"\nREDIRECTS ({len(redirects)}):")
            print("-" * 40)
            for url, result in redirects[:10]:  # Show first 10
                print(f"  {url} → {result.redirect_url}")
            if len(redirects) > 10:
                print(f"  ... and {len(redirects) - 10} more")

    def save_report(self, output_file: Path):
        """Save detailed link checking report to JSON file."""
        # Prepare detailed link information
        detailed_links = []
        for link in self.all_links:
            result = self.link_results.get(link.url, LinkResult(link.url, None, False, "Not validated"))
            detailed_links.append({
                'source_file': link.source_file,
                'line_number': link.line_number,
                'url': link.url,
                'link_text': link.link_text,
                'link_type': link.link_type,
                'is_valid': result.is_valid,
                'status_code': result.status_code,
                'error_message': result.error_message,
                'response_time': result.response_time,
                'redirect_url': result.redirect_url
            })
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'links': detailed_links,
            'broken_links': [
                {
                    'url': url,
                    'status_code': result.status_code,
                    'error_message': result.error_message,
                    'sources': [
                        {
                            'file': link.source_file,
                            'line': link.line_number,
                            'text': link.link_text
                        }
                        for link in self.all_links if link.url == url
                    ]
                }
                for url, result in self.link_results.items() if not result.is_valid
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Detailed report saved to {output_file}", 'info')


async def main():
    parser = argparse.ArgumentParser(description='Check links in AI Knowledge Website content')
    parser.add_argument('--content-dir',
                       type=str,
                       default='apps/site/src/content',
                       help='Path to content directory')
    parser.add_argument('--base-url',
                       type=str,
                       default='https://ai-knowledge.example.com',
                       help='Base URL for the website')
    parser.add_argument('--timeout',
                       type=int,
                       default=10,
                       help='Timeout for external link checks (seconds)')
    parser.add_argument('--max-concurrent',
                       type=int,
                       default=10,
                       help='Maximum concurrent external link checks')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--output-report',
                       type=str,
                       help='Save detailed report to JSON file')
    parser.add_argument('--fail-on-broken',
                       action='store_true',
                       help='Exit with error code if broken links are found')
    
    args = parser.parse_args()
    
    # Resolve content directory path
    content_dir = Path(args.content_dir)
    if not content_dir.exists():
        print(f"Error: Content directory not found: {content_dir}")
        sys.exit(1)
    
    # Create link checker
    checker = LinkChecker(
        content_dir=content_dir,
        base_url=args.base_url,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent
    )
    checker.verbose = args.verbose
    
    # Run link checking
    await checker.check_all_links()
    
    # Print summary
    checker.print_summary()
    
    # Save report if requested
    if args.output_report:
        checker.save_report(Path(args.output_report))
    
    # Determine exit code
    if checker.stats['invalid_links'] > 0:
        if args.fail_on_broken:
            print(f"\n❌ Link validation FAILED - {checker.stats['invalid_links']} broken links found")
            sys.exit(1)
        else:
            print(f"\n⚠️ Link validation completed with {checker.stats['invalid_links']} broken links")
    else:
        print(f"\n✅ Link validation PASSED - All {checker.stats['valid_links']} links are valid")
    
    sys.exit(0)


if __name__ == '__main__':
    asyncio.run(main())