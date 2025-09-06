#!/usr/bin/env python3
"""
Integration Tests for AI Knowledge Website

This script runs comprehensive integration tests that validate the entire
system working together, including the content pipeline and website.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import asyncpg


@dataclass
class TestResult:
    test_name: str
    passed: bool
    duration: float
    error_message: str = ""
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class IntegrationTestRunner:
    def __init__(self, base_url: str, database_url: str = "", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.database_url = database_url
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.temp_dir = None

    def log(self, message: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level.upper()}] {message}")

    async def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """Run a single test and capture results."""
        self.log(f"Running test: {test_name}", 'info')
        start_time = time.time()
        
        try:
            await test_func(*args, **kwargs)
            duration = time.time() - start_time
            result = TestResult(test_name, True, duration)
            self.log(f"✅ {test_name} passed ({duration:.2f}s)", 'info')
            return result
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(test_name, False, duration, str(e))
            self.log(f"❌ {test_name} failed: {str(e)}", 'error')
            return result

    async def test_website_accessibility(self):
        """Test that the website is accessible and responding."""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test main pages
            pages = ['/', '/about', '/articles', '/glossary']
            
            for page in pages:
                url = f"{self.base_url}{page}"
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Page {page} returned status {response.status}")
                    
                    content = await response.text()
                    if len(content) < 100:
                        raise Exception(f"Page {page} has suspiciously short content")
                    
                    # Check for basic HTML structure
                    if not all(tag in content for tag in ['<html', '<head', '<body']):
                        raise Exception(f"Page {page} missing basic HTML structure")

    async def test_content_schema_validation(self):
        """Test that content follows the expected schema."""
        # This would typically connect to the Astro dev server or check built files
        content_dir = Path("apps/site/src/content")
        if not content_dir.exists():
            raise Exception("Content directory not found")
        
        # Check for required content structure
        required_dirs = ["articles", "glossary"]
        for dir_name in required_dirs:
            dir_path = content_dir / dir_name
            if not dir_path.exists():
                raise Exception(f"Required content directory missing: {dir_name}")
        
        # Validate some content files
        md_files = list(content_dir.rglob("*.md"))
        if len(md_files) == 0:
            raise Exception("No markdown content files found")
        
        # Sample validation of frontmatter
        import yaml
        for md_file in md_files[:5]:  # Check first 5 files
            with open(md_file, 'r') as f:
                content = f.read()
            
            if content.startswith('---'):
                try:
                    parts = content.split('---', 2)
                    frontmatter = yaml.safe_load(parts[1])
                    
                    if not isinstance(frontmatter, dict):
                        raise Exception(f"Invalid frontmatter in {md_file}")
                    
                    if 'title' not in frontmatter:
                        raise Exception(f"Missing title in {md_file}")
                        
                except yaml.YAMLError as e:
                    raise Exception(f"Invalid YAML frontmatter in {md_file}: {e}")

    async def test_database_connectivity(self):
        """Test database connectivity and basic operations."""
        if not self.database_url:
            self.log("Skipping database test - no database URL provided", 'info')
            return
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise Exception("Basic database query failed")
            
            # Test if tables exist (basic schema check)
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            tables = await conn.fetch(tables_query)
            table_names = [row['table_name'] for row in tables]
            
            # Check for expected tables (adjust based on your schema)
            expected_tables = ['articles', 'sources', 'content_items']  # Adjust as needed
            missing_tables = [t for t in expected_tables if t not in table_names]
            
            if missing_tables and len(table_names) == 0:
                # If no tables exist, it might be a fresh database
                self.log("Database appears to be empty - this might be expected for testing", 'warning')
            elif missing_tables:
                raise Exception(f"Missing expected tables: {missing_tables}")
            
            await conn.close()
            
        except Exception as e:
            raise Exception(f"Database connectivity test failed: {e}")

    async def test_content_pipeline_basic(self):
        """Test basic content pipeline functionality."""
        # This is a simplified test - in a real scenario, you might want to:
        # 1. Create test content
        # 2. Run the pipeline
        # 3. Verify output
        
        pipeline_dir = Path("pipelines")
        if not pipeline_dir.exists():
            raise Exception("Pipeline directory not found")
        
        # Check for key pipeline components
        required_files = [
            "requirements.txt",
            "config.py",
            "__init__.py"
        ]
        
        for file_name in required_files:
            file_path = pipeline_dir / file_name
            if not file_path.exists():
                raise Exception(f"Required pipeline file missing: {file_name}")
        
        # Test imports (basic syntax check)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", pipeline_dir / "config.py")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
        except Exception as e:
            raise Exception(f"Pipeline configuration import failed: {e}")

    async def test_api_endpoints(self):
        """Test any API endpoints that might exist."""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test common API patterns
            api_endpoints = [
                '/api/health',
                '/api/search',
                '/.well-known/security.txt'
            ]
            
            found_endpoints = []
            for endpoint in api_endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url) as response:
                        if response.status < 500:  # Don't fail on 404s, but fail on server errors
                            found_endpoints.append(endpoint)
                except:
                    pass  # Endpoint doesn't exist or is unreachable
            
            # For now, we don't require any specific endpoints to exist
            # But we log what we found
            if found_endpoints:
                self.log(f"Found API endpoints: {found_endpoints}", 'info')

    async def test_performance_thresholds(self):
        """Test that key pages meet performance thresholds."""
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test page load times
            pages = ['/']  # Start with just homepage
            
            for page in pages:
                url = f"{self.base_url}{page}"
                start_time = time.time()
                
                async with session.get(url) as response:
                    content = await response.read()
                    load_time = time.time() - start_time
                    
                    # Performance thresholds
                    max_load_time = 5.0  # 5 seconds max
                    max_content_size = 2 * 1024 * 1024  # 2MB max
                    
                    if load_time > max_load_time:
                        raise Exception(f"Page {page} load time {load_time:.2f}s exceeds threshold {max_load_time}s")
                    
                    if len(content) > max_content_size:
                        raise Exception(f"Page {page} size {len(content)} bytes exceeds threshold {max_content_size} bytes")

    async def test_content_freshness(self):
        """Test that content is reasonably fresh."""
        content_dir = Path("apps/site/src/content")
        if not content_dir.exists():
            raise Exception("Content directory not found")
        
        # Check modification times
        md_files = list(content_dir.rglob("*.md"))
        if not md_files:
            raise Exception("No content files found")
        
        # Check that at least some content is recent
        import os
        now = time.time()
        recent_threshold = 90 * 24 * 60 * 60  # 90 days
        
        recent_files = []
        for md_file in md_files:
            mtime = os.path.getmtime(md_file)
            if now - mtime < recent_threshold:
                recent_files.append(md_file)
        
        if len(recent_files) == 0:
            self.log("Warning: No content files modified in the last 90 days", 'warning')
        
        # This is more of a warning than a failure
        freshness_ratio = len(recent_files) / len(md_files)
        if freshness_ratio < 0.1:  # Less than 10% recent content
            self.log(f"Warning: Only {freshness_ratio:.1%} of content is recent", 'warning')

    async def test_seo_basics(self):
        """Test basic SEO requirements."""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Test homepage SEO
            url = self.base_url
            async with session.get(url) as response:
                content = await response.text()
                
                # Check for basic SEO elements
                seo_checks = {
                    'title': '<title>' in content,
                    'meta_description': 'name="description"' in content,
                    'meta_viewport': 'name="viewport"' in content,
                    'lang_attribute': 'lang=' in content,
                    'h1_tag': '<h1' in content
                }
                
                missing_seo = [item for item, present in seo_checks.items() if not present]
                if missing_seo:
                    raise Exception(f"Missing SEO elements: {missing_seo}")

    async def test_build_artifacts(self):
        """Test that build artifacts are properly generated."""
        build_dir = Path("apps/site/dist")
        if not build_dir.exists():
            raise Exception("Build directory not found - run 'npm run build' first")
        
        # Check for essential files
        essential_files = [
            "index.html",
            "about/index.html",
            "articles/index.html",
            "glossary/index.html"
        ]
        
        for file_path in essential_files:
            full_path = build_dir / file_path
            if not full_path.exists():
                raise Exception(f"Essential build file missing: {file_path}")
        
        # Check index.html content
        index_path = build_dir / "index.html"
        with open(index_path, 'r') as f:
            content = f.read()
        
        if len(content) < 500:
            raise Exception("index.html appears to be too short")
        
        # Check for minification (basic check)
        if content.count('\n') > content.count('<') / 2:
            self.log("HTML doesn't appear to be minified", 'warning')

    async def run_all_tests(self):
        """Run all integration tests."""
        self.log("Starting integration tests...", 'info')
        
        # Define test suite
        test_suite = [
            (self.test_website_accessibility, "Website Accessibility"),
            (self.test_content_schema_validation, "Content Schema Validation"),
            (self.test_database_connectivity, "Database Connectivity"),
            (self.test_content_pipeline_basic, "Content Pipeline Basic"),
            (self.test_api_endpoints, "API Endpoints"),
            (self.test_performance_thresholds, "Performance Thresholds"),
            (self.test_content_freshness, "Content Freshness"),
            (self.test_seo_basics, "SEO Basics"),
            (self.test_build_artifacts, "Build Artifacts")
        ]
        
        # Run tests
        self.results = []
        for test_func, test_name in test_suite:
            result = await self.run_test(test_func, test_name)
            self.results.append(result)
        
        return self.results

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Show individual results
        print(f"\nTEST RESULTS:")
        print("-" * 40)
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.test_name} ({result.duration:.2f}s)")
            if not result.passed:
                print(f"      Error: {result.error_message}")
        
        # Show failures in detail
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            print(f"\nFAILURE DETAILS:")
            print("-" * 40)
            for result in failed_results:
                print(f"❌ {result.test_name}")
                print(f"   Error: {result.error_message}")
                if result.details:
                    print(f"   Details: {result.details}")
                print()

    def save_report(self, output_file: Path):
        """Save test results to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'database_url': '***' if self.database_url else None,
            'summary': {
                'total_tests': len(self.results),
                'passed_tests': sum(1 for r in self.results if r.passed),
                'failed_tests': sum(1 for r in self.results if not r.passed),
                'total_duration': sum(r.duration for r in self.results),
                'success_rate': (sum(1 for r in self.results if r.passed) / len(self.results)) * 100 if self.results else 0
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'error_message': r.error_message,
                    'details': r.details
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Test report saved to {output_file}", 'info')


async def main():
    parser = argparse.ArgumentParser(description='Integration tests for AI Knowledge Website')
    parser.add_argument('--base-url',
                       type=str,
                       default='http://localhost:4321',
                       help='Base URL for the website')
    parser.add_argument('--database-url',
                       type=str,
                       help='Database connection URL')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--output-report',
                       type=str,
                       help='Save test report to JSON file')
    parser.add_argument('--fail-fast',
                       action='store_true',
                       help='Stop on first test failure')
    
    args = parser.parse_args()
    
    # Get database URL from environment if not provided
    database_url = args.database_url or os.environ.get('DATABASE_URL', '')
    
    # Create test runner
    runner = IntegrationTestRunner(
        base_url=args.base_url,
        database_url=database_url,
        verbose=args.verbose
    )
    
    try:
        # Run tests
        results = await runner.run_all_tests()
        
        # Print summary
        runner.print_summary()
        
        # Save report if requested
        if args.output_report:
            runner.save_report(Path(args.output_report))
        
        # Determine exit code
        failed_tests = sum(1 for r in results if not r.passed)
        if failed_tests > 0:
            print(f"\n❌ Integration tests FAILED ({failed_tests} failures)")
            sys.exit(1)
        else:
            print(f"\n✅ All integration tests PASSED")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())