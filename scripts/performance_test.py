#!/usr/bin/env python3
"""
Performance Testing Script for AI Knowledge Website

This script performs comprehensive performance testing including:
- Page load times
- Core Web Vitals simulation  
- Resource optimization checks
- Accessibility testing
- SEO validation
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

import aiohttp
from urllib.parse import urljoin, urlparse


@dataclass
class PageMetrics:
    url: str
    load_time: float
    status_code: int
    content_size: int
    response_headers: Dict[str, str]
    error_message: str = ""


@dataclass
class ResourceMetric:
    url: str
    resource_type: str
    size: int
    load_time: float
    cached: bool = False


@dataclass
class PerformanceBenchmark:
    metric_name: str
    value: float
    threshold: float
    unit: str
    passed: bool
    
    @property
    def status(self) -> str:
        return "✅ PASS" if self.passed else "❌ FAIL"


class PerformanceTester:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verbose = False
        
        # Test results
        self.page_metrics: List[PageMetrics] = []
        self.benchmarks: List[PerformanceBenchmark] = []
        
        # Performance thresholds (in seconds)
        self.thresholds = {
            'page_load_time': 2.0,      # 2 seconds
            'first_byte_time': 0.5,     # 500ms
            'content_size_limit': 1024 * 1024,  # 1MB
            'image_size_limit': 500 * 1024,     # 500KB
            'js_size_limit': 200 * 1024,        # 200KB
            'css_size_limit': 100 * 1024,       # 100KB
        }
        
        # Test pages
        self.test_pages = [
            '/',
            '/about',
            '/articles',
            '/glossary',
            '/articles/example',  # Will be replaced with actual articles
            '/glossary/example',  # Will be replaced with actual glossary entries
        ]

    def log(self, message: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level.upper()}] {message}")

    async def measure_page_load(self, session: aiohttp.ClientSession, url: str) -> PageMetrics:
        """Measure page load performance."""
        full_url = urljoin(self.base_url, url)
        start_time = time.time()
        
        try:
            async with session.get(full_url, timeout=self.timeout) as response:
                content = await response.read()
                load_time = time.time() - start_time
                
                return PageMetrics(
                    url=url,
                    load_time=load_time,
                    status_code=response.status,
                    content_size=len(content),
                    response_headers=dict(response.headers)
                )
        
        except asyncio.TimeoutError:
            return PageMetrics(
                url=url,
                load_time=time.time() - start_time,
                status_code=0,
                content_size=0,
                response_headers={},
                error_message="Request timeout"
            )
        
        except Exception as e:
            return PageMetrics(
                url=url,
                load_time=time.time() - start_time,
                status_code=0,
                content_size=0,
                response_headers={},
                error_message=str(e)
            )

    async def check_compression(self, session: aiohttp.ClientSession, url: str) -> Dict[str, bool]:
        """Check if responses are properly compressed."""
        full_url = urljoin(self.base_url, url)
        compression_support = {
            'gzip': False,
            'brotli': False,
            'deflate': False
        }
        
        for encoding in compression_support.keys():
            try:
                headers = {'Accept-Encoding': encoding}
                async with session.get(full_url, headers=headers, timeout=10) as response:
                    content_encoding = response.headers.get('Content-Encoding', '').lower()
                    compression_support[encoding] = encoding in content_encoding
            except:
                pass
        
        return compression_support

    async def check_caching_headers(self, session: aiohttp.ClientSession, url: str) -> Dict[str, str]:
        """Check caching headers for optimization."""
        full_url = urljoin(self.base_url, url)
        caching_headers = {}
        
        try:
            async with session.get(full_url, timeout=10) as response:
                headers_to_check = [
                    'Cache-Control', 'ETag', 'Last-Modified', 
                    'Expires', 'Vary', 'Age'
                ]
                
                for header in headers_to_check:
                    value = response.headers.get(header)
                    if value:
                        caching_headers[header] = value
        
        except Exception as e:
            self.log(f"Error checking caching headers for {url}: {e}", 'warning')
        
        return caching_headers

    async def discover_test_pages(self, session: aiohttp.ClientSession):
        """Discover actual content pages for testing."""
        discovered_pages = []
        
        # Try to get sitemap
        try:
            sitemap_url = urljoin(self.base_url, '/sitemap.xml')
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    # Simple sitemap parsing (could be improved with proper XML parsing)
                    import re
                    urls = re.findall(r'<loc>(.*?)</loc>', content)
                    for url in urls:
                        if self.base_url in url:
                            path = url.replace(self.base_url, '')
                            discovered_pages.append(path)
        except:
            self.log("Could not fetch sitemap, using default test pages", 'info')
        
        # Use discovered pages or fallback to defaults
        if discovered_pages:
            # Limit to reasonable number for testing
            self.test_pages = discovered_pages[:20]
            self.log(f"Discovered {len(self.test_pages)} pages to test", 'info')
        else:
            self.log("Using default test pages", 'info')

    async def run_core_web_vitals_simulation(self, session: aiohttp.ClientSession):
        """Simulate Core Web Vitals measurements."""
        self.log("Running Core Web Vitals simulation...", 'info')
        
        # Test key pages for performance
        key_pages = ['/', '/articles', '/glossary']
        
        for page in key_pages:
            metrics = await self.measure_page_load(session, page)
            self.page_metrics.append(metrics)
            
            if metrics.error_message:
                self.log(f"❌ {page}: {metrics.error_message}", 'error')
                continue
            
            self.log(f"📄 {page}: {metrics.load_time:.2f}s ({metrics.content_size} bytes)", 'info')
            
            # Evaluate against thresholds
            load_time_ok = metrics.load_time <= self.thresholds['page_load_time']
            content_size_ok = metrics.content_size <= self.thresholds['content_size_limit']
            
            self.benchmarks.append(PerformanceBenchmark(
                metric_name=f"Page Load Time ({page})",
                value=metrics.load_time,
                threshold=self.thresholds['page_load_time'],
                unit="seconds",
                passed=load_time_ok
            ))
            
            self.benchmarks.append(PerformanceBenchmark(
                metric_name=f"Content Size ({page})",
                value=metrics.content_size / 1024,  # Convert to KB
                threshold=self.thresholds['content_size_limit'] / 1024,
                unit="KB",
                passed=content_size_ok
            ))

    async def test_static_assets(self, session: aiohttp.ClientSession):
        """Test static asset optimization."""
        self.log("Testing static asset optimization...", 'info')
        
        # Common static asset paths to test
        asset_paths = [
            '/favicon.svg',
            '/assets/main.css',  # Common CSS path
            '/assets/main.js',   # Common JS path
        ]
        
        for asset_path in asset_paths:
            try:
                full_url = urljoin(self.base_url, asset_path)
                start_time = time.time()
                
                async with session.get(full_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.read()
                        load_time = time.time() - start_time
                        
                        # Determine asset type
                        asset_type = 'unknown'
                        if asset_path.endswith('.css'):
                            asset_type = 'css'
                            threshold = self.thresholds['css_size_limit']
                        elif asset_path.endswith('.js'):
                            asset_type = 'javascript'
                            threshold = self.thresholds['js_size_limit']
                        elif asset_path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                            asset_type = 'image'
                            threshold = self.thresholds['image_size_limit']
                        else:
                            threshold = self.thresholds['content_size_limit']
                        
                        size_ok = len(content) <= threshold
                        speed_ok = load_time <= 1.0  # 1 second for static assets
                        
                        self.benchmarks.append(PerformanceBenchmark(
                            metric_name=f"Asset Size ({asset_path})",
                            value=len(content) / 1024,
                            threshold=threshold / 1024,
                            unit="KB",
                            passed=size_ok
                        ))
                        
                        self.benchmarks.append(PerformanceBenchmark(
                            metric_name=f"Asset Load Time ({asset_path})",
                            value=load_time,
                            threshold=1.0,
                            unit="seconds",
                            passed=speed_ok
                        ))
                        
                        self.log(f"📦 {asset_path}: {len(content)/1024:.1f}KB, {load_time:.2f}s", 'info')
                    
                    else:
                        self.log(f"⚠️ {asset_path}: HTTP {response.status}", 'warning')
            
            except Exception as e:
                self.log(f"❌ Error testing {asset_path}: {e}", 'warning')

    async def test_response_headers(self, session: aiohttp.ClientSession):
        """Test HTTP response headers for optimization."""
        self.log("Testing response headers optimization...", 'info')
        
        test_url = '/'
        
        # Check compression
        compression = await self.check_compression(session, test_url)
        gzip_enabled = compression.get('gzip', False)
        
        self.benchmarks.append(PerformanceBenchmark(
            metric_name="GZIP Compression",
            value=1.0 if gzip_enabled else 0.0,
            threshold=1.0,
            unit="enabled",
            passed=gzip_enabled
        ))
        
        # Check caching headers
        caching = await self.check_caching_headers(session, test_url)
        has_cache_control = 'Cache-Control' in caching
        
        self.benchmarks.append(PerformanceBenchmark(
            metric_name="Cache-Control Header",
            value=1.0 if has_cache_control else 0.0,
            threshold=1.0,
            unit="present",
            passed=has_cache_control
        ))
        
        # Check security headers
        full_url = urljoin(self.base_url, test_url)
        try:
            async with session.get(full_url, timeout=10) as response:
                security_headers = [
                    'X-Content-Type-Options',
                    'X-Frame-Options', 
                    'X-XSS-Protection',
                    'Strict-Transport-Security'
                ]
                
                present_headers = sum(1 for header in security_headers 
                                    if header in response.headers)
                security_score = present_headers / len(security_headers)
                
                self.benchmarks.append(PerformanceBenchmark(
                    metric_name="Security Headers",
                    value=security_score * 100,
                    threshold=75.0,  # At least 75% of security headers should be present
                    unit="percent",
                    passed=security_score >= 0.75
                ))
        
        except Exception as e:
            self.log(f"Error checking security headers: {e}", 'warning')

    async def test_accessibility_basics(self, session: aiohttp.ClientSession):
        """Test basic accessibility features."""
        self.log("Testing basic accessibility features...", 'info')
        
        test_pages = ['/', '/articles', '/glossary']
        
        for page in test_pages:
            try:
                full_url = urljoin(self.base_url, page)
                async with session.get(full_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Basic accessibility checks
                        has_lang_attr = 'lang=' in content
                        has_title = '<title>' in content
                        has_meta_viewport = 'name="viewport"' in content
                        has_skip_links = 'skip' in content.lower()
                        
                        # Alt text check (basic)
                        import re
                        img_tags = re.findall(r'<img[^>]*>', content, re.IGNORECASE)
                        imgs_with_alt = sum(1 for img in img_tags if 'alt=' in img)
                        alt_text_score = (imgs_with_alt / len(img_tags)) if img_tags else 1.0
                        
                        self.benchmarks.append(PerformanceBenchmark(
                            metric_name=f"HTML Lang Attribute ({page})",
                            value=1.0 if has_lang_attr else 0.0,
                            threshold=1.0,
                            unit="present",
                            passed=has_lang_attr
                        ))
                        
                        self.benchmarks.append(PerformanceBenchmark(
                            metric_name=f"Page Title ({page})",
                            value=1.0 if has_title else 0.0,
                            threshold=1.0,
                            unit="present",
                            passed=has_title
                        ))
                        
                        self.benchmarks.append(PerformanceBenchmark(
                            metric_name=f"Alt Text Coverage ({page})",
                            value=alt_text_score * 100,
                            threshold=90.0,
                            unit="percent",
                            passed=alt_text_score >= 0.9
                        ))
            
            except Exception as e:
                self.log(f"Error testing accessibility for {page}: {e}", 'warning')

    async def run_load_test(self, concurrent_users: int = 5, duration: int = 30):
        """Run basic load test simulation."""
        self.log(f"Running load test ({concurrent_users} users, {duration}s)...", 'info')
        
        start_time = time.time()
        request_times = []
        error_count = 0
        
        async def user_session():
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while time.time() - start_time < duration:
                    try:
                        # Test random pages
                        import random
                        page = random.choice(self.test_pages)
                        
                        request_start = time.time()
                        async with session.get(urljoin(self.base_url, page)) as response:
                            await response.read()
                            request_time = time.time() - request_start
                            request_times.append(request_time)
                            
                            if response.status >= 400:
                                nonlocal error_count
                                error_count += 1
                    
                    except Exception:
                        error_count += 1
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
        
        # Run concurrent user sessions
        await asyncio.gather(*[user_session() for _ in range(concurrent_users)])
        
        # Calculate metrics
        if request_times:
            avg_response_time = statistics.mean(request_times)
            p95_response_time = statistics.quantiles(request_times, n=20)[18]  # 95th percentile
            total_requests = len(request_times)
            requests_per_second = total_requests / duration
            error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
            
            self.benchmarks.extend([
                PerformanceBenchmark(
                    metric_name="Load Test - Average Response Time",
                    value=avg_response_time,
                    threshold=2.0,
                    unit="seconds",
                    passed=avg_response_time <= 2.0
                ),
                PerformanceBenchmark(
                    metric_name="Load Test - 95th Percentile Response Time",
                    value=p95_response_time,
                    threshold=5.0,
                    unit="seconds",
                    passed=p95_response_time <= 5.0
                ),
                PerformanceBenchmark(
                    metric_name="Load Test - Requests per Second",
                    value=requests_per_second,
                    threshold=10.0,
                    unit="req/s",
                    passed=requests_per_second >= 10.0
                ),
                PerformanceBenchmark(
                    metric_name="Load Test - Error Rate",
                    value=error_rate,
                    threshold=5.0,
                    unit="percent",
                    passed=error_rate <= 5.0
                )
            ])
            
            self.log(f"Load test completed: {total_requests} requests, "
                    f"{avg_response_time:.2f}s avg, {error_rate:.1f}% errors", 'info')

    async def run_all_tests(self, include_load_test: bool = True):
        """Run all performance tests."""
        self.log("Starting comprehensive performance testing...", 'info')
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Discover pages to test
            await self.discover_test_pages(session)
            
            # Run test suites
            await self.run_core_web_vitals_simulation(session)
            await self.test_static_assets(session)
            await self.test_response_headers(session)
            await self.test_accessibility_basics(session)
        
        # Run load test separately (creates its own session)
        if include_load_test:
            await self.run_load_test()

    def print_summary(self):
        """Print performance test summary."""
        print("\n" + "="*70)
        print("PERFORMANCE TEST SUMMARY")
        print("="*70)
        
        # Overall statistics
        total_benchmarks = len(self.benchmarks)
        passed_benchmarks = sum(1 for b in self.benchmarks if b.passed)
        success_rate = (passed_benchmarks / total_benchmarks) * 100 if total_benchmarks > 0 else 0
        
        print(f"Total benchmarks: {total_benchmarks}")
        print(f"Passed benchmarks: {passed_benchmarks}")
        print(f"Success rate: {success_rate:.1f}%")
        print()
        
        # Group benchmarks by category
        categories = {
            'Page Performance': [b for b in self.benchmarks if 'Page Load Time' in b.metric_name or 'Content Size' in b.metric_name],
            'Asset Optimization': [b for b in self.benchmarks if 'Asset' in b.metric_name],
            'HTTP Headers': [b for b in self.benchmarks if any(h in b.metric_name for h in ['Compression', 'Cache', 'Security'])],
            'Accessibility': [b for b in self.benchmarks if any(a in b.metric_name for a in ['Lang', 'Title', 'Alt'])],
            'Load Testing': [b for b in self.benchmarks if 'Load Test' in b.metric_name],
        }
        
        for category, benchmarks in categories.items():
            if benchmarks:
                print(f"{category}:")
                print("-" * len(category))
                
                for benchmark in benchmarks:
                    value_str = f"{benchmark.value:.2f} {benchmark.unit}"
                    threshold_str = f"<= {benchmark.threshold:.2f} {benchmark.unit}"
                    print(f"  {benchmark.status} {benchmark.metric_name}")
                    print(f"      Value: {value_str}, Threshold: {threshold_str}")
                print()
        
        # Show failed benchmarks
        failed_benchmarks = [b for b in self.benchmarks if not b.passed]
        if failed_benchmarks:
            print("FAILED BENCHMARKS:")
            print("-" * 20)
            for benchmark in failed_benchmarks:
                print(f"  ❌ {benchmark.metric_name}: {benchmark.value:.2f} {benchmark.unit} "
                      f"(threshold: {benchmark.threshold:.2f} {benchmark.unit})")
        
        # Page load summary
        if self.page_metrics:
            print("\nPAGE LOAD SUMMARY:")
            print("-" * 20)
            for metric in self.page_metrics:
                if metric.error_message:
                    print(f"  ❌ {metric.url}: {metric.error_message}")
                else:
                    size_mb = metric.content_size / (1024 * 1024)
                    print(f"  📄 {metric.url}: {metric.load_time:.2f}s, {size_mb:.2f}MB")

    def save_report(self, output_file: Path):
        """Save detailed performance report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'summary': {
                'total_benchmarks': len(self.benchmarks),
                'passed_benchmarks': sum(1 for b in self.benchmarks if b.passed),
                'success_rate': (sum(1 for b in self.benchmarks if b.passed) / len(self.benchmarks)) * 100 if self.benchmarks else 0
            },
            'benchmarks': [
                {
                    'metric_name': b.metric_name,
                    'value': b.value,
                    'threshold': b.threshold,
                    'unit': b.unit,
                    'passed': b.passed
                }
                for b in self.benchmarks
            ],
            'page_metrics': [
                {
                    'url': m.url,
                    'load_time': m.load_time,
                    'status_code': m.status_code,
                    'content_size': m.content_size,
                    'error_message': m.error_message
                }
                for m in self.page_metrics
            ],
            'thresholds': self.thresholds
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Performance report saved to {output_file}", 'info')


async def main():
    parser = argparse.ArgumentParser(description='Performance testing for AI Knowledge Website')
    parser.add_argument('--target-url',
                       type=str,
                       default='http://localhost:4321',
                       help='Target URL to test')
    parser.add_argument('--timeout',
                       type=int,
                       default=30,
                       help='Request timeout in seconds')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--output-report',
                       type=str,
                       help='Save detailed report to JSON file')
    parser.add_argument('--skip-load-test',
                       action='store_true',
                       help='Skip the load testing phase')
    parser.add_argument('--fail-threshold',
                       type=float,
                       default=80.0,
                       help='Minimum success rate to pass (percentage)')
    
    args = parser.parse_args()
    
    # Create performance tester
    tester = PerformanceTester(
        base_url=args.target_url,
        timeout=args.timeout
    )
    tester.verbose = args.verbose
    
    # Run all tests
    try:
        await tester.run_all_tests(include_load_test=not args.skip_load_test)
    except Exception as e:
        print(f"Error running performance tests: {e}")
        sys.exit(1)
    
    # Print summary
    tester.print_summary()
    
    # Save report if requested
    if args.output_report:
        tester.save_report(Path(args.output_report))
    
    # Determine exit code
    if tester.benchmarks:
        success_rate = (sum(1 for b in tester.benchmarks if b.passed) / len(tester.benchmarks)) * 100
        
        if success_rate >= args.fail_threshold:
            print(f"\n✅ Performance tests PASSED ({success_rate:.1f}% success rate)")
            sys.exit(0)
        else:
            print(f"\n❌ Performance tests FAILED ({success_rate:.1f}% success rate, threshold: {args.fail_threshold}%)")
            sys.exit(1)
    else:
        print("\n⚠️ No performance benchmarks were run")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())