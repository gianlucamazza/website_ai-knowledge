"""
Pipeline monitoring and metrics collection.

Provides performance monitoring, health checks, and metrics
collection for the content pipeline.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from .config import config
from .database import get_db_session
from .database.models import Article, PipelineRun, Source, ContentStatus, PipelineStage

logger = logging.getLogger(__name__)


class PipelineMetrics:
    """Prometheus metrics for pipeline monitoring."""

    def __init__(self):
        # Counters
        self.articles_processed = Counter(
            "pipeline_articles_processed_total",
            "Total number of articles processed",
            ["stage", "status", "source"],
        )

        self.errors_total = Counter(
            "pipeline_errors_total", "Total number of pipeline errors", ["stage", "error_type"]
        )

        self.pipeline_runs = Counter(
            "pipeline_runs_total", "Total number of pipeline runs", ["status", "trigger"]
        )

        # Histograms for timing
        self.stage_duration = Histogram(
            "pipeline_stage_duration_seconds",
            "Time spent in each pipeline stage",
            ["stage", "status"],
        )

        self.article_processing_time = Histogram(
            "pipeline_article_processing_seconds", "Time to process individual articles", ["stage"]
        )

        self.pipeline_run_duration = Histogram(
            "pipeline_run_duration_seconds", "Total pipeline run duration", ["status"]
        )

        # Gauges for current state
        self.active_runs = Gauge("pipeline_active_runs", "Number of currently active pipeline runs")

        self.articles_by_stage = Gauge(
            "pipeline_articles_by_stage", "Number of articles in each stage", ["stage", "status"]
        )

        self.queue_size = Gauge(
            "pipeline_queue_size", "Number of articles waiting to be processed", ["stage"]
        )

        self.duplicate_rate = Gauge(
            "pipeline_duplicate_rate", "Rate of duplicate articles detected"
        )

        self.quality_score_avg = Gauge(
            "pipeline_quality_score_average", "Average quality score of processed articles"
        )


class PerformanceMonitor:
    """Performance monitoring for pipeline operations."""

    def __init__(self):
        self.metrics = PipelineMetrics()
        self.timing_data = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.last_health_check = None

    @asynccontextmanager
    async def time_operation(self, operation: str, stage: Optional[str] = None, **labels):
        """Context manager for timing operations."""
        start_time = time.time()

        try:
            yield
            duration = time.time() - start_time

            # Record metrics
            if stage:
                self.metrics.stage_duration.labels(stage=stage, status="success").observe(duration)

            # Store timing data for analysis
            self.timing_data[operation].append(
                {
                    "duration": duration,
                    "timestamp": datetime.utcnow(),
                    "stage": stage,
                    "labels": labels,
                }
            )

            # Keep only recent timing data
            if len(self.timing_data[operation]) > 1000:
                self.timing_data[operation].popleft()

        except Exception as e:
            duration = time.time() - start_time

            if stage:
                self.metrics.stage_duration.labels(stage=stage, status="error").observe(duration)
                self.metrics.errors_total.labels(stage=stage, error_type=type(e).__name__).inc()

            self.error_counts[f"{stage or 'unknown'}_{type(e).__name__}"] += 1
            raise

    def record_article_processed(self, stage: str, status: str, source: str = "unknown"):
        """Record article processing metric."""
        self.metrics.articles_processed.labels(stage=stage, status=status, source=source).inc()

    def record_pipeline_run(
        self, status: str, trigger: str = "manual", duration: Optional[float] = None
    ):
        """Record pipeline run metric."""
        self.metrics.pipeline_runs.labels(status=status, trigger=trigger).inc()

        if duration:
            self.metrics.pipeline_run_duration.labels(status=status).observe(duration)

    async def update_gauge_metrics(self):
        """Update gauge metrics from database."""
        try:
            async with get_db_session() as session:
                # Count active runs
                active_runs = await session.execute(
                    select(func.count(PipelineRun.id)).where(
                        PipelineRun.status == ContentStatus.PROCESSING
                    )
                )
                self.metrics.active_runs.set(active_runs.scalar())

                # Articles by stage
                for stage in PipelineStage:
                    for status in ContentStatus:
                        count = await session.execute(
                            select(func.count(Article.id)).where(
                                Article.current_stage == stage, Article.status == status
                            )
                        )
                        self.metrics.articles_by_stage.labels(
                            stage=stage.value, status=status.value
                        ).set(count.scalar())

                # Average quality score
                avg_quality = await session.execute(
                    select(func.avg(Article.quality_score)).where(
                        Article.quality_score.is_not(None)
                    )
                )
                quality_score = avg_quality.scalar()
                if quality_score:
                    self.metrics.quality_score_avg.set(quality_score)

        except Exception as e:
            logger.error(f"Failed to update gauge metrics: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last hour."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        summary = {
            "operations": {},
            "error_counts": dict(self.error_counts),
            "last_updated": datetime.utcnow().isoformat(),
        }

        for operation, timings in self.timing_data.items():
            recent_timings = [t for t in timings if t["timestamp"] > cutoff_time]

            if recent_timings:
                durations = [t["duration"] for t in recent_timings]
                summary["operations"][operation] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations),
                }

        return summary


class HealthChecker:
    """Health checking for pipeline components."""

    def __init__(self):
        self.component_status = {}
        self.last_check = None

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "overall": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "issues": [],
        }

        # Check database connectivity
        db_status = await self._check_database_health()
        health_status["components"]["database"] = db_status

        # Check content sources
        sources_status = await self._check_sources_health()
        health_status["components"]["sources"] = sources_status

        # Check disk space
        disk_status = self._check_disk_space()
        health_status["components"]["disk"] = disk_status

        # Check recent pipeline performance
        pipeline_status = await self._check_pipeline_health()
        health_status["components"]["pipeline"] = pipeline_status

        # Determine overall health
        component_statuses = [status["status"] for status in health_status["components"].values()]

        if "critical" in component_statuses:
            health_status["overall"] = "critical"
        elif "warning" in component_statuses:
            health_status["overall"] = "warning"

        # Collect issues
        for component, status in health_status["components"].items():
            if status["status"] != "healthy" and status.get("message"):
                health_status["issues"].append(f"{component}: {status['message']}")

        self.last_check = datetime.utcnow()
        return health_status

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()

            async with get_db_session() as session:
                # Simple connectivity test
                await session.execute(select(1))

                # Check for recent activity
                recent_articles = await session.execute(
                    select(func.count(Article.id)).where(
                        Article.created_at > datetime.utcnow() - timedelta(hours=24)
                    )
                )

                response_time = time.time() - start_time

                return {
                    "status": "healthy",
                    "response_time_ms": response_time * 1000,
                    "recent_articles_24h": recent_articles.scalar(),
                    "message": "Database connectivity OK",
                }

        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "message": "Database connectivity failed",
            }

    async def _check_sources_health(self) -> Dict[str, Any]:
        """Check content source availability."""
        try:
            async with get_db_session() as session:
                # Get active sources
                active_sources = await session.execute(
                    select(Source).where(Source.is_active == True)
                )
                sources = active_sources.scalars().all()

                # Check for stale sources (not crawled recently)
                stale_threshold = datetime.utcnow() - timedelta(hours=24)
                stale_sources = [
                    s for s in sources if not s.last_crawl or s.last_crawl < stale_threshold
                ]

                status = "healthy"
                message = f"{len(sources)} sources configured"

                if stale_sources:
                    if len(stale_sources) > len(sources) // 2:
                        status = "warning"
                    message += f", {len(stale_sources)} stale"

                return {
                    "status": status,
                    "total_sources": len(sources),
                    "stale_sources": len(stale_sources),
                    "message": message,
                }

        except Exception as e:
            return {"status": "warning", "error": str(e), "message": "Could not check sources"}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil

            # Check data directory space
            data_path = config.get_data_path()
            total, used, free = shutil.disk_usage(data_path)

            free_gb = free // (1024**3)
            total_gb = total // (1024**3)
            usage_percent = (used / total) * 100

            status = "healthy"
            if usage_percent > 90:
                status = "critical"
            elif usage_percent > 80:
                status = "warning"

            return {
                "status": status,
                "free_gb": free_gb,
                "total_gb": total_gb,
                "usage_percent": round(usage_percent, 1),
                "message": f"{free_gb}GB free ({usage_percent:.1f}% used)",
            }

        except Exception as e:
            return {"status": "warning", "error": str(e), "message": "Could not check disk space"}

    async def _check_pipeline_health(self) -> Dict[str, Any]:
        """Check recent pipeline performance."""
        try:
            async with get_db_session() as session:
                # Check recent pipeline runs
                recent_runs = await session.execute(
                    select(PipelineRun)
                    .where(PipelineRun.started_at > datetime.utcnow() - timedelta(hours=24))
                    .order_by(PipelineRun.started_at.desc())
                    .limit(10)
                )
                runs = recent_runs.scalars().all()

                if not runs:
                    return {
                        "status": "warning",
                        "message": "No recent pipeline runs",
                        "recent_runs": 0,
                    }

                # Analyze run results
                failed_runs = [r for r in runs if r.status == ContentStatus.FAILED]
                success_rate = (len(runs) - len(failed_runs)) / len(runs) * 100

                status = "healthy"
                if success_rate < 50:
                    status = "critical"
                elif success_rate < 80:
                    status = "warning"

                avg_duration = None
                completed_runs = [r for r in runs if r.duration_seconds]
                if completed_runs:
                    avg_duration = sum(r.duration_seconds for r in completed_runs) / len(
                        completed_runs
                    )

                return {
                    "status": status,
                    "recent_runs": len(runs),
                    "success_rate": round(success_rate, 1),
                    "failed_runs": len(failed_runs),
                    "avg_duration_seconds": round(avg_duration, 1) if avg_duration else None,
                    "message": f"Success rate: {success_rate:.1f}%",
                }

        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "message": "Could not check pipeline health",
            }


class AlertManager:
    """Alert management for pipeline issues."""

    def __init__(self):
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "disk_usage": 0.9,  # 90% disk usage
            "pipeline_failure_rate": 0.5,  # 50% pipeline failure rate
        }

    async def check_alerts(
        self, health_status: Dict[str, Any], performance_summary: Dict[str, Any]
    ):
        """Check for alert conditions."""
        alerts = []

        # Check health status
        if health_status["overall"] == "critical":
            alerts.append(
                {
                    "severity": "critical",
                    "title": "Pipeline Health Critical",
                    "message": f"Critical issues detected: {', '.join(health_status['issues'])}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Check error rates
        total_ops = sum(op["count"] for op in performance_summary["operations"].values())
        error_count = sum(count for error, count in performance_summary["error_counts"].items())

        if total_ops > 0:
            error_rate = error_count / total_ops
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append(
                    {
                        "severity": "warning",
                        "title": "High Error Rate",
                        "message": f"Error rate: {error_rate:.2%} (threshold: {self.alert_thresholds['error_rate']:.2%})",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"Alert: {alert['title']} - {alert['message']}")

        return alerts

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        return [
            alert for alert in self.alerts if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]


# Global monitoring instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()
alert_manager = AlertManager()


async def start_monitoring():
    """Start background monitoring tasks."""

    async def monitoring_loop():
        while True:
            try:
                # Update metrics
                await performance_monitor.update_gauge_metrics()

                # Check health
                health_status = await health_checker.check_health()

                # Get performance summary
                performance_summary = performance_monitor.get_performance_summary()

                # Check for alerts
                await alert_manager.check_alerts(health_status, performance_summary)

                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    # Start monitoring task
    task = asyncio.create_task(monitoring_loop())
    logger.info("Started pipeline monitoring")
    return task


def get_metrics_endpoint() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest().decode("utf-8")
