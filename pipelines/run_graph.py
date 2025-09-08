"""
CLI interface for running the content pipeline using Typer.

Provides commands for running individual stages, full pipeline,
and monitoring pipeline execution.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .config import config
from .database import init_database, close_database, get_db_session
from .orchestrators.langgraph import PipelineWorkflow

# Initialize Typer app
app = typer.Typer(
    name="content-pipeline",
    help="AI Knowledge Content Pipeline - Ingest, process, and publish AI/ML content",
    add_completion=False,
)

# Rich console for formatted output
console = Console()

# Global workflow instance
workflow = PipelineWorkflow()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
):
    """AI Knowledge Content Pipeline CLI."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration if specified
    if config_file and config_file.exists():
        # TODO: Load configuration from file
        rprint(f"[yellow]Loading configuration from: {config_file}[/yellow]")


@app.command()
def run(
    sources: List[str] = typer.Option([], "--source", "-s", help="Filter by source names"),
    stages: List[str] = typer.Option([], "--stage", help="Run specific stages only"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Processing batch size"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum retry attempts"),
    run_name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name for tracking"),
):
    """Run the complete content pipeline."""

    async def _run_pipeline():
        try:
            # Initialize database
            await init_database()

            # Prepare pipeline configuration
            run_config = {
                "run_id": f"cli_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "pipeline_name": run_name or "cli_pipeline",
                "source_filters": sources,
                "stage_filters": stages,
                "batch_size": batch_size,
                "max_retries": max_retries,
            }

            # Display configuration
            config_table = Table(title="Pipeline Configuration")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="magenta")

            config_table.add_row("Run ID", run_config["run_id"])
            config_table.add_row("Sources", ", ".join(sources) if sources else "All")
            config_table.add_row("Stages", ", ".join(stages) if stages else "All")
            config_table.add_row("Batch Size", str(batch_size))
            config_table.add_row("Max Retries", str(max_retries))

            console.print(config_table)
            console.print()

            # Run pipeline with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:

                task = progress.add_task("Running pipeline...", total=100)

                # Start pipeline
                rprint("[green]Starting content pipeline...[/green]")
                results = await workflow.run_pipeline(run_config)

                progress.update(task, completed=100)

            # Display results
            display_pipeline_results(results)

        except Exception as e:
            console.print(f"[red]Pipeline execution failed: {e}[/red]")
            raise typer.Exit(1)

        finally:
            await close_database()

    asyncio.run(_run_pipeline())


@app.command()
def ingest(
    sources: List[str] = typer.Option([], "--source", "-s", help="Filter by source names"),
):
    """Run only the content ingestion stage."""

    async def _run_ingest():
        try:
            await init_database()

            from .ingest import SourceManager

            source_manager = SourceManager()

            async with get_db_session() as session:
                await source_manager.sync_sources_to_database(session)

                if sources:
                    rprint(f"[cyan]Running ingestion for sources: {', '.join(sources)}[/cyan]")
                else:
                    rprint("[cyan]Running scheduled ingestion for all active sources[/cyan]")

                results = await source_manager.run_scheduled_ingestion(session)

                # Display results
                results_table = Table(title="Ingestion Results")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="green")

                results_table.add_row("Sources Processed", str(results["sources_processed"]))
                results_table.add_row(
                    "Articles Discovered", str(results["total_articles_discovered"])
                )
                results_table.add_row("Articles Ingested", str(results["total_articles_ingested"]))
                results_table.add_row("Articles Skipped", str(results["total_articles_skipped"]))
                results_table.add_row("Errors", str(results["total_errors"]))

                console.print(results_table)

                if results["source_results"]:
                    console.print()
                    source_table = Table(title="Per-Source Results")
                    source_table.add_column("Source")
                    source_table.add_column("Discovered")
                    source_table.add_column("Ingested")
                    source_table.add_column("Errors")

                    for source_result in results["source_results"]:
                        source_table.add_row(
                            source_result["source_name"],
                            str(source_result["articles_discovered"]),
                            str(source_result["articles_ingested"]),
                            str(source_result["errors"]),
                        )

                    console.print(source_table)

        except Exception as e:
            console.print(f"[red]Ingestion failed: {e}[/red]")
            raise typer.Exit(1)

        finally:
            await close_database()

    asyncio.run(_run_ingest())


@app.command()
def publish(
    article_ids: List[str] = typer.Option(
        [], "--article", "-a", help="Specific article IDs to publish"
    ),
    all_ready: bool = typer.Option(False, "--all", help="Publish all ready articles"),
):
    """Publish articles as markdown files."""

    async def _run_publish():
        try:
            await init_database()

            from .publish import MarkdownGenerator
            from .database.models import Article, ContentStatus, PipelineStage
            from sqlalchemy import select

            markdown_generator = MarkdownGenerator()

            async with get_db_session() as session:
                articles_to_publish = []

                if article_ids:
                    # Publish specific articles
                    articles_to_publish = article_ids
                    rprint(f"[cyan]Publishing {len(article_ids)} specific articles[/cyan]")

                elif all_ready:
                    # Find all articles ready for publishing
                    query = select(Article.id).where(
                        Article.current_stage == PipelineStage.ENRICH,
                        Article.status == ContentStatus.COMPLETED,
                        Article.cleaned_content.is_not(None),
                    )
                    result = await session.execute(query)
                    articles_to_publish = [str(row[0]) for row in result.fetchall()]
                    rprint(f"[cyan]Publishing {len(articles_to_publish)} ready articles[/cyan]")

                else:
                    rprint("[red]Please specify either --article IDs or --all flag[/red]")
                    raise typer.Exit(1)

                if not articles_to_publish:
                    rprint("[yellow]No articles to publish[/yellow]")
                    return

                # Publish articles
                with Progress(console=console) as progress:
                    task = progress.add_task(
                        "Publishing articles...", total=len(articles_to_publish)
                    )

                    results = await markdown_generator.publish_batch(articles_to_publish, session)
                    progress.update(task, completed=len(articles_to_publish))

                # Update taxonomies
                await markdown_generator.update_taxonomies(session)

                # Display results
                results_table = Table(title="Publishing Results")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="green")

                results_table.add_row("Total Articles", str(results["total_articles"]))
                results_table.add_row("Published", str(results["published"]))
                results_table.add_row("Failed", str(results["failed"]))
                results_table.add_row("Skipped", str(results["skipped"]))

                console.print(results_table)

                if results["errors"]:
                    console.print()
                    console.print("[red]Errors:[/red]")
                    for error in results["errors"]:
                        console.print(f"  - {error}")

        except Exception as e:
            console.print(f"[red]Publishing failed: {e}[/red]")
            raise typer.Exit(1)

        finally:
            await close_database()

    asyncio.run(_run_publish())


@app.command()
def status(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r", help="Show status for specific run"
    ),
    recent: int = typer.Option(10, "--recent", "-n", help="Show N most recent runs"),
):
    """Show pipeline status and recent runs."""

    async def _show_status():
        try:
            await init_database()

            from .database.models import PipelineRun, Article, ContentStatus, PipelineStage
            from sqlalchemy import select, func, desc

            async with get_db_session() as session:
                if run_id:
                    # Show specific run status
                    query = select(PipelineRun).where(PipelineRun.id == run_id)
                    result = await session.execute(query)
                    pipeline_run = result.scalar_one_or_none()

                    if not pipeline_run:
                        console.print(f"[red]Run not found: {run_id}[/red]")
                        return

                    display_run_status(pipeline_run)

                else:
                    # Show recent runs and overall statistics

                    # Recent runs
                    query = select(PipelineRun).order_by(desc(PipelineRun.started_at)).limit(recent)
                    result = await session.execute(query)
                    recent_runs = result.scalars().all()

                    if recent_runs:
                        runs_table = Table(title=f"Recent Pipeline Runs (Last {recent})")
                        runs_table.add_column("ID", style="cyan")
                        runs_table.add_column("Name")
                        runs_table.add_column("Status", style="magenta")
                        runs_table.add_column("Started", style="green")
                        runs_table.add_column("Duration")
                        runs_table.add_column("Articles")

                        for run in recent_runs:
                            duration = (
                                "Running..."
                                if not run.completed_at
                                else f"{run.duration_seconds:.1f}s"
                            )
                            runs_table.add_row(
                                str(run.id)[:8],
                                run.run_name or "Unnamed",
                                run.status.value,
                                run.started_at.strftime("%Y-%m-%d %H:%M"),
                                duration,
                                str(run.articles_processed or 0),
                            )

                        console.print(runs_table)
                        console.print()

                    # Overall statistics
                    stats_query = select(
                        func.count(Article.id).label("total_articles"),
                        func.count()
                        .filter(Article.status == ContentStatus.COMPLETED)
                        .label("completed"),
                        func.count()
                        .filter(Article.status == ContentStatus.PROCESSING)
                        .label("processing"),
                        func.count().filter(Article.status == ContentStatus.FAILED).label("failed"),
                        func.count()
                        .filter(Article.current_stage == PipelineStage.INGEST)
                        .label("ingest"),
                        func.count()
                        .filter(Article.current_stage == PipelineStage.NORMALIZE)
                        .label("normalize"),
                        func.count()
                        .filter(Article.current_stage == PipelineStage.DEDUP)
                        .label("dedup"),
                        func.count()
                        .filter(Article.current_stage == PipelineStage.ENRICH)
                        .label("enrich"),
                        func.count()
                        .filter(Article.current_stage == PipelineStage.PUBLISH)
                        .label("publish"),
                    ).select_from(Article)

                    result = await session.execute(stats_query)
                    stats = result.first()

                    stats_table = Table(title="Content Statistics")
                    stats_table.add_column("Stage", style="cyan")
                    stats_table.add_column("Count", style="green")

                    stats_table.add_row("Total Articles", str(stats.total_articles))
                    stats_table.add_row("Completed", str(stats.completed))
                    stats_table.add_row("Processing", str(stats.processing))
                    stats_table.add_row("Failed", str(stats.failed))
                    stats_table.add_row("", "")  # Separator
                    stats_table.add_row("In Ingest", str(stats.ingest))
                    stats_table.add_row("In Normalize", str(stats.normalize))
                    stats_table.add_row("In Dedup", str(stats.dedup))
                    stats_table.add_row("In Enrich", str(stats.enrich))
                    stats_table.add_row("In Publish", str(stats.publish))

                    console.print(stats_table)

        except Exception as e:
            console.print(f"[red]Failed to get status: {e}[/red]")
            raise typer.Exit(1)

        finally:
            await close_database()

    asyncio.run(_show_status())


@app.command()
def sources(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all configured sources"),
    sync: bool = typer.Option(False, "--sync", help="Sync source configuration to database"),
):
    """Manage content sources."""

    async def _manage_sources():
        try:
            await init_database()

            from .ingest import SourceManager

            source_manager = SourceManager()

            async with get_db_session() as session:
                if sync:
                    rprint("[cyan]Syncing source configuration to database...[/cyan]")
                    await source_manager.sync_sources_to_database(session)
                    rprint("[green]Source configuration synced successfully[/green]")

                if list_all or sync:
                    # Display sources
                    from .database.models import Source
                    from sqlalchemy import select

                    query = select(Source).order_by(Source.name)
                    result = await session.execute(query)
                    sources_list = result.scalars().all()

                    if sources_list:
                        sources_table = Table(title="Configured Sources")
                        sources_table.add_column("Name", style="cyan")
                        sources_table.add_column("Type")
                        sources_table.add_column("URL", style="blue")
                        sources_table.add_column("Active", style="green")
                        sources_table.add_column("Last Crawl")
                        sources_table.add_column("Frequency")

                        for source in sources_list:
                            last_crawl = (
                                source.last_crawl.strftime("%Y-%m-%d %H:%M")
                                if source.last_crawl
                                else "Never"
                            )
                            frequency = (
                                f"{source.crawl_frequency}s" if source.crawl_frequency else "N/A"
                            )

                            sources_table.add_row(
                                source.name,
                                source.source_type,
                                source.base_url,
                                "✓" if source.is_active else "✗",
                                last_crawl,
                                frequency,
                            )

                        console.print(sources_table)
                    else:
                        console.print("[yellow]No sources configured[/yellow]")

        except Exception as e:
            console.print(f"[red]Failed to manage sources: {e}[/red]")
            raise typer.Exit(1)

        finally:
            await close_database()

    asyncio.run(_manage_sources())


@app.command()
def workflow():
    """Show workflow visualization."""
    visualization = workflow.get_workflow_visualization()

    panel = Panel(
        visualization,
        title="Content Pipeline Workflow",
        border_style="cyan",
    )

    console.print(panel)


def display_pipeline_results(results: dict):
    """Display formatted pipeline results."""

    # Main results panel
    status_color = "green" if results["status"] == "completed" else "red"

    panel = Panel(
        f"[{status_color}]Status: {results['status'].upper()}[/{status_color}]\n"
        f"Run ID: {results['run_id']}\n"
        f"Duration: {calculate_duration(results.get('start_time'), results.get('end_time'))}",
        title="Pipeline Results",
        border_style=status_color,
    )

    console.print(panel)
    console.print()

    # Statistics table
    stats_table = Table(title="Processing Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Articles", str(results.get("total_articles", 0)))
    stats_table.add_row("Processed", str(results.get("processed_count", 0)))
    stats_table.add_row("Successful", str(results.get("success_count", 0)))
    stats_table.add_row("Failed", str(results.get("failure_count", 0)))

    console.print(stats_table)
    console.print()

    # Stage results
    stage_results = results.get("stage_results", {})
    if stage_results:
        stage_table = Table(title="Stage Results")
        stage_table.add_column("Stage", style="cyan")
        stage_table.add_column("Status", style="magenta")
        stage_table.add_column("Details")

        for stage, stage_data in stage_results.items():
            if stage_data:
                details = ", ".join(
                    [f"{k}: {v}" for k, v in stage_data.items() if isinstance(v, (int, float, str))]
                )
                stage_table.add_row(
                    stage.title(), "✓", details[:50] + "..." if len(details) > 50 else details
                )

        console.print(stage_table)

    # Errors
    errors = results.get("errors", [])
    if errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for error in errors[-5:]:  # Show last 5 errors
            console.print(
                f"  - [{error.get('stage', 'unknown')}] {error.get('message', 'Unknown error')}"
            )


def display_run_status(pipeline_run):
    """Display detailed status for a pipeline run."""

    status_color = "green" if pipeline_run.status == ContentStatus.COMPLETED else "yellow"

    panel = Panel(
        (
            f"[{status_color}]Status: {pipeline_run.status.value.upper()}[/{status_color}]\n"
            f"Name: {pipeline_run.run_name}\n"
            f"Started: {pipeline_run.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current Stage: {pipeline_run.current_stage.value if pipeline_run.current_stage else 'N/A'}\n"
            f"Duration: {pipeline_run.duration_seconds:.1f}s"
            if pipeline_run.duration_seconds
            else "Running..."
        ),
        title=f"Pipeline Run {str(pipeline_run.id)[:8]}",
        border_style=status_color,
    )

    console.print(panel)


def calculate_duration(start_time: Optional[str], end_time: Optional[str]) -> str:
    """Calculate and format duration."""
    if not start_time:
        return "Unknown"

    if not end_time:
        return "Running..."

    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        duration = (end - start).total_seconds()

        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"

    except Exception:
        return "Unknown"


if __name__ == "__main__":
    app()
