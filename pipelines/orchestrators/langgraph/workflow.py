"""
LangGraph workflow definition for the 5-stage content pipeline.

Orchestrates the complete pipeline: ingest → normalize → dedup → enrich → publish
with state persistence, error handling, and conditional routing.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from ...config import config
from ...database.models import PipelineStage, ContentStatus
from .nodes import PipelineNodes

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State model for the pipeline workflow."""
    
    # Pipeline metadata
    run_id: str
    pipeline_name: str
    current_stage: str
    status: str
    
    # Configuration
    source_filters: List[str]
    stage_filters: List[str]
    batch_size: int
    
    # Processing data
    articles_to_process: List[str]  # List of article IDs
    current_article_batch: List[str]
    processed_articles: List[str]
    failed_articles: List[str]
    skipped_articles: List[str]
    
    # Stage results
    ingest_results: Dict
    normalize_results: Dict
    dedup_results: Dict
    enrich_results: Dict
    publish_results: Dict
    
    # Error handling
    errors: List[Dict]
    retry_count: int
    max_retries: int
    
    # Human review
    requires_human_review: bool
    review_items: List[Dict]
    
    # Statistics
    start_time: str
    end_time: Optional[str]
    total_articles: int
    processed_count: int
    success_count: int
    failure_count: int
    
    # Database session info
    session_id: Optional[str]


class PipelineWorkflow:
    """LangGraph workflow for content pipeline orchestration."""
    
    def __init__(self):
        self.nodes = PipelineNodes()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        # Create workflow graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("initialize", self.nodes.initialize_pipeline)
        workflow.add_node("ingest", self.nodes.ingest_content)
        workflow.add_node("normalize", self.nodes.normalize_content)
        workflow.add_node("dedup", self.nodes.deduplicate_content)
        workflow.add_node("enrich", self.nodes.enrich_content)
        workflow.add_node("publish", self.nodes.publish_content)
        workflow.add_node("finalize", self.nodes.finalize_pipeline)
        
        # Human review nodes
        workflow.add_node("human_review_check", self.nodes.check_human_review)
        workflow.add_node("human_review_wait", self.nodes.wait_for_human_review)
        
        # Error handling nodes
        workflow.add_node("handle_error", self.nodes.handle_pipeline_error)
        workflow.add_node("retry_check", self.nodes.check_retry_conditions)
        
        # Define workflow edges
        self._add_workflow_edges(workflow)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        return workflow.compile()
    
    def _add_workflow_edges(self, workflow: StateGraph) -> None:
        """Add edges to define the workflow routing."""
        
        # Main pipeline flow
        workflow.add_edge("initialize", "ingest")
        workflow.add_conditional_edges(
            "ingest",
            self._route_after_ingest,
            {
                "normalize": "normalize",
                "error": "handle_error",
                "complete": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "normalize",
            self._route_after_normalize,
            {
                "dedup": "dedup",
                "error": "handle_error",
                "review": "human_review_check"
            }
        )
        
        workflow.add_conditional_edges(
            "dedup",
            self._route_after_dedup,
            {
                "enrich": "enrich",
                "error": "handle_error",
                "review": "human_review_check"
            }
        )
        
        workflow.add_conditional_edges(
            "enrich",
            self._route_after_enrich,
            {
                "publish": "publish",
                "error": "handle_error",
                "review": "human_review_check"
            }
        )
        
        workflow.add_conditional_edges(
            "publish",
            self._route_after_publish,
            {
                "finalize": "finalize",
                "error": "handle_error"
            }
        )
        
        # Human review flow
        workflow.add_conditional_edges(
            "human_review_check",
            self._route_human_review,
            {
                "wait": "human_review_wait",
                "continue": "enrich",  # Default continue path
                "skip": "publish"
            }
        )
        
        workflow.add_edge("human_review_wait", "enrich")
        
        # Error handling flow
        workflow.add_conditional_edges(
            "handle_error",
            self._route_error_handling,
            {
                "retry": "retry_check",
                "skip": "finalize",
                "abort": END
            }
        )
        
        workflow.add_conditional_edges(
            "retry_check",
            self._route_retry,
            {
                "ingest": "ingest",
                "normalize": "normalize", 
                "dedup": "dedup",
                "enrich": "enrich",
                "publish": "publish",
                "abort": END
            }
        )
        
        workflow.add_edge("finalize", END)
    
    def _route_after_ingest(self, state: PipelineState) -> str:
        """Route after ingest stage."""
        if state["status"] == "error":
            return "error"
        
        if not state.get("articles_to_process"):
            logger.info("No articles to process after ingest")
            return "complete"
        
        # Skip to next enabled stage
        if "normalize" not in state.get("stage_filters", []):
            return self._find_next_stage("normalize", state)
        
        return "normalize"
    
    def _route_after_normalize(self, state: PipelineState) -> str:
        """Route after normalize stage."""
        if state["status"] == "error":
            return "error"
        
        # Check if human review is required
        if self._requires_human_review(state, "normalize"):
            return "review"
        
        # Skip to next enabled stage
        if "dedup" not in state.get("stage_filters", []):
            return self._find_next_stage("dedup", state)
        
        return "dedup"
    
    def _route_after_dedup(self, state: PipelineState) -> str:
        """Route after dedup stage."""
        if state["status"] == "error":
            return "error"
        
        # Check if duplicates found require review
        dedup_results = state.get("dedup_results", {})
        if dedup_results.get("duplicates_found", 0) > 0 and config.enrichment.enable_human_review:
            return "review"
        
        # Skip to next enabled stage
        if "enrich" not in state.get("stage_filters", []):
            return self._find_next_stage("enrich", state)
        
        return "enrich"
    
    def _route_after_enrich(self, state: PipelineState) -> str:
        """Route after enrich stage."""
        if state["status"] == "error":
            return "error"
        
        # Check if enrichment results require review
        if self._requires_human_review(state, "enrich"):
            return "review"
        
        # Skip to next enabled stage
        if "publish" not in state.get("stage_filters", []):
            return self._find_next_stage("publish", state)
        
        return "publish"
    
    def _route_after_publish(self, state: PipelineState) -> str:
        """Route after publish stage."""
        if state["status"] == "error":
            return "error"
        
        return "finalize"
    
    def _route_human_review(self, state: PipelineState) -> str:
        """Route human review decision."""
        if not state.get("requires_human_review", False):
            return "continue"
        
        # In a real implementation, this would check for human input
        # For now, we'll implement automatic approval for certain cases
        review_items = state.get("review_items", [])
        
        # Auto-approve low-risk items
        auto_approve = all(
            item.get("risk_level", "high") == "low" 
            for item in review_items
        )
        
        if auto_approve:
            return "continue"
        
        # High-risk items need human review
        return "wait"
    
    def _route_error_handling(self, state: PipelineState) -> str:
        """Route error handling based on error type and retry count."""
        errors = state.get("errors", [])
        if not errors:
            return "skip"
        
        latest_error = errors[-1]
        error_type = latest_error.get("type", "unknown")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        # Don't retry certain error types
        non_retryable_errors = ["validation_error", "configuration_error"]
        if error_type in non_retryable_errors:
            return "abort"
        
        # Check retry limit
        if retry_count >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded for pipeline")
            return "abort"
        
        # Retry for retryable errors
        return "retry"
    
    def _route_retry(self, state: PipelineState) -> str:
        """Route retry to appropriate stage."""
        current_stage = state.get("current_stage", "ingest")
        
        # Map stages to retry targets
        retry_routes = {
            "ingest": "ingest",
            "normalize": "normalize",
            "dedup": "dedup", 
            "enrich": "enrich",
            "publish": "publish"
        }
        
        return retry_routes.get(current_stage, "abort")
    
    def _find_next_stage(self, current_stage: str, state: PipelineState) -> str:
        """Find the next enabled stage after skipping current stage."""
        stage_order = ["ingest", "normalize", "dedup", "enrich", "publish"]
        stage_filters = state.get("stage_filters", stage_order)
        
        try:
            current_index = stage_order.index(current_stage)
            
            # Find next enabled stage
            for i in range(current_index + 1, len(stage_order)):
                next_stage = stage_order[i]
                if not stage_filters or next_stage in stage_filters:
                    return next_stage
            
            # No more stages, finalize
            return "finalize"
            
        except ValueError:
            return "finalize"
    
    def _requires_human_review(self, state: PipelineState, stage: str) -> bool:
        """Determine if human review is required for a stage."""
        # Check global human review setting
        if not getattr(config.enrichment, 'enable_human_review', False):
            return False
        
        # Stage-specific review logic
        if stage == "normalize":
            normalize_results = state.get("normalize_results", {})
            # Review if quality score is low
            return normalize_results.get("avg_quality_score", 1.0) < 0.6
        
        elif stage == "enrich":
            enrich_results = state.get("enrich_results", {})
            # Review if enrichment confidence is low
            return enrich_results.get("avg_confidence", 1.0) < 0.7
        
        return False
    
    async def run_pipeline(self, config_dict: Dict) -> Dict:
        """
        Run the complete pipeline with the given configuration.
        
        Args:
            config_dict: Pipeline configuration
            
        Returns:
            Dict with pipeline execution results
        """
        try:
            # Initialize pipeline state
            initial_state = self._create_initial_state(config_dict)
            
            # Execute workflow
            final_state = await self.graph.ainvoke(
                initial_state,
                config=RunnableConfig(recursion_limit=50)
            )
            
            # Return results
            return self._format_pipeline_results(final_state)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "processed_articles": 0,
                "success_count": 0,
                "failure_count": 1
            }
    
    def _create_initial_state(self, config_dict: Dict) -> PipelineState:
        """Create initial state for pipeline execution."""
        return PipelineState(
            run_id=config_dict.get("run_id", f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
            pipeline_name=config_dict.get("pipeline_name", "content_pipeline"),
            current_stage="initialize",
            status="pending",
            source_filters=config_dict.get("source_filters", []),
            stage_filters=config_dict.get("stage_filters", []),
            batch_size=config_dict.get("batch_size", 10),
            articles_to_process=[],
            current_article_batch=[],
            processed_articles=[],
            failed_articles=[],
            skipped_articles=[],
            ingest_results={},
            normalize_results={},
            dedup_results={},
            enrich_results={},
            publish_results={},
            errors=[],
            retry_count=0,
            max_retries=config_dict.get("max_retries", 3),
            requires_human_review=False,
            review_items=[],
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            total_articles=0,
            processed_count=0,
            success_count=0,
            failure_count=0,
            session_id=config_dict.get("session_id"),
        )
    
    def _format_pipeline_results(self, state: PipelineState) -> Dict:
        """Format final pipeline results."""
        return {
            "run_id": state["run_id"],
            "status": state["status"],
            "start_time": state["start_time"],
            "end_time": state.get("end_time"),
            "total_articles": state["total_articles"],
            "processed_count": state["processed_count"],
            "success_count": state["success_count"],
            "failure_count": state["failure_count"],
            "stage_results": {
                "ingest": state.get("ingest_results", {}),
                "normalize": state.get("normalize_results", {}),
                "dedup": state.get("dedup_results", {}),
                "enrich": state.get("enrich_results", {}),
                "publish": state.get("publish_results", {}),
            },
            "errors": state.get("errors", []),
            "processed_articles": state.get("processed_articles", []),
            "failed_articles": state.get("failed_articles", []),
        }
    
    async def resume_pipeline(self, run_id: str, checkpoint_data: Dict) -> Dict:
        """Resume a pipeline from a checkpoint."""
        try:
            # Restore state from checkpoint
            state = PipelineState(**checkpoint_data)
            state["status"] = "resuming"
            
            # Resume execution
            final_state = await self.graph.ainvoke(
                state,
                config=RunnableConfig(recursion_limit=50)
            )
            
            return self._format_pipeline_results(final_state)
            
        except Exception as e:
            logger.error(f"Failed to resume pipeline {run_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "run_id": run_id
            }
    
    def get_workflow_visualization(self) -> str:
        """Get a visual representation of the workflow graph."""
        try:
            # This would generate a visual representation
            # For now, return a text description
            return """
            Content Pipeline Workflow:
            
            initialize → ingest → normalize → dedup → enrich → publish → finalize
                  ↓         ↓         ↓         ↓        ↓        ↓
            handle_error ← error_check ← error_check ← error_check ← error_check
                  ↓
            retry_check → [retry to appropriate stage]
            
            human_review_check → human_review_wait → [continue]
                  ↓
            [auto-continue or skip]
            """
        except Exception as e:
            logger.error(f"Error generating workflow visualization: {e}")
            return "Workflow visualization not available"