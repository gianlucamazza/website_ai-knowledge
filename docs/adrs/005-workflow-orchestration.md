# ADR-005: Workflow Orchestration

## Status
Accepted

## Context

The content pipeline requires sophisticated workflow orchestration to manage the complex multi-stage processing with the following requirements:

- **State Management**: Track processing state across pipeline stages
- **Error Handling**: Robust retry logic and failure recovery
- **Conditional Logic**: Branch workflows based on content type and quality
- **Human-in-the-Loop**: Support manual review and approval steps
- **Observability**: Complete visibility into workflow execution
- **Scalability**: Handle hundreds of concurrent workflows

Current challenges:
- Complex dependencies between pipeline stages
- Need for conditional branching based on content analysis
- Requirement for human review of ambiguous duplicates
- Integration with external APIs and services
- State persistence across system restarts

## Decision

We will use **LangGraph** as our primary workflow orchestration engine with the following architecture:

### Core Components

1. **StateGraph Definition**: Declarative workflow definitions
2. **Persistent State**: PostgreSQL backend for state storage
3. **Event-Driven Execution**: React to external events and triggers
4. **Human-in-the-Loop**: Interactive approval and review steps
5. **Monitoring Integration**: OpenTelemetry tracing and metrics

### Workflow Architecture

```python
from langgraph import StateGraph, END, START
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, List, Optional

class ContentPipelineState(TypedDict):
    # Input data
    source_config: SourceConfig
    raw_content: List[RawContentItem]
    
    # Processing state
    current_stage: str
    stage_results: Dict[str, Any]
    errors: List[ProcessingError]
    
    # Quality metadata
    quality_score: Optional[float]
    duplicate_matches: List[DuplicateMatch]
    review_required: bool
    
    # Output
    published_content: List[PublishedContent]
    workflow_status: str

# Define the workflow graph
def create_content_workflow() -> StateGraph:
    workflow = StateGraph(ContentPipelineState)
    
    # Add nodes
    workflow.add_node("ingest", ingest_content)
    workflow.add_node("normalize", normalize_content)
    workflow.add_node("dedup_check", check_duplicates)
    workflow.add_node("quality_check", assess_quality)
    workflow.add_node("human_review", human_review_step)
    workflow.add_node("enrich", enrich_content)
    workflow.add_node("publish", publish_content)
    workflow.add_node("handle_error", error_handler)
    
    # Define the flow
    workflow.set_entry_point("ingest")
    
    # Linear flow with conditional branching
    workflow.add_edge("ingest", "normalize")
    workflow.add_edge("normalize", "dedup_check")
    
    # Conditional: duplicate handling
    workflow.add_conditional_edges(
        "dedup_check",
        duplicate_router,
        {
            "continue": "quality_check",
            "human_review": "human_review",
            "merge_duplicate": "enrich"
        }
    )
    
    workflow.add_edge("human_review", "quality_check")
    
    # Conditional: quality assessment
    workflow.add_conditional_edges(
        "quality_check",
        quality_router,
        {
            "publish": "enrich",
            "needs_review": "human_review",
            "reject": END
        }
    )
    
    workflow.add_edge("enrich", "publish")
    workflow.add_edge("publish", END)
    
    # Error handling
    workflow.add_edge("handle_error", END)
    
    return workflow

# Conditional routing functions
def duplicate_router(state: ContentPipelineState) -> str:
    if state["duplicate_matches"]:
        highest_similarity = max(
            match.similarity_score for match in state["duplicate_matches"]
        )
        if highest_similarity > 0.9:
            return "merge_duplicate"
        elif highest_similarity > 0.7:
            return "human_review"
    return "continue"

def quality_router(state: ContentPipelineState) -> str:
    quality_score = state.get("quality_score", 0)
    if quality_score > 0.8:
        return "publish"
    elif quality_score > 0.6:
        return "needs_review"
    else:
        return "reject"
```

### Node Implementation Pattern

```python
async def ingest_content(state: ContentPipelineState) -> ContentPipelineState:
    """Ingest content from external sources"""
    try:
        source_config = state["source_config"]
        
        # Create scraper based on source type
        scraper = ScraperFactory.create(source_config.type)
        
        # Fetch content with rate limiting
        raw_content = await scraper.fetch_content(
            source_config.url,
            rate_limit=source_config.rate_limit
        )
        
        # Update state
        state["raw_content"] = raw_content
        state["current_stage"] = "ingest"
        state["stage_results"]["ingest"] = {
            "items_fetched": len(raw_content),
            "timestamp": datetime.utcnow(),
            "success": True
        }
        
        return state
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        state["errors"].append(ProcessingError(
            stage="ingest",
            error=str(e),
            timestamp=datetime.utcnow(),
            retryable=is_retryable_error(e)
        ))
        return state
```

### Human-in-the-Loop Implementation

```python
async def human_review_step(state: ContentPipelineState) -> ContentPipelineState:
    """Pause workflow for human review"""
    # Create review task
    review_task = ReviewTask(
        workflow_id=state.get("workflow_id"),
        content_summary=generate_summary(state["raw_content"]),
        duplicate_matches=state["duplicate_matches"],
        quality_concerns=state.get("quality_concerns", []),
        created_at=datetime.utcnow()
    )
    
    # Store review task
    await review_service.create_task(review_task)
    
    # Send notification to review team
    await notification_service.notify_reviewers(review_task)
    
    # Wait for human input (workflow will be resumed externally)
    state["workflow_status"] = "awaiting_human_review"
    state["review_task_id"] = review_task.id
    
    return state

# External API to resume workflow after human review
@app.post("/api/v1/review/{task_id}/complete")
async def complete_review(
    task_id: str,
    decision: ReviewDecision,
    checkpointer: PostgresSaver = Depends(get_checkpointer)
):
    # Load workflow state
    workflow_id = await review_service.get_workflow_id(task_id)
    state = await checkpointer.get_state(workflow_id)
    
    # Apply human decision
    if decision.action == "approve":
        state["review_required"] = False
        state["human_decision"] = "approved"
    elif decision.action == "merge":
        state["merge_target"] = decision.target_id
        state["human_decision"] = "merge"
    elif decision.action == "reject":
        state["workflow_status"] = "rejected"
        state["human_decision"] = "rejected"
    
    # Resume workflow
    await workflow_executor.resume(workflow_id, state)
    
    return {"status": "resumed"}
```

### Error Handling and Retry Logic

```python
async def error_handler(state: ContentPipelineState) -> ContentPipelineState:
    """Handle workflow errors with retry logic"""
    errors = state.get("errors", [])
    retryable_errors = [e for e in errors if e.retryable]
    
    if retryable_errors:
        # Implement exponential backoff
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIES:
            delay = min(INITIAL_DELAY * (2 ** retry_count), MAX_DELAY)
            
            # Schedule retry
            await schedule_workflow_retry(
                workflow_id=state["workflow_id"],
                delay_seconds=delay
            )
            
            state["retry_count"] = retry_count + 1
            state["workflow_status"] = "scheduled_retry"
            
        else:
            # Max retries exceeded
            state["workflow_status"] = "failed"
            await notification_service.alert_operators(
                f"Workflow {state['workflow_id']} failed after {MAX_RETRIES} retries"
            )
    
    return state

# Retry scheduler integration
@app.post("/api/v1/workflow/{workflow_id}/retry")
async def retry_workflow(
    workflow_id: str,
    checkpointer: PostgresSaver = Depends(get_checkpointer)
):
    # Load previous state
    state = await checkpointer.get_state(workflow_id)
    
    # Reset error state
    state["errors"] = []
    state["workflow_status"] = "retrying"
    
    # Resume from failed stage
    failed_stage = state.get("current_stage")
    await workflow_executor.resume_from_stage(workflow_id, failed_stage, state)
    
    return {"status": "retrying", "stage": failed_stage}
```

## Consequences

### Positive
- **State Persistence**: PostgreSQL checkpoint ensures no work is lost
- **Human Integration**: Native support for approval workflows
- **Observability**: Built-in tracing and monitoring capabilities
- **Flexibility**: Easy to modify workflows without code changes
- **Reliability**: Robust error handling and retry mechanisms
- **Scalability**: Can handle thousands of concurrent workflows

### Negative
- **Learning Curve**: Team needs to learn LangGraph concepts and patterns
- **Database Dependency**: Requires PostgreSQL for state persistence
- **Complexity**: Advanced workflows can become difficult to debug
- **Resource Usage**: State persistence adds database and memory overhead

### Risk Mitigation

1. **Workflow Testing**
   ```python
   # Comprehensive workflow testing
   async def test_content_workflow():
       # Create test state
       test_state = ContentPipelineState(
           source_config=create_test_source(),
           raw_content=[],
           current_stage="start",
           # ... other fields
       )
       
       # Run workflow in test mode
       workflow = create_content_workflow()
       result = await workflow.ainvoke(test_state)
       
       # Assert expected outcomes
       assert result["workflow_status"] == "completed"
       assert len(result["published_content"]) > 0
   ```

2. **Workflow Versioning**
   ```python
   class WorkflowVersion(BaseModel):
       version: str
       graph_definition: Dict
       deployed_at: datetime
       deprecated_at: Optional[datetime]
   
   # Gradual rollout of workflow changes
   async def deploy_workflow_version(version: str, rollout_percentage: int):
       # Deploy new version to percentage of traffic
       pass
   ```

3. **Monitoring and Alerting**
   ```python
   # Workflow health metrics
   WORKFLOW_METRICS = [
       'workflow_completion_rate',
       'average_execution_time',
       'error_rate_by_stage',
       'human_review_queue_depth',
       'retry_rate'
   ]
   
   # Critical alerts
   WORKFLOW_ALERTS = [
       {'metric': 'workflow_completion_rate', 'threshold': 0.95, 'window': '5m'},
       {'metric': 'human_review_queue_depth', 'threshold': 100, 'window': '1m'},
       {'metric': 'error_rate_by_stage', 'threshold': 0.05, 'window': '10m'}
   ]
   ```

## Alternatives Considered

### 1. Apache Airflow
**Pros**: Mature, rich UI, strong community
**Cons**: Heavy infrastructure, complex setup, not designed for real-time
**Rejected**: Too heavyweight for our use case

### 2. Celery + Redis
**Pros**: Simple, well-known, good performance
**Cons**: No native workflow visualization, limited conditional logic
**Rejected**: Insufficient workflow management capabilities

### 3. Temporal.io
**Pros**: Excellent reliability, strong consistency guarantees
**Cons**: Additional infrastructure, learning curve, overkill for content processing
**Rejected**: Too complex for current requirements

### 4. AWS Step Functions
**Pros**: Managed service, good integration with AWS
**Cons**: Vendor lock-in, limited local development, JSON-only configuration
**Rejected**: Want to maintain cloud independence

## Implementation Plan

### Phase 1: Core Workflow (Week 1-2)
- Implement basic linear workflow (ingest → normalize → publish)
- Set up PostgreSQL checkpointing
- Create basic monitoring and logging

### Phase 2: Conditional Logic (Week 3-4)
- Add duplicate detection routing
- Implement quality assessment branching
- Create error handling and retry logic

### Phase 3: Human-in-the-Loop (Week 5-6)
- Implement review task system
- Create review UI for operators
- Add notification system

### Phase 4: Advanced Features (Week 7-8)
- Add workflow versioning and rollout
- Implement comprehensive monitoring
- Create workflow debugging tools

## Monitoring Strategy

```python
# OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

tracer = trace.get_tracer(__name__)

async def traced_workflow_execution(workflow_id: str, state: ContentPipelineState):
    with tracer.start_as_current_span("workflow_execution") as span:
        span.set_attribute("workflow.id", workflow_id)
        span.set_attribute("workflow.stage", state["current_stage"])
        
        try:
            result = await execute_workflow(workflow_id, state)
            span.set_attribute("workflow.status", result["workflow_status"])
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

# Custom metrics
workflow_duration = Histogram(
    "workflow_execution_duration_seconds",
    "Time spent executing workflows",
    ["workflow_type", "stage"]
)

workflow_errors = Counter(
    "workflow_errors_total",
    "Total workflow errors",
    ["workflow_type", "stage", "error_type"]
)
```

## Review Date
This ADR should be reviewed in 4 months (January 2025) after gaining operational experience with LangGraph workflows in production.