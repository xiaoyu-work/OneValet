"""
OneValet Workflow Models - Data structures for multi-agent workflow orchestration

This module implements the 4-keyword workflow system as defined in DESIGN_V2.md (Section 5.4.6):
- run: Execute agents sequentially
- parallel: Execute agents in parallel
- then: Aggregator agent (receives results from previous step)
- stages: Multi-phase workflow (sequential groups of parallel tasks)

Supports three workflow types:
- scheduled: Triggered by cron expression
- event_triggered: Triggered by system events
- interactive: Triggered by user message

Uses three-layer parameter resolution:
- Template: Default values (shared across all users)
- Profile: User preferences (per-user defaults)
- Instance: User overrides (per-user customization)
Priority: Instance > Profile > Template
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum


class WorkflowType(str, Enum):
    """Type of workflow trigger"""
    SCHEDULED = "scheduled"          # Triggered by cron expression
    EVENT_TRIGGERED = "event_triggered"  # Triggered by system event
    INTERACTIVE = "interactive"      # Triggered by user message


class WorkflowStatus(str, Enum):
    """Current status of a workflow execution"""
    PENDING = "pending"        # Waiting to start
    RUNNING = "running"        # Currently executing
    PAUSED = "paused"          # Paused by user or system
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Failed with error
    CANCELLED = "cancelled"    # Cancelled by user


class StageStatus(str, Enum):
    """Status of a single stage in a workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageDefinition:
    """
    Definition of a single stage in a multi-stage workflow.

    A stage can contain:
    - run: Sequential execution of agents
    - parallel: Parallel execution of agents
    - then: Aggregator agent for the stage

    Example:
        stage = StageDefinition(
            parallel=["SpamDetector", "UrgencyClassifier"],
            then="ClassificationAggregator"
        )
    """
    run: Optional[List[str]] = None
    parallel: Optional[List[str]] = None
    then: Optional[str] = None

    # Optional condition for stage execution (Jinja2 expression)
    condition: Optional[str] = None


@dataclass
class WorkflowTemplate:
    """
    Workflow template definition (Layer 1 - shared across all users).

    This is the main workflow definition loaded from YAML configuration.
    Only uses 4 keywords: run, parallel, then, stages.

    Example (Sequential):
        WorkflowTemplate(
            id="morning_brief",
            type=WorkflowType.INTERACTIVE,
            description="Personalized morning briefing",
            triggers=["morning brief", "good morning"],
            inputs={"location": "{{user.city}}", "topics": "{{user.news_preferences}}"},
            run=["WeatherAgent", "CalendarAgent", "NewsAgent", "DigestAgent"]
        )

    Example (Parallel with aggregator):
        WorkflowTemplate(
            id="travel_research",
            type=WorkflowType.INTERACTIVE,
            triggers=["plan trip to", "travel to"],
            inputs={"destination": None, "dates": None},
            parallel=["FlightSearchAgent", "HotelSearchAgent", "RestaurantAgent"],
            then="TravelPlannerAgent"
        )

    Example (Multi-stage):
        WorkflowTemplate(
            id="email_processing",
            type=WorkflowType.EVENT_TRIGGERED,
            trigger_config={"event_type": "email_received"},
            inputs={"email_id": "{{event.email_id}}"},
            stages=[
                StageDefinition(parallel=["SpamDetector", "UrgencyClassifier"], then="Aggregator"),
                StageDefinition(run=["EmailRouter"]),
                StageDefinition(parallel=["NotificationAgent", "DatabaseAgent"])
            ]
        )
    """
    # Identity
    id: str
    description: str = ""

    # Workflow type
    type: WorkflowType = WorkflowType.INTERACTIVE

    # Triggers (for interactive workflows)
    triggers: List[str] = field(default_factory=list)

    # Trigger configuration (for event-triggered workflows)
    trigger_config: Optional[Dict[str, Any]] = None

    # Schedule (for scheduled workflows)
    schedule: Optional[str] = None  # Cron expression

    # Input schema with defaults (supports {{user.xxx}} variables)
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Execution definition - ONLY 4 KEYWORDS
    # Use exactly one of these:
    run: Optional[List[str]] = None       # Sequential execution
    parallel: Optional[List[str]] = None  # Parallel execution
    stages: Optional[List[StageDefinition]] = None  # Multi-stage execution

    # Aggregator (used with parallel)
    then: Optional[str] = None

    # Execution settings
    timeout_seconds: int = 3600  # 1 hour default
    retry_on_failure: bool = False
    max_retries: int = 3

    # Approval settings
    # auto_approve: List of agent types that should be auto-approved in this workflow
    # If empty, all agents requiring approval will request user confirmation
    auto_approve: List[str] = field(default_factory=list)
    # auto_approve_all: If True, auto-approve all agents in this workflow
    auto_approve_all: bool = False

    # Metadata
    tags: List[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_execution_type(self) -> Literal["run", "parallel", "stages"]:
        """Determine the execution type based on which keyword is set"""
        if self.stages:
            return "stages"
        elif self.parallel:
            return "parallel"
        else:
            return "run"

    def validate(self) -> List[str]:
        """Validate the workflow template, returns list of errors"""
        errors = []

        # Must have exactly one execution type
        execution_count = sum([
            bool(self.run),
            bool(self.parallel),
            bool(self.stages)
        ])

        if execution_count == 0:
            errors.append("Workflow must define at least one of: run, parallel, or stages")
        elif execution_count > 1:
            errors.append("Workflow must define only one of: run, parallel, or stages (not multiple)")

        # 'then' only makes sense with 'parallel' or within stages
        if self.then and not self.parallel and not self.stages:
            errors.append("'then' keyword can only be used with 'parallel' or within 'stages'")

        # Validate by workflow type
        if self.type == WorkflowType.SCHEDULED and not self.schedule:
            errors.append("Scheduled workflow must have a 'schedule' (cron expression)")

        if self.type == WorkflowType.EVENT_TRIGGERED and not self.trigger_config:
            errors.append("Event-triggered workflow must have 'trigger_config'")

        if self.type == WorkflowType.INTERACTIVE and not self.triggers:
            errors.append("Interactive workflow should have 'triggers' defined")

        return errors


@dataclass
class UserProfile:
    """
    User profile data (Layer 2 - per-user defaults).

    Contains user preferences and settings that can be referenced
    in workflow inputs using {{user.xxx}} syntax.

    Example:
        UserProfile(
            id="alice",
            email="alice@example.com",
            data={
                "city": "San Francisco",
                "news_preferences": ["tech", "startups", "ai"],
                "weather_units": "fahrenheit",
                "timezone": "America/Los_Angeles"
            }
        )
    """
    id: str
    email: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from user profile data"""
        return self.data.get(key, default)


@dataclass
class UserWorkflowInstance:
    """
    User-specific workflow instance (Layer 3 - per-user overrides).

    Allows users to customize a workflow template with their own settings.
    These overrides take priority over template defaults and user profile.

    Example:
        # Alice is traveling to London, override her location
        UserWorkflowInstance(
            id="alice_morning_travel",
            user_id="alice",
            template_id="morning_brief",
            schedule="0 8 * * *",
            inputs={
                "location": "London",  # Override (usually from profile)
                "weather_units": "celsius"
            },
            enabled=True
        )
    """
    id: str
    user_id: str
    template_id: str

    # Optional schedule override (for scheduled workflows)
    schedule: Optional[str] = None

    # User-specific input overrides
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Enable/disable this instance
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResolvedInputs:
    """
    Result of three-layer parameter resolution.

    Contains the final resolved values after applying:
    1. Template defaults
    2. User profile values
    3. Instance overrides

    Plus system-provided values like user_id, today, timezone.
    """
    values: Dict[str, Any] = field(default_factory=dict)

    # Track where each value came from (for debugging)
    sources: Dict[str, Literal["template", "profile", "instance", "system"]] = field(
        default_factory=dict
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a resolved value"""
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values


@dataclass
class WorkflowAgentResult:
    """Result from a single agent execution within a workflow"""
    agent_id: str
    agent_type: str
    status: Literal["completed", "failed", "skipped"]

    # Output data
    output: Any = None
    raw_message: Optional[str] = None

    # Error info
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class StageResult:
    """Result from executing a single stage in a workflow"""
    stage_index: int
    status: StageStatus

    # Results from agents in this stage
    agent_results: List[WorkflowAgentResult] = field(default_factory=list)

    # Aggregator result (if 'then' was specified)
    aggregator_result: Optional[WorkflowAgentResult] = None

    # Combined output (from aggregator or last agent)
    output: Any = None

    # Error info
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowResult:
    """Result from executing a complete workflow"""
    workflow_id: str
    execution_id: str
    user_id: str
    status: WorkflowStatus

    # For sequential workflows
    agent_results: List[WorkflowAgentResult] = field(default_factory=list)

    # For multi-stage workflows
    stage_results: List[StageResult] = field(default_factory=list)

    # Final output (from last agent or aggregator)
    final_output: Any = None
    final_message: Optional[str] = None

    # Collected outputs from all agents (keyed by agent_type)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Error info
    error: Optional[str] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total execution duration in seconds"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if workflow completed successfully"""
        return self.status == WorkflowStatus.COMPLETED


@dataclass
class WorkflowExecution:
    """
    Tracks the execution state of a running workflow.

    This is used to track progress and support pause/resume.
    """
    id: str
    workflow_id: str
    user_id: str

    # Current state
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_stage_index: int = 0
    current_agent_index: int = 0

    # Resolved inputs
    resolved_inputs: ResolvedInputs = field(default_factory=ResolvedInputs)

    # Partial results (for pause/resume)
    agent_results: List[WorkflowAgentResult] = field(default_factory=list)
    stage_results: List[StageResult] = field(default_factory=list)

    # Collected outputs (accumulated during execution)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Error info
    error: Optional[str] = None

    # Trigger info
    trigger_type: Literal["schedule", "event", "interactive", "manual"] = "manual"
    trigger_data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def add_agent_result(self, result: WorkflowAgentResult) -> None:
        """Add an agent result and update outputs"""
        self.agent_results.append(result)
        if result.output is not None:
            self.outputs[result.agent_type] = result.output

    def add_stage_result(self, result: StageResult) -> None:
        """Add a stage result"""
        self.stage_results.append(result)

        # Add all agent outputs from the stage
        for agent_result in result.agent_results:
            if agent_result.output is not None:
                self.outputs[agent_result.agent_type] = agent_result.output

        # Add aggregator output if present
        if result.aggregator_result and result.aggregator_result.output is not None:
            self.outputs[result.aggregator_result.agent_type] = result.aggregator_result.output

    def to_result(self) -> WorkflowResult:
        """Convert execution state to final result"""
        return WorkflowResult(
            workflow_id=self.workflow_id,
            execution_id=self.id,
            user_id=self.user_id,
            status=self.status,
            agent_results=self.agent_results,
            stage_results=self.stage_results,
            final_output=self._get_final_output(),
            final_message=self._get_final_message(),
            outputs=self.outputs,
            error=self.error,
            started_at=self.started_at or datetime.now(),
            completed_at=self.completed_at
        )

    def _get_final_output(self) -> Any:
        """Get the final output from the last executed agent/stage"""
        if self.stage_results:
            last_stage = self.stage_results[-1]
            return last_stage.output
        elif self.agent_results:
            return self.agent_results[-1].output
        return None

    def _get_final_message(self) -> Optional[str]:
        """Get the final message from the last executed agent/stage"""
        if self.stage_results:
            last_stage = self.stage_results[-1]
            if last_stage.aggregator_result:
                return last_stage.aggregator_result.raw_message
            elif last_stage.agent_results:
                return last_stage.agent_results[-1].raw_message
        elif self.agent_results:
            return self.agent_results[-1].raw_message
        return None


@dataclass
class WorkflowContext:
    """
    Context available during workflow execution.

    Provides access to:
    - resolved_inputs: Final resolved parameter values
    - outputs: Results from completed agents (keyed by agent_type)
    - user: User profile data
    - event: Event data (for event-triggered workflows)
    - system: System-provided values (today, timezone, etc.)
    """
    workflow_id: str
    execution_id: str
    user_id: str

    # Resolved inputs (after 3-layer resolution)
    resolved_inputs: ResolvedInputs = field(default_factory=ResolvedInputs)

    # Outputs from completed agents
    outputs: Dict[str, Any] = field(default_factory=dict)

    # User profile data
    user: Dict[str, Any] = field(default_factory=dict)

    # Event data (for event-triggered workflows)
    event: Dict[str, Any] = field(default_factory=dict)

    # System-provided values
    system: Dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get value by dot-notation path.

        Examples:
            ctx.get("outputs.WeatherAgent.temperature")
            ctx.get("user.city")
            ctx.get("event.email_id")
            ctx.get("system.today")
        """
        parts = path.split(".", 1)

        if len(parts) == 1:
            # No dot, check top-level attributes
            if hasattr(self, parts[0]):
                attr = getattr(self, parts[0])
                if isinstance(attr, dict):
                    return attr
                return attr
            return default

        root, rest = parts

        # Get root object
        if root == "resolved_inputs":
            current = self.resolved_inputs.values
        elif hasattr(self, root):
            current = getattr(self, root)
        else:
            return default

        # Navigate the rest of the path
        for part in rest.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default

            if current is None:
                return default

        return current

    def get_input(self, name: str, default: Any = None) -> Any:
        """Get a resolved input value by name"""
        return self.resolved_inputs.get(name, default)

    def get_output(self, agent_type: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get output from a completed agent"""
        output = self.outputs.get(agent_type)
        if output is None:
            return default

        if key is None:
            return output

        if isinstance(output, dict):
            return output.get(key, default)

        return default
