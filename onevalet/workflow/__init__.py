"""
OneValet Workflow System - Multi-agent workflow orchestration

This module provides the workflow system for orchestrating multiple agents:

Workflow Types:
- scheduled: Triggered by cron expression
- event_triggered: Triggered by system events
- interactive: Triggered by user message

Execution Patterns (4 keywords only):
- run: Sequential execution of agents
- parallel: Parallel execution of agents
- then: Aggregator agent (receives results from previous step)
- stages: Multi-phase workflow

Three-Layer Parameter Resolution:
- Template: Default values (shared across all users)
- Profile: User preferences (per-user defaults)
- Instance: User overrides (per-user customization)
Priority: Instance > Profile > Template

Example usage:
    from onevalet.workflow import (
        WorkflowLoader,
        WorkflowExecutor,
        ParameterResolver,
    )

    # Load workflows from YAML
    loader = WorkflowLoader()
    loader.load_from_file("workflows.yaml")

    # Find workflow matching user message
    workflow = loader.match_trigger("good morning")

    # Execute workflow
    executor = WorkflowExecutor(agent_factory=my_factory)
    result = await executor.execute(
        workflow=workflow,
        user_id="alice",
        user_profile=alice_profile
    )
"""

# Models
from .models import (
    # Enums
    WorkflowType,
    WorkflowStatus,
    StageStatus,
    # Stage definition
    StageDefinition,
    # Three-layer system
    WorkflowTemplate,
    UserProfile,
    UserWorkflowInstance,
    ResolvedInputs,
    # Results
    WorkflowAgentResult,
    StageResult,
    WorkflowResult,
    # Execution tracking
    WorkflowExecution,
    WorkflowContext,
)

# Loader
from .loader import (
    WorkflowLoader,
    WorkflowLoadError,
    WorkflowValidationError,
)

# Resolver
from .resolver import (
    ParameterResolver,
    AgentInputMatcher,
    ResolutionError,
)

# Executor
from .executor import (
    WorkflowExecutor,
    SimpleAgentFactory,
    ExecutionError,
    AgentFactoryProtocol,
    AgentExecutorProtocol,
)

__all__ = [
    # Enums
    "WorkflowType",
    "WorkflowStatus",
    "StageStatus",
    # Stage definition
    "StageDefinition",
    # Three-layer system
    "WorkflowTemplate",
    "UserProfile",
    "UserWorkflowInstance",
    "ResolvedInputs",
    # Results
    "WorkflowAgentResult",
    "StageResult",
    "WorkflowResult",
    # Execution tracking
    "WorkflowExecution",
    "WorkflowContext",
    # Loader
    "WorkflowLoader",
    "WorkflowLoadError",
    "WorkflowValidationError",
    # Resolver
    "ParameterResolver",
    "AgentInputMatcher",
    "ResolutionError",
    # Executor
    "WorkflowExecutor",
    "SimpleAgentFactory",
    "ExecutionError",
    "AgentFactoryProtocol",
    "AgentExecutorProtocol",
]
