"""
FlowAgent Agent Group - Parallel and sequential agent execution with state merging

This module provides:
- AgentGroup: Execute multiple agents in parallel or sequential patterns
- MergeStrategy: Strategies for merging state from parallel agents
- MapReduce: Dynamic parallelism for processing variable-length collections

Patterns:
- sequential: Agent A → Agent B → Agent C
- parallel: All agents execute simultaneously with state merging
- hierarchical: Manager agent delegates to worker agents

Example usage:
    from flowagents.group import AgentGroup, MergeStrategy

    # Parallel execution with state merging
    group = AgentGroup(
        agents=[GoogleSearchAgent(), ArxivSearchAgent(), WikiSearchAgent()],
        pattern="parallel",
        merge_strategy={
            "results": MergeStrategy.ADD,      # Concatenate lists
            "confidence": MergeStrategy.MAX,   # Keep highest
            "sources": MergeStrategy.UNION,    # Merge unique values
        }
    )

    result = await group.execute(message)
"""

from .models import (
    ExecutionPattern,
    MergeStrategy,
    GroupResult,
    AgentExecutionResult,
)

from .group import (
    AgentGroup,
    GroupExecutionError,
)

from .merge import (
    StateMerger,
    merge_values,
)

from .map_reduce import (
    MapReduceExecutor,
)

__all__ = [
    # Models
    "ExecutionPattern",
    "MergeStrategy",
    "GroupResult",
    "AgentExecutionResult",
    # Group
    "AgentGroup",
    "GroupExecutionError",
    # Merge
    "StateMerger",
    "merge_values",
    # MapReduce
    "MapReduceExecutor",
]
