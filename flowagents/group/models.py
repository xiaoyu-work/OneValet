"""
FlowAgent Group Models - Data structures for agent groups

This module defines:
- ExecutionPattern: Sequential, parallel, hierarchical
- MergeStrategy: How to merge state from parallel agents
- GroupResult: Result from group execution
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Callable
from enum import Enum


class ExecutionPattern(str, Enum):
    """Pattern for executing agents in a group"""
    SEQUENTIAL = "sequential"      # A → B → C
    PARALLEL = "parallel"          # A, B, C all at once
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers


class MergeStrategy(str, Enum):
    """
    Strategies for merging state from parallel agents.

    | Strategy | Field Type | Example |
    |----------|------------|---------|
    | REPLACE  | Any        | [1, 2, 3] → 3 (last) |
    | ADD      | List       | [[1,2], [3,4]] → [1,2,3,4] |
    | ADD      | Number     | [10, 20, 30] → 60 |
    | MERGE    | Dict       | [{a:1}, {b:2}] → {a:1, b:2} |
    | MAX      | Number     | [5, 3, 8] → 8 |
    | MIN      | Number     | [5, 3, 8] → 3 |
    | UNION    | Set/List   | [[1,2], [2,3]] → {1,2,3} |
    | FIRST    | Any        | [1, 2, 3] → 1 |
    | CUSTOM   | Any        | Use custom function |
    """
    REPLACE = "replace"    # Last write wins (default)
    ADD = "add"           # Concatenate lists or add numbers
    MERGE = "merge"       # Merge dictionaries
    MAX = "max"           # Keep maximum value
    MIN = "min"           # Keep minimum value
    UNION = "union"       # Set union (unique values)
    FIRST = "first"       # Keep first non-null value
    CUSTOM = "custom"     # Use custom merge function


@dataclass
class AgentExecutionResult:
    """Result from executing a single agent within a group"""
    agent_id: str
    agent_type: str
    status: Literal["completed", "failed", "skipped"]

    # Output data
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    raw_message: Optional[str] = None

    # Data output (for passing to next agent)
    output: Any = None

    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if execution succeeded"""
        return self.status == "completed"


@dataclass
class GroupResult:
    """Result from executing an agent group"""
    group_id: str
    pattern: ExecutionPattern
    status: Literal["completed", "partial", "failed"]

    # Individual agent results
    agent_results: List[AgentExecutionResult] = field(default_factory=list)

    # Merged state (for parallel execution)
    merged_fields: Dict[str, Any] = field(default_factory=dict)

    # Final output (from last agent or merged)
    final_output: Any = None
    final_message: Optional[str] = None

    # Statistics
    total_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0

    # Error summary
    errors: List[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total execution duration"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_agents == 0:
            return 0.0
        return self.completed_agents / self.total_agents

    def add_result(self, result: AgentExecutionResult) -> None:
        """Add an agent result and update statistics"""
        self.agent_results.append(result)
        self.total_agents += 1

        if result.is_success:
            self.completed_agents += 1
        else:
            self.failed_agents += 1
            if result.error:
                self.errors.append(f"{result.agent_type}: {result.error}")


@dataclass
class MergeConfig:
    """Configuration for field merging"""
    field_name: str
    strategy: MergeStrategy
    custom_fn: Optional[Callable] = None

    # Options for specific strategies
    default_value: Any = None  # Default if all values are None
    ignore_none: bool = True   # Skip None values when merging


@dataclass
class GroupConfig:
    """Configuration for an agent group"""
    group_id: str
    pattern: ExecutionPattern = ExecutionPattern.PARALLEL

    # Agents in the group
    agent_types: List[str] = field(default_factory=list)

    # Merge configuration
    merge_strategies: Dict[str, MergeStrategy] = field(default_factory=dict)
    custom_merge_fns: Dict[str, Callable] = field(default_factory=dict)

    # Execution settings
    max_concurrency: int = 10
    timeout_seconds: int = 300
    continue_on_error: bool = True  # Continue with other agents if one fails

    # Input mapping
    shared_inputs: Dict[str, Any] = field(default_factory=dict)

    def get_merge_config(self, field_name: str) -> MergeConfig:
        """Get merge configuration for a field"""
        strategy = self.merge_strategies.get(field_name, MergeStrategy.REPLACE)
        custom_fn = self.custom_merge_fns.get(field_name)

        return MergeConfig(
            field_name=field_name,
            strategy=strategy,
            custom_fn=custom_fn
        )
