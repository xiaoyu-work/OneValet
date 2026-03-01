"""DAG Executor - Topological sort and result tracking for multi-intent execution.

Provides utilities for ordering sub-tasks by their dependencies and
tracking execution results.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .intent_analyzer import SubTask

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    """Result of a single sub-task execution."""

    sub_task_id: int
    description: str
    response: str
    status: str  # "completed" or "error"
    duration_ms: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)


def topological_sort(sub_tasks: List[SubTask]) -> List[List[SubTask]]:
    """Sort sub-tasks into parallel execution levels.

    Returns a list of levels. Tasks within the same level have all their
    dependencies satisfied by prior levels and can execute in parallel.

    Raises:
        ValueError: If a dependency cycle is detected.
    """
    if not sub_tasks:
        return []

    id_to_task = {st.id: st for st in sub_tasks}
    in_degree = {st.id: 0 for st in sub_tasks}
    dependents: Dict[int, List[int]] = defaultdict(list)

    for st in sub_tasks:
        for dep_id in st.depends_on:
            if dep_id in id_to_task:
                in_degree[st.id] += 1
                dependents[dep_id].append(st.id)

    levels: List[List[SubTask]] = []
    remaining = set(id_to_task.keys())

    while remaining:
        # Collect all tasks with in_degree 0
        ready_ids = [sid for sid in remaining if in_degree[sid] == 0]
        if not ready_ids:
            raise ValueError(
                f"Cycle detected in sub-task dependencies. "
                f"Remaining tasks: {remaining}"
            )

        level = [id_to_task[sid] for sid in sorted(ready_ids)]
        levels.append(level)

        for sid in ready_ids:
            remaining.remove(sid)
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1

    return levels
