"""Tests for onevalet.orchestrator.dag_executor

Tests cover:
- topological_sort() — no dependencies, linear chain, diamond, parallel, cycle detection
- SubTaskResult dataclass
"""

import pytest

from onevalet.orchestrator.intent_analyzer import SubTask
from onevalet.orchestrator.dag_executor import topological_sort, SubTaskResult


# ── Tests: topological_sort ──


class TestTopologicalSort:
    def test_empty_input(self):
        assert topological_sort([]) == []

    def test_single_task(self):
        tasks = [SubTask(id=1, description="task 1", domain="general")]
        levels = topological_sort(tasks)

        assert len(levels) == 1
        assert len(levels[0]) == 1
        assert levels[0][0].id == 1

    def test_two_independent_tasks(self):
        """Two tasks with no dependencies should be in the same level."""
        tasks = [
            SubTask(id=1, description="task 1", domain="communication"),
            SubTask(id=2, description="task 2", domain="productivity"),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 1
        assert len(levels[0]) == 2
        ids = {t.id for t in levels[0]}
        assert ids == {1, 2}

    def test_three_independent_tasks(self):
        tasks = [
            SubTask(id=1, description="task 1", domain="communication"),
            SubTask(id=2, description="task 2", domain="productivity"),
            SubTask(id=3, description="task 3", domain="lifestyle"),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_linear_chain(self):
        """A → B → C should produce 3 levels of 1 task each."""
        tasks = [
            SubTask(id=1, description="first", domain="productivity"),
            SubTask(id=2, description="second", domain="communication", depends_on=[1]),
            SubTask(id=3, description="third", domain="lifestyle", depends_on=[2]),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 3
        assert levels[0][0].id == 1
        assert levels[1][0].id == 2
        assert levels[2][0].id == 3

    def test_diamond_dependency(self):
        """Diamond: A → B, A → C, B → D, C → D.

        Expected levels:
        - Level 0: [A]
        - Level 1: [B, C]
        - Level 2: [D]
        """
        tasks = [
            SubTask(id=1, description="A", domain="general"),
            SubTask(id=2, description="B", domain="communication", depends_on=[1]),
            SubTask(id=3, description="C", domain="productivity", depends_on=[1]),
            SubTask(id=4, description="D", domain="lifestyle", depends_on=[2, 3]),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 3
        assert levels[0][0].id == 1
        level1_ids = {t.id for t in levels[1]}
        assert level1_ids == {2, 3}
        assert levels[2][0].id == 4

    def test_partial_parallel(self):
        """Tasks 1,2 independent; task 3 depends on 1 only.

        Expected:
        - Level 0: [1, 2]
        - Level 1: [3]
        """
        tasks = [
            SubTask(id=1, description="email", domain="communication"),
            SubTask(id=2, description="expense", domain="lifestyle"),
            SubTask(id=3, description="follow up", domain="communication", depends_on=[1]),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 2
        level0_ids = {t.id for t in levels[0]}
        assert level0_ids == {1, 2}
        assert levels[1][0].id == 3

    def test_cycle_detection_self_loop(self):
        tasks = [SubTask(id=1, description="self loop", domain="general", depends_on=[1])]
        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort(tasks)

    def test_cycle_detection_two_node(self):
        tasks = [
            SubTask(id=1, description="A", domain="general", depends_on=[2]),
            SubTask(id=2, description="B", domain="general", depends_on=[1]),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort(tasks)

    def test_cycle_detection_three_node(self):
        tasks = [
            SubTask(id=1, description="A", domain="general", depends_on=[3]),
            SubTask(id=2, description="B", domain="general", depends_on=[1]),
            SubTask(id=3, description="C", domain="general", depends_on=[2]),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort(tasks)

    def test_unknown_dependency_ignored(self):
        """Dependencies referencing non-existent task IDs should be silently ignored."""
        tasks = [
            SubTask(id=1, description="task 1", domain="general"),
            SubTask(id=2, description="task 2", domain="general", depends_on=[99]),
        ]
        levels = topological_sort(tasks)

        # Task 2 depends on id=99 which doesn't exist, so it's treated as no dependency
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_sorted_within_level(self):
        """Tasks within a level should be sorted by ID."""
        tasks = [
            SubTask(id=5, description="E", domain="general"),
            SubTask(id=3, description="C", domain="general"),
            SubTask(id=1, description="A", domain="general"),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 1
        assert [t.id for t in levels[0]] == [1, 3, 5]

    def test_complex_dag(self):
        """
        1 → 3
        2 → 3
        2 → 4
        3 → 5
        4 → 5

        Expected levels:
        - Level 0: [1, 2]
        - Level 1: [3, 4]
        - Level 2: [5]
        """
        tasks = [
            SubTask(id=1, description="t1", domain="communication"),
            SubTask(id=2, description="t2", domain="productivity"),
            SubTask(id=3, description="t3", domain="lifestyle", depends_on=[1, 2]),
            SubTask(id=4, description="t4", domain="travel", depends_on=[2]),
            SubTask(id=5, description="t5", domain="general", depends_on=[3, 4]),
        ]
        levels = topological_sort(tasks)

        assert len(levels) == 3
        assert {t.id for t in levels[0]} == {1, 2}
        assert {t.id for t in levels[1]} == {3, 4}
        assert {t.id for t in levels[2]} == {5}


# ── Tests: SubTaskResult dataclass ──


class TestSubTaskResult:
    def test_creation(self):
        result = SubTaskResult(
            sub_task_id=1,
            description="Send email",
            response="Email sent successfully",
            status="completed",
        )
        assert result.sub_task_id == 1
        assert result.status == "completed"
        assert result.duration_ms == 0
        assert result.token_usage == {}

    def test_with_optional_fields(self):
        result = SubTaskResult(
            sub_task_id=2,
            description="Check calendar",
            response="3 events found",
            status="completed",
            duration_ms=1500,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert result.duration_ms == 1500
        assert result.token_usage["prompt_tokens"] == 100

    def test_error_status(self):
        result = SubTaskResult(
            sub_task_id=3,
            description="Failed task",
            response="Connection timeout",
            status="error",
        )
        assert result.status == "error"
