"""
Tests for FlowAgent Agent Group System.

Tests:
- Merge strategies
- Map-reduce operations
- AgentGroup parallel and sequential execution
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional

from flowagents.group import (
    # Models
    ExecutionPattern,
    MergeStrategy,
    GroupResult,
    AgentExecutionResult,
    # Group
    AgentGroup,
    GroupExecutionError,
    # Merge
    StateMerger,
    merge_values,
    # MapReduce
    MapReduceExecutor,
)
from flowagents.group.map_reduce import (
    reduce_flatten,
    reduce_sum,
    reduce_max,
    reduce_filter,
)


# ============================================================================
# Test Merge
# ============================================================================

class TestMergeValues:
    """Tests for merge_values function"""

    def test_replace_strategy(self):
        """Test REPLACE strategy (last wins)"""
        result = merge_values([1, 2, 3], MergeStrategy.REPLACE)
        assert result == 3

    def test_first_strategy(self):
        """Test FIRST strategy (first wins)"""
        result = merge_values([1, 2, 3], MergeStrategy.FIRST)
        assert result == 1

    def test_add_numbers(self):
        """Test ADD strategy with numbers"""
        result = merge_values([10, 20, 30], MergeStrategy.ADD)
        assert result == 60

    def test_add_lists(self):
        """Test ADD strategy with lists"""
        result = merge_values([[1, 2], [3, 4], [5]], MergeStrategy.ADD)
        assert result == [1, 2, 3, 4, 5]

    def test_add_strings(self):
        """Test ADD strategy with strings"""
        result = merge_values(["Hello", " ", "World"], MergeStrategy.ADD)
        assert result == "Hello World"

    def test_merge_dicts(self):
        """Test MERGE strategy with dicts"""
        result = merge_values(
            [{"a": 1}, {"b": 2}, {"c": 3}],
            MergeStrategy.MERGE
        )
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_overlapping_dicts(self):
        """Test MERGE with overlapping keys (later wins)"""
        result = merge_values(
            [{"a": 1, "b": 2}, {"b": 3, "c": 4}],
            MergeStrategy.MERGE
        )
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_max_strategy(self):
        """Test MAX strategy"""
        result = merge_values([5, 3, 8, 2], MergeStrategy.MAX)
        assert result == 8

    def test_min_strategy(self):
        """Test MIN strategy"""
        result = merge_values([5, 3, 8, 2], MergeStrategy.MIN)
        assert result == 2

    def test_union_strategy(self):
        """Test UNION strategy"""
        result = merge_values([[1, 2], [2, 3], [3, 4]], MergeStrategy.UNION)
        assert result == {1, 2, 3, 4}

    def test_custom_strategy(self):
        """Test CUSTOM strategy with function"""
        def average(values):
            return sum(values) / len(values)

        result = merge_values([10, 20, 30], MergeStrategy.CUSTOM, custom_fn=average)
        assert result == 20.0

    def test_ignore_none(self):
        """Test ignoring None values"""
        result = merge_values([1, None, 3, None, 5], MergeStrategy.ADD)
        assert result == 9

    def test_empty_list(self):
        """Test with empty list"""
        result = merge_values([], MergeStrategy.ADD)
        assert result is None


class TestStateMerger:
    """Tests for StateMerger"""

    def test_basic_merge(self):
        """Test merging states with strategies"""
        merger = StateMerger(
            merge_strategies={
                "results": MergeStrategy.ADD,
                "score": MergeStrategy.MAX,
            }
        )

        states = [
            {"results": [1, 2], "score": 80},
            {"results": [3, 4], "score": 90},
            {"results": [5], "score": 70},
        ]

        merged = merger.merge(states)

        assert merged["results"] == [1, 2, 3, 4, 5]
        assert merged["score"] == 90

    def test_default_strategy(self):
        """Test default strategy for unspecified fields"""
        merger = StateMerger(default_strategy=MergeStrategy.FIRST)

        states = [
            {"value": "first"},
            {"value": "second"},
        ]

        merged = merger.merge(states)
        assert merged["value"] == "first"

    def test_custom_merge_function(self):
        """Test using custom merge function"""
        def weighted_average(values):
            return sum(values) / len(values)

        merger = StateMerger(
            merge_strategies={"rating": MergeStrategy.CUSTOM},
            custom_merge_fns={"rating": weighted_average}
        )

        states = [
            {"rating": 4.0},
            {"rating": 5.0},
            {"rating": 3.0},
        ]

        merged = merger.merge(states)
        assert merged["rating"] == 4.0

    def test_empty_states(self):
        """Test merging empty states"""
        merger = StateMerger()
        merged = merger.merge([])
        assert merged == {}


# ============================================================================
# Test Map-Reduce
# ============================================================================

class TestMapReduceExecutor:
    """Tests for MapReduceExecutor"""

    @pytest.mark.asyncio
    async def test_basic_map(self):
        """Test basic map operation"""
        executor = MapReduceExecutor(max_concurrency=5)

        async def double(x):
            return x * 2

        result = await executor.execute([1, 2, 3, 4, 5], double)

        assert result.total_items == 5
        assert result.successful_items == 5
        assert result.failed_items == 0
        assert result.reduced_result == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_map_with_reduce(self):
        """Test map with custom reduce"""
        executor = MapReduceExecutor()

        async def square(x):
            return x * x

        result = await executor.execute(
            [1, 2, 3, 4],
            square,
            reduce_fn=reduce_sum
        )

        assert result.reduced_result == 30  # 1 + 4 + 9 + 16

    @pytest.mark.asyncio
    async def test_map_with_errors(self):
        """Test map handling errors"""
        executor = MapReduceExecutor(continue_on_error=True)

        async def risky(x):
            if x == 3:
                raise ValueError("Bad value")
            return x * 2

        result = await executor.execute([1, 2, 3, 4, 5], risky)

        assert result.total_items == 5
        assert result.successful_items == 4
        assert result.failed_items == 1
        assert result.reduced_result == [2, 4, 8, 10]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test concurrency is limited"""
        executor = MapReduceExecutor(max_concurrency=2)
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency(x):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return x

        await executor.execute([1, 2, 3, 4, 5, 6], track_concurrency)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback"""
        executor = MapReduceExecutor()
        progress_updates = []

        async def track_progress(completed, total):
            progress_updates.append((completed, total))

        async def identity(x):
            return x

        await executor.execute(
            [1, 2, 3],
            identity,
            progress_callback=track_progress
        )

        assert len(progress_updates) == 3
        assert progress_updates[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_map_only(self):
        """Test map_only convenience method"""
        executor = MapReduceExecutor()

        async def double(x):
            return x * 2

        result = await executor.map_only([1, 2, 3], double)

        assert result == [2, 4, 6]


class TestReduceFunctions:
    """Tests for built-in reduce functions"""

    def test_reduce_flatten(self):
        """Test flatten reduce"""
        result = reduce_flatten([[1, 2], [3, 4], [5]])
        assert result == [1, 2, 3, 4, 5]

    def test_reduce_sum(self):
        """Test sum reduce"""
        result = reduce_sum([1, 2, 3, 4, 5])
        assert result == 15

    def test_reduce_max(self):
        """Test max reduce"""
        result = reduce_max([3, 1, 4, 1, 5])
        assert result == 5

    def test_reduce_filter(self):
        """Test filter reduce"""
        is_even = lambda x: x % 2 == 0
        reducer = reduce_filter(is_even)
        result = reducer([1, 2, 3, 4, 5, 6])
        assert result == [2, 4, 6]


# ============================================================================
# Test Agent Group
# ============================================================================

class MockReply:
    """Mock reply object"""
    def __init__(self, message, data=None):
        self.raw_message = message
        self.data = data or message


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, agent_type: str, user_id: str, context_hints: Dict[str, Any] = None):
        self.agent_id = f"{agent_type}_{id(self)}"
        self.agent_type = agent_type
        self.user_id = user_id
        self.collected_fields = dict(context_hints or {})
        self.execution_state = {}

    async def reply(self, message):
        # Simulate different behavior per agent type
        if "Search" in self.agent_type:
            self.collected_fields["results"] = [f"result_from_{self.agent_type}"]
            self.collected_fields["source"] = self.agent_type
            return MockReply(f"Found results from {self.agent_type}", {"results": self.collected_fields["results"]})
        else:
            return MockReply(f"Executed {self.agent_type}")


class MockFactory:
    """Mock agent factory"""
    def __init__(self, fail_types: Optional[List[str]] = None):
        self.fail_types = fail_types or []
        self.created_agents = []

    async def create_agent(self, agent_type: str, user_id: str, context_hints: Optional[Dict[str, Any]] = None):
        if agent_type in self.fail_types:
            raise ValueError(f"Failed to create {agent_type}")

        agent = MockAgent(agent_type, user_id, context_hints)
        self.created_agents.append(agent)
        return agent


class TestAgentGroup:
    """Tests for AgentGroup"""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel agent execution"""
        group = AgentGroup(
            agent_types=["GoogleSearch", "ArxivSearch", "WikiSearch"],
            pattern=ExecutionPattern.PARALLEL,
            merge_strategy={
                "results": MergeStrategy.ADD,
            }
        )

        factory = MockFactory()
        result = await group.execute(
            message="search query",
            user_id="user_1",
            factory=factory
        )

        assert result.status == "completed"
        assert result.total_agents == 3
        assert result.completed_agents == 3
        assert len(factory.created_agents) == 3

        # Check merged results
        assert "results" in result.merged_fields
        assert len(result.merged_fields["results"]) == 3

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential agent execution"""
        group = AgentGroup(
            agent_types=["Analyzer", "Summarizer", "Writer"],
            pattern=ExecutionPattern.SEQUENTIAL,
        )

        factory = MockFactory()
        result = await group.execute(
            message="input text",
            user_id="user_1",
            factory=factory
        )

        assert result.status == "completed"
        assert result.total_agents == 3
        assert result.completed_agents == 3

    @pytest.mark.asyncio
    async def test_hierarchical_execution(self):
        """Test hierarchical agent execution"""
        group = AgentGroup(
            agent_types=["Manager", "Worker1", "Worker2"],
            pattern=ExecutionPattern.HIERARCHICAL,
        )

        factory = MockFactory()
        result = await group.execute(
            message="task",
            user_id="user_1",
            factory=factory
        )

        assert result.status == "completed"
        assert result.total_agents == 3
        # Manager + 2 workers

    @pytest.mark.asyncio
    async def test_continue_on_error(self):
        """Test continuing when agent fails"""
        group = AgentGroup(
            agent_types=["Search1", "FailAgent", "Search2"],
            pattern=ExecutionPattern.PARALLEL,
            continue_on_error=True
        )

        factory = MockFactory(fail_types=["FailAgent"])
        result = await group.execute(
            message="query",
            user_id="user_1",
            factory=factory
        )

        assert result.status == "partial"
        assert result.completed_agents == 2
        assert result.failed_agents == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_shared_inputs(self):
        """Test shared inputs passed to all agents"""
        group = AgentGroup(
            agent_types=["Agent1", "Agent2"],
            pattern=ExecutionPattern.PARALLEL,
        )

        factory = MockFactory()
        result = await group.execute(
            message="query",
            user_id="user_1",
            factory=factory,
            shared_inputs={"api_key": "secret123"}
        )

        # All agents should receive shared inputs
        for agent in factory.created_agents:
            assert "api_key" in agent.collected_fields

    @pytest.mark.asyncio
    async def test_set_merge_strategy(self):
        """Test setting merge strategy after creation"""
        group = AgentGroup(
            agent_types=["Agent1", "Agent2"],
            pattern=ExecutionPattern.PARALLEL,
        )

        group.set_merge_strategy("scores", MergeStrategy.MAX)

        assert group.merger.get_strategy("scores") == MergeStrategy.MAX

    @pytest.mark.asyncio
    async def test_add_remove_agent(self):
        """Test adding and removing agents"""
        group = AgentGroup(
            agent_types=["Agent1"],
            pattern=ExecutionPattern.PARALLEL,
        )

        group.add_agent("Agent2")
        assert "Agent2" in group.agent_types

        group.remove_agent("Agent1")
        assert "Agent1" not in group.agent_types
        assert len(group.agent_types) == 1


class TestAgentExecutionResult:
    """Tests for AgentExecutionResult model"""

    def test_success_result(self):
        """Test successful result"""
        result = AgentExecutionResult(
            agent_id="agent_1",
            agent_type="TestAgent",
            status="completed",
            output={"data": "value"},
            raw_message="Done"
        )

        assert result.is_success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result"""
        result = AgentExecutionResult(
            agent_id="agent_1",
            agent_type="TestAgent",
            status="failed",
            error="Something went wrong"
        )

        assert result.is_success is False
        assert result.error == "Something went wrong"


class TestGroupResult:
    """Tests for GroupResult model"""

    def test_add_result(self):
        """Test adding results to group"""
        group_result = GroupResult(
            group_id="group_1",
            pattern=ExecutionPattern.PARALLEL,
            status="completed"
        )

        group_result.add_result(AgentExecutionResult(
            agent_id="a1",
            agent_type="Type1",
            status="completed"
        ))

        group_result.add_result(AgentExecutionResult(
            agent_id="a2",
            agent_type="Type2",
            status="failed",
            error="Failed"
        ))

        assert group_result.total_agents == 2
        assert group_result.completed_agents == 1
        assert group_result.failed_agents == 1
        assert len(group_result.errors) == 1

    def test_success_rate(self):
        """Test success rate calculation"""
        group_result = GroupResult(
            group_id="group_1",
            pattern=ExecutionPattern.PARALLEL,
            status="completed",
            total_agents=4,
            completed_agents=3,
            failed_agents=1
        )

        assert group_result.success_rate == 0.75
