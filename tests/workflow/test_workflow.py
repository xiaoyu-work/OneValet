"""
Tests for FlowAgent Workflow System.

Tests the 4-keyword workflow system:
- run: Sequential execution
- parallel: Parallel execution
- then: Aggregator pattern
- stages: Multi-stage execution

Also tests the three-layer parameter resolution:
- Template defaults
- User profile values
- Instance overrides
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional

from flowagents.workflow import (
    # Models
    WorkflowType,
    WorkflowStatus,
    StageStatus,
    StageDefinition,
    WorkflowTemplate,
    UserProfile,
    UserWorkflowInstance,
    ResolvedInputs,
    WorkflowResult,
    WorkflowContext,
    # Loader
    WorkflowLoader,
    WorkflowLoadError,
    WorkflowValidationError,
    # Resolver
    ParameterResolver,
    AgentInputMatcher,
    # Executor
    WorkflowExecutor,
    SimpleAgentFactory,
)


# ============================================================================
# Test Models
# ============================================================================

class TestWorkflowTemplate:
    """Tests for WorkflowTemplate model"""

    def test_sequential_workflow(self):
        """Test creating a sequential workflow"""
        workflow = WorkflowTemplate(
            id="morning_brief",
            description="Morning briefing",
            type=WorkflowType.INTERACTIVE,
            triggers=["morning brief", "good morning"],
            inputs={"location": "{{user.city}}"},
            run=["WeatherAgent", "CalendarAgent", "NewsAgent"]
        )

        assert workflow.id == "morning_brief"
        assert workflow.get_execution_type() == "run"
        assert len(workflow.run) == 3
        assert workflow.validate() == []

    def test_parallel_workflow(self):
        """Test creating a parallel workflow with aggregator"""
        workflow = WorkflowTemplate(
            id="travel_research",
            type=WorkflowType.INTERACTIVE,
            triggers=["plan trip"],
            inputs={"destination": None},
            parallel=["FlightAgent", "HotelAgent", "RestaurantAgent"],
            then="TravelPlannerAgent"
        )

        assert workflow.get_execution_type() == "parallel"
        assert len(workflow.parallel) == 3
        assert workflow.then == "TravelPlannerAgent"
        assert workflow.validate() == []

    def test_multi_stage_workflow(self):
        """Test creating a multi-stage workflow"""
        workflow = WorkflowTemplate(
            id="email_processing",
            type=WorkflowType.EVENT_TRIGGERED,
            trigger_config={"event_type": "email_received"},
            inputs={"email_id": "{{event.email_id}}"},
            stages=[
                StageDefinition(
                    parallel=["SpamDetector", "UrgencyClassifier"],
                    then="ClassificationAggregator"
                ),
                StageDefinition(run=["EmailRouter"]),
                StageDefinition(parallel=["NotificationAgent", "DatabaseAgent"])
            ]
        )

        assert workflow.get_execution_type() == "stages"
        assert len(workflow.stages) == 3
        assert workflow.validate() == []

    def test_scheduled_workflow(self):
        """Test creating a scheduled workflow"""
        workflow = WorkflowTemplate(
            id="daily_digest",
            type=WorkflowType.SCHEDULED,
            schedule="0 9 * * *",
            inputs={"format": "brief"},
            run=["DigestAgent"]
        )

        assert workflow.type == WorkflowType.SCHEDULED
        assert workflow.schedule == "0 9 * * *"
        assert workflow.validate() == []

    def test_validation_errors(self):
        """Test workflow validation catches errors"""
        # No execution defined
        workflow = WorkflowTemplate(id="empty")
        errors = workflow.validate()
        assert any("run, parallel, or stages" in e for e in errors)

        # Scheduled without schedule
        workflow = WorkflowTemplate(
            id="bad_scheduled",
            type=WorkflowType.SCHEDULED,
            run=["Agent1"]
        )
        errors = workflow.validate()
        assert any("schedule" in e for e in errors)

        # Event-triggered without trigger_config
        workflow = WorkflowTemplate(
            id="bad_event",
            type=WorkflowType.EVENT_TRIGGERED,
            run=["Agent1"]
        )
        errors = workflow.validate()
        assert any("trigger_config" in e for e in errors)


class TestUserProfile:
    """Tests for UserProfile model"""

    def test_user_profile_creation(self):
        """Test creating a user profile"""
        profile = UserProfile(
            id="alice",
            email="alice@example.com",
            data={
                "city": "San Francisco",
                "news_preferences": ["tech", "ai"],
                "weather_units": "fahrenheit"
            }
        )

        assert profile.id == "alice"
        assert profile.get("city") == "San Francisco"
        assert profile.get("news_preferences") == ["tech", "ai"]
        assert profile.get("missing", "default") == "default"


class TestResolvedInputs:
    """Tests for ResolvedInputs model"""

    def test_resolved_inputs(self):
        """Test resolved inputs container"""
        resolved = ResolvedInputs(
            values={"location": "London", "topics": ["tech"]},
            sources={"location": "instance", "topics": "profile"}
        )

        assert resolved.get("location") == "London"
        assert resolved["topics"] == ["tech"]
        assert "location" in resolved
        assert "missing" not in resolved


# ============================================================================
# Test Loader
# ============================================================================

class TestWorkflowLoader:
    """Tests for WorkflowLoader"""

    def test_load_from_dict(self):
        """Test loading workflow from dictionary"""
        loader = WorkflowLoader()

        data = {
            "workflows": {
                "morning_brief": {
                    "type": "interactive",
                    "description": "Morning briefing",
                    "triggers": ["good morning"],
                    "inputs": {"location": "{{user.city}}"},
                    "run": ["WeatherAgent", "NewsAgent"]
                }
            }
        }

        workflows = loader.load_from_dict(data)
        assert len(workflows) == 1

        workflow = loader.get("morning_brief")
        assert workflow is not None
        assert workflow.description == "Morning briefing"
        assert len(workflow.run) == 2

    def test_trigger_matching(self):
        """Test finding workflow by trigger"""
        loader = WorkflowLoader()

        loader.load_from_dict({
            "workflows": {
                "morning": {
                    "triggers": ["good morning", "morning brief"],
                    "run": ["Agent1"]
                },
                "travel": {
                    "triggers": ["plan trip to"],
                    "run": ["Agent2"]
                }
            }
        })

        # Exact match
        workflow = loader.match_trigger("good morning")
        assert workflow.id == "morning"

        # Prefix match
        workflow = loader.match_trigger("plan trip to Paris")
        assert workflow.id == "travel"

        # No match
        workflow = loader.match_trigger("random message")
        assert workflow is None

    def test_load_parallel_workflow(self):
        """Test loading parallel workflow with aggregator"""
        loader = WorkflowLoader()

        data = {
            "workflows": {
                "research": {
                    "triggers": ["research"],
                    "parallel": ["Agent1", "Agent2", "Agent3"],
                    "then": "AggregatorAgent"
                }
            }
        }

        loader.load_from_dict(data)
        workflow = loader.get("research")

        assert workflow.get_execution_type() == "parallel"
        assert len(workflow.parallel) == 3
        assert workflow.then == "AggregatorAgent"

    def test_load_stages_workflow(self):
        """Test loading multi-stage workflow"""
        loader = WorkflowLoader()

        data = {
            "workflows": {
                "complex": {
                    "type": "event_triggered",
                    "trigger_config": {"event_type": "new_email"},
                    "stages": [
                        {"parallel": ["A1", "A2"], "then": "Agg1"},
                        {"run": ["A3"]},
                        {"parallel": ["A4", "A5"]}
                    ]
                }
            }
        }

        loader.load_from_dict(data)
        workflow = loader.get("complex")

        assert workflow.get_execution_type() == "stages"
        assert len(workflow.stages) == 3
        assert workflow.stages[0].parallel == ["A1", "A2"]
        assert workflow.stages[0].then == "Agg1"
        assert workflow.stages[1].run == ["A3"]

    def test_get_workflows_by_type(self):
        """Test filtering workflows by type"""
        loader = WorkflowLoader()

        loader.load_from_dict({
            "workflows": {
                "w1": {"type": "interactive", "triggers": ["hi"], "run": ["A1"]},
                "w2": {"type": "scheduled", "schedule": "0 9 * * *", "run": ["A2"]},
                "w3": {"type": "interactive", "triggers": ["bye"], "run": ["A3"]}
            }
        })

        interactive = loader.get_interactive_workflows()
        assert len(interactive) == 2

        scheduled = loader.get_scheduled_workflows()
        assert len(scheduled) == 1


# ============================================================================
# Test Resolver
# ============================================================================

class TestParameterResolver:
    """Tests for ParameterResolver"""

    def test_basic_resolution(self):
        """Test basic parameter resolution"""
        resolver = ParameterResolver()

        template = WorkflowTemplate(
            id="test",
            inputs={
                "location": "{{user.city}}",
                "format": "detailed"
            },
            run=["Agent1"]
        )

        profile = UserProfile(
            id="alice",
            data={"city": "San Francisco"}
        )

        resolved = resolver.resolve(template=template, user_profile=profile)

        assert resolved.get("location") == "San Francisco"
        assert resolved.get("format") == "detailed"
        assert resolved.sources["location"] == "profile"
        assert resolved.sources["format"] == "template"

    def test_instance_override(self):
        """Test instance values override profile values"""
        resolver = ParameterResolver()

        template = WorkflowTemplate(
            id="test",
            inputs={"location": "{{user.city}}"},
            run=["Agent1"]
        )

        profile = UserProfile(
            id="alice",
            data={"city": "San Francisco"}
        )

        instance = UserWorkflowInstance(
            id="alice_travel",
            user_id="alice",
            template_id="test",
            inputs={"location": "London"}
        )

        resolved = resolver.resolve(
            template=template,
            user_profile=profile,
            instance=instance
        )

        assert resolved.get("location") == "London"
        assert resolved.sources["location"] == "instance"

    def test_list_value_resolution(self):
        """Test resolving list values from profile"""
        resolver = ParameterResolver()

        template = WorkflowTemplate(
            id="test",
            inputs={"topics": "{{user.news_preferences}}"},
            run=["Agent1"]
        )

        profile = UserProfile(
            id="alice",
            data={"news_preferences": ["tech", "startups", "ai"]}
        )

        resolved = resolver.resolve(template=template, user_profile=profile)

        assert resolved.get("topics") == ["tech", "startups", "ai"]
        assert isinstance(resolved.get("topics"), list)

    def test_system_values(self):
        """Test system-provided values"""
        resolver = ParameterResolver(default_timezone="America/New_York")

        template = WorkflowTemplate(
            id="test",
            inputs={"date": "{{today}}"},
            run=["Agent1"]
        )

        resolved = resolver.resolve(template=template)

        # Should have today's date in YYYY-MM-DD format
        today = resolved.get("today")
        assert today is not None
        assert len(today) == 10  # YYYY-MM-DD

        # Should have timezone
        assert resolved.get("timezone") == "America/New_York"

    def test_event_data_resolution(self):
        """Test resolving event data for event-triggered workflows"""
        resolver = ParameterResolver()

        template = WorkflowTemplate(
            id="test",
            type=WorkflowType.EVENT_TRIGGERED,
            trigger_config={"event_type": "email_received"},
            inputs={"email_id": "{{event.email_id}}"},
            run=["Agent1"]
        )

        resolved = resolver.resolve(
            template=template,
            event_data={"email_id": "msg_12345", "sender": "bob@example.com"}
        )

        assert resolved.get("email_id") == "msg_12345"


class TestAgentInputMatcher:
    """Tests for AgentInputMatcher"""

    def test_match_required_fields(self):
        """Test matching required fields"""
        matcher = AgentInputMatcher()

        resolved = ResolvedInputs(
            values={"location": "SF", "topics": ["tech"], "extra": "value"}
        )

        matched = matcher.match_inputs(
            resolved_inputs=resolved,
            agent_type="WeatherAgent",
            required_fields=["location"],
            optional_fields=["format"]
        )

        assert matched["location"] == "SF"
        assert "extra" not in matched
        assert "format" not in matched

    def test_missing_required_field(self):
        """Test error when required field is missing"""
        matcher = AgentInputMatcher()

        resolved = ResolvedInputs(values={"other": "value"})

        with pytest.raises(Exception) as exc:
            matcher.match_inputs(
                resolved_inputs=resolved,
                agent_type="WeatherAgent",
                required_fields=["location"]
            )

        assert "location" in str(exc.value)


# ============================================================================
# Test Executor
# ============================================================================

class MockAgent:
    """Mock agent for testing"""

    def __init__(self, user_id: str, context_hints: Dict[str, Any] = None):
        self.user_id = user_id
        self.context_hints = context_hints or {}
        self.executed = False

    async def _execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        self.executed = True
        return {
            "result": f"Executed with {fields}",
            "agent_type": self.__class__.__name__
        }


class WeatherAgent(MockAgent):
    async def _execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        return {"temperature": 72, "condition": "sunny"}


class NewsAgent(MockAgent):
    async def _execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        return {"headlines": ["Tech news 1", "AI news 2"]}


class DigestAgent(MockAgent):
    async def _execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        # Aggregator - receives outputs from previous agents
        outputs = fields.get("outputs", {})
        return {"summary": f"Combined {len(outputs)} sources"}


@pytest.fixture
def agent_factory():
    """Create agent factory with mock agents"""
    factory = SimpleAgentFactory()
    factory.register("WeatherAgent", WeatherAgent, required_fields=["location"])
    factory.register("NewsAgent", NewsAgent, required_fields=["topics"])
    factory.register("DigestAgent", DigestAgent, required_fields=[])
    return factory


class TestWorkflowExecutor:
    """Tests for WorkflowExecutor"""

    @pytest.mark.asyncio
    async def test_sequential_execution(self, agent_factory):
        """Test sequential workflow execution"""
        executor = WorkflowExecutor(agent_factory=agent_factory)

        workflow = WorkflowTemplate(
            id="test_seq",
            inputs={"location": "SF", "topics": ["tech"]},
            run=["WeatherAgent", "NewsAgent"]
        )

        result = await executor.execute(
            workflow=workflow,
            user_id="test_user"
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.agent_results) == 2
        assert "WeatherAgent" in result.outputs
        assert "NewsAgent" in result.outputs

    @pytest.mark.asyncio
    async def test_parallel_execution(self, agent_factory):
        """Test parallel workflow execution"""
        executor = WorkflowExecutor(agent_factory=agent_factory)

        workflow = WorkflowTemplate(
            id="test_parallel",
            inputs={"location": "SF", "topics": ["tech"]},
            parallel=["WeatherAgent", "NewsAgent"]
        )

        result = await executor.execute(
            workflow=workflow,
            user_id="test_user"
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.agent_results) == 2

    @pytest.mark.asyncio
    async def test_parallel_with_aggregator(self, agent_factory):
        """Test parallel workflow with aggregator"""
        executor = WorkflowExecutor(agent_factory=agent_factory)

        workflow = WorkflowTemplate(
            id="test_agg",
            inputs={"location": "SF", "topics": ["tech"]},
            parallel=["WeatherAgent", "NewsAgent"],
            then="DigestAgent"
        )

        result = await executor.execute(
            workflow=workflow,
            user_id="test_user"
        )

        assert result.status == WorkflowStatus.COMPLETED
        # 2 parallel + 1 aggregator
        assert len(result.agent_results) == 3
        assert "DigestAgent" in result.outputs

    @pytest.mark.asyncio
    async def test_execution_with_profile(self, agent_factory):
        """Test execution with user profile parameter resolution"""
        executor = WorkflowExecutor(agent_factory=agent_factory)

        workflow = WorkflowTemplate(
            id="test_profile",
            inputs={
                "location": "{{user.city}}",
                "topics": "{{user.interests}}"
            },
            run=["WeatherAgent", "NewsAgent"]
        )

        profile = UserProfile(
            id="alice",
            data={
                "city": "New York",
                "interests": ["finance", "tech"]
            }
        )

        result = await executor.execute(
            workflow=workflow,
            user_id="alice",
            user_profile=profile
        )

        assert result.status == WorkflowStatus.COMPLETED


class TestWorkflowContext:
    """Tests for WorkflowContext"""

    def test_get_nested_value(self):
        """Test getting nested values from context"""
        context = WorkflowContext(
            workflow_id="test",
            execution_id="exec_123",
            user_id="alice",
            outputs={
                "WeatherAgent": {"temperature": 72},
                "NewsAgent": {"headlines": ["News 1"]}
            },
            user={"city": "SF", "preferences": {"theme": "dark"}}
        )

        # Simple access
        assert context.get("user.city") == "SF"

        # Nested access
        assert context.get("outputs.WeatherAgent.temperature") == 72
        assert context.get("user.preferences.theme") == "dark"

        # Missing value
        assert context.get("user.missing", "default") == "default"

    def test_get_output(self):
        """Test getting agent outputs"""
        context = WorkflowContext(
            workflow_id="test",
            execution_id="exec_123",
            user_id="alice",
            outputs={"WeatherAgent": {"temp": 72, "condition": "sunny"}}
        )

        # Get full output
        output = context.get_output("WeatherAgent")
        assert output["temp"] == 72

        # Get specific key
        temp = context.get_output("WeatherAgent", "temp")
        assert temp == 72

        # Missing agent
        missing = context.get_output("MissingAgent", default="none")
        assert missing == "none"
