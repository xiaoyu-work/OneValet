"""Tests for onevalet.agents.decorator"""

import pytest

from onevalet.fields import InputField, OutputField
from onevalet.agents.decorator import (
    AGENT_REGISTRY,
    AgentMetadata,
    _extract_fields,
    generate_tool_schema,
    get_agent_metadata,
    get_schema_version,
    is_valet,
    valet,
)


# =========================================================================
# _extract_fields
# =========================================================================


class TestExtractFields:

    def test_extracts_input_and_output_fields(self):
        class MyAgent:
            name = InputField("What's your name?")
            age = InputField("How old?", required=False, default=0)
            result_id = OutputField(str, "ID of result")
            _private = InputField("should be skipped")

        inputs, outputs = _extract_fields(MyAgent)
        assert len(inputs) == 2
        assert inputs[0].name == "name"
        assert inputs[0].required is True
        assert inputs[1].name == "age"
        assert inputs[1].required is False
        assert inputs[1].default == 0
        assert len(outputs) == 1
        assert outputs[0].name == "result_id"

    def test_empty_class(self):
        class Empty:
            pass

        inputs, outputs = _extract_fields(Empty)
        assert inputs == []
        assert outputs == []

    def test_private_fields_skipped(self):
        class WithPrivate:
            _hidden = InputField("hidden")
            __dunder = InputField("dunder")
            visible = InputField("visible")

        inputs, outputs = _extract_fields(WithPrivate)
        assert len(inputs) == 1
        assert inputs[0].name == "visible"

    def test_regular_attributes_ignored(self):
        class Mixed:
            name = InputField("name?")
            counter = 0
            label = "static"
            items = [1, 2, 3]

        inputs, outputs = _extract_fields(Mixed)
        assert len(inputs) == 1
        assert len(outputs) == 0


# =========================================================================
# valet decorator
# =========================================================================


class TestValetDecorator:

    def setup_method(self):
        # Clean up registry between tests
        self._original = dict(AGENT_REGISTRY)

    def teardown_method(self):
        # Restore registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(self._original)

    def test_bare_decorator(self):
        @valet
        class TestAgentBare:
            """A test agent"""
            query = InputField("query?")

        assert "TestAgentBare" in AGENT_REGISTRY
        meta = AGENT_REGISTRY["TestAgentBare"]
        assert meta.description == "A test agent"
        assert len(meta.inputs) == 1

    def test_decorator_with_args(self):
        @valet(capabilities=["email", "messaging"], enable_memory=True)
        class TestAgentWithArgs:
            """Agent with args"""

        meta = AGENT_REGISTRY["TestAgentWithArgs"]
        assert meta.capabilities == ["email", "messaging"]
        assert meta.enable_memory is True

    def test_docstring_cleanup(self):
        @valet
        class TestAgentDoc:
            """  Multiple   spaces
                 and indentation  """

        meta = AGENT_REGISTRY["TestAgentDoc"]
        assert "  " not in meta.description

    def test_no_docstring_uses_classname(self):
        @valet
        class TestAgentNoDoc:
            pass

        meta = AGENT_REGISTRY["TestAgentNoDoc"]
        assert "TestAgentNoDoc" in meta.description

    def test_metadata_attached_to_class(self):
        @valet
        class TestAgentMeta:
            """test"""

        assert hasattr(TestAgentMeta, "_valet_metadata")
        assert isinstance(TestAgentMeta._valet_metadata, AgentMetadata)

    def test_expose_as_tool_default_true(self):
        @valet
        class TestAgentExpose:
            """test"""

        assert AGENT_REGISTRY["TestAgentExpose"].expose_as_tool is True

    def test_expose_as_tool_false(self):
        @valet(expose_as_tool=False)
        class TestAgentHidden:
            """test"""

        assert AGENT_REGISTRY["TestAgentHidden"].expose_as_tool is False


# =========================================================================
# get_agent_metadata / is_valet
# =========================================================================


class TestMetadataHelpers:

    def test_get_metadata_decorated(self):
        @valet
        class DecoratedAgent:
            """test"""

        meta = get_agent_metadata(DecoratedAgent)
        assert meta is not None
        assert meta.name == "DecoratedAgent"

    def test_get_metadata_undecorated(self):
        class PlainClass:
            pass

        assert get_agent_metadata(PlainClass) is None

    def test_is_valet_true(self):
        @valet
        class V:
            """test"""

        assert is_valet(V) is True

    def test_is_valet_false(self):
        class NotV:
            pass

        assert is_valet(NotV) is False


# =========================================================================
# generate_tool_schema
# =========================================================================


class TestGenerateToolSchema:

    def setup_method(self):
        self._original = dict(AGENT_REGISTRY)

    def teardown_method(self):
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(self._original)

    def test_generates_valid_schema(self):
        @valet
        class SchemaAgent:
            """Search for weather data"""

        schema = generate_tool_schema(SchemaAgent)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "SchemaAgent"
        assert schema["function"]["description"] == "Search for weather data"
        params = schema["function"]["parameters"]
        assert "task_instruction" in params["properties"]
        assert params["required"] == ["task_instruction"]

    def test_undecorated_raises(self):
        class NotDecorated:
            pass

        with pytest.raises(ValueError, match="not decorated"):
            generate_tool_schema(NotDecorated)


# =========================================================================
# get_schema_version
# =========================================================================


class TestGetSchemaVersion:

    def setup_method(self):
        self._original = dict(AGENT_REGISTRY)

    def teardown_method(self):
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(self._original)

    def test_deterministic(self):
        @valet
        class VersionAgent:
            """test"""
            name = InputField("name?")
            age = InputField("age?", required=False, default=0)

        v1 = get_schema_version(VersionAgent)
        v2 = get_schema_version(VersionAgent)
        assert v1 == v2
        assert isinstance(v1, int)

    def test_different_fields_different_version(self):
        @valet
        class AgentA:
            """test"""
            name = InputField("name?")

        @valet
        class AgentB:
            """test"""
            email = InputField("email?")

        assert get_schema_version(AgentA) != get_schema_version(AgentB)

    def test_undecorated_returns_zero(self):
        class Plain:
            pass

        assert get_schema_version(Plain) == 0

    def test_no_inputs_stable(self):
        @valet
        class NoInputAgent:
            """test"""

        v = get_schema_version(NoInputAgent)
        assert isinstance(v, int)
