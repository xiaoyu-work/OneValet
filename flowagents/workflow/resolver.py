"""
FlowAgent Parameter Resolver - Three-layer parameter resolution

This module implements the three-layer parameter resolution system:
1. Template: Default values (shared across all users)
2. Profile: User preferences (per-user defaults)
3. Instance: User overrides (per-user customization)

Priority: Instance > Profile > Template

Supports variable syntax:
- {{user.FIELD}} - From user profile
- {{event.FIELD}} - From event data
- {{today}} - Current date (YYYY-MM-DD)
- {{timezone}} - User's timezone
- Direct values - Override values
"""

import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Literal
from zoneinfo import ZoneInfo

from .models import (
    WorkflowTemplate,
    UserProfile,
    UserWorkflowInstance,
    ResolvedInputs,
)


class ResolutionError(Exception):
    """Raised when parameter resolution fails"""
    pass


class ParameterResolver:
    """
    Resolves workflow parameters using three-layer priority.

    Resolution order (later wins):
    1. Template defaults
    2. User profile values (via {{user.xxx}} syntax)
    3. Instance overrides

    Example:
        resolver = ParameterResolver()

        resolved = resolver.resolve(
            template=workflow_template,
            user_profile=user_profile,
            instance=user_instance,
            event_data={"email_id": "123"}
        )

        # Access resolved values
        location = resolved.get("location")
        topics = resolved.get("topics")
    """

    # Pattern to match {{variable}} syntax
    VARIABLE_PATTERN = re.compile(r"\{\{\s*(\w+(?:\.\w+)*)\s*\}\}")

    def __init__(self, default_timezone: str = "UTC"):
        self.default_timezone = default_timezone

    def resolve(
        self,
        template: WorkflowTemplate,
        user_profile: Optional[UserProfile] = None,
        instance: Optional[UserWorkflowInstance] = None,
        event_data: Optional[Dict[str, Any]] = None,
        system_overrides: Optional[Dict[str, Any]] = None
    ) -> ResolvedInputs:
        """
        Resolve all parameters for a workflow execution.

        Args:
            template: Workflow template with default input values
            user_profile: User profile with preferences (Layer 2)
            instance: User-specific overrides (Layer 3)
            event_data: Event data for event-triggered workflows
            system_overrides: Additional system values to inject

        Returns:
            ResolvedInputs with all resolved values and their sources
        """
        result = ResolvedInputs()

        # Build context for variable resolution
        context = self._build_context(
            user_profile=user_profile,
            event_data=event_data,
            system_overrides=system_overrides
        )

        # Process each input from template
        for key, template_value in template.inputs.items():
            resolved_value, source = self._resolve_value(
                key=key,
                template_value=template_value,
                instance_inputs=instance.inputs if instance else {},
                context=context
            )
            result.values[key] = resolved_value
            result.sources[key] = source

        # Add system-provided values if not already present
        system_values = {
            "user_id": user_profile.id if user_profile else None,
            "today": context["system"]["today"],
            "timezone": context["system"]["timezone"],
        }

        for key, value in system_values.items():
            if key not in result.values and value is not None:
                result.values[key] = value
                result.sources[key] = "system"

        return result

    def _build_context(
        self,
        user_profile: Optional[UserProfile] = None,
        event_data: Optional[Dict[str, Any]] = None,
        system_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build the context dictionary for variable resolution"""

        # Get timezone from user profile or default
        timezone = self.default_timezone
        if user_profile and user_profile.data.get("timezone"):
            timezone = user_profile.data["timezone"]

        # Build system context
        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()

        system_context = {
            "today": now.strftime("%Y-%m-%d"),
            "now": now.isoformat(),
            "timezone": timezone,
            "year": now.year,
            "month": now.month,
            "day": now.day,
        }

        # Apply any system overrides
        if system_overrides:
            system_context.update(system_overrides)

        return {
            "user": user_profile.data if user_profile else {},
            "event": event_data or {},
            "system": system_context,
            # Shortcuts for common variables
            "today": system_context["today"],
            "timezone": timezone,
        }

    def _resolve_value(
        self,
        key: str,
        template_value: Any,
        instance_inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Any, Literal["template", "profile", "instance", "system"]]:
        """
        Resolve a single value through the three layers.

        Returns:
            Tuple of (resolved_value, source)
        """
        # Layer 3: Instance override (highest priority)
        if key in instance_inputs:
            value = instance_inputs[key]
            if value is not None:
                # Resolve any variables in instance value too
                resolved = self._resolve_variables(value, context)
                return resolved, "instance"

        # Layer 1 & 2: Template value with profile resolution
        if template_value is None:
            return None, "template"

        # If template value contains variables, resolve them
        if self._contains_variables(template_value):
            resolved = self._resolve_variables(template_value, context)
            # Determine source based on what was resolved
            source = self._determine_source(template_value, resolved)
            return resolved, source
        else:
            # Static template value
            return template_value, "template"

    def _contains_variables(self, value: Any) -> bool:
        """Check if a value contains variable references"""
        if isinstance(value, str):
            return bool(self.VARIABLE_PATTERN.search(value))
        elif isinstance(value, list):
            return any(self._contains_variables(item) for item in value)
        elif isinstance(value, dict):
            return any(
                self._contains_variables(v) for v in value.values()
            )
        return False

    def _resolve_variables(self, value: Any, context: Dict[str, Any]) -> Any:
        """Recursively resolve variables in a value"""
        if isinstance(value, str):
            return self._resolve_string(value, context)
        elif isinstance(value, list):
            return [self._resolve_variables(item, context) for item in value]
        elif isinstance(value, dict):
            return {
                k: self._resolve_variables(v, context)
                for k, v in value.items()
            }
        return value

    def _resolve_string(self, value: str, context: Dict[str, Any]) -> Any:
        """
        Resolve variables in a string value.

        If the entire string is a single variable (e.g., "{{user.topics}}"),
        the resolved value maintains its original type (list, dict, etc.).

        If the string contains mixed content (e.g., "Hello {{user.name}}"),
        the result is a string with variables interpolated.
        """
        # Check if the entire string is a single variable
        match = self.VARIABLE_PATTERN.fullmatch(value.strip())
        if match:
            # Single variable - preserve type
            var_path = match.group(1)
            resolved = self._get_context_value(var_path, context)
            return resolved

        # Mixed content - interpolate as strings
        def replace_var(m):
            var_path = m.group(1)
            resolved = self._get_context_value(var_path, context)
            if resolved is None:
                return ""
            return str(resolved)

        return self.VARIABLE_PATTERN.sub(replace_var, value)

    def _get_context_value(
        self,
        path: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Get a value from context using dot notation.

        Examples:
            _get_context_value("user.city", context)
            _get_context_value("event.email_id", context)
            _get_context_value("today", context)
        """
        parts = path.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

            if current is None:
                return None

        return current

    def _determine_source(
        self,
        template_value: Any,
        resolved_value: Any
    ) -> Literal["template", "profile", "instance", "system"]:
        """
        Determine the source of a resolved value.

        If the template contains {{user.xxx}}, source is "profile".
        If the template contains {{event.xxx}} or {{system.xxx}}, source is "system".
        Otherwise, source is "template".
        """
        if not isinstance(template_value, str):
            return "template"

        if "{{user." in template_value:
            return "profile"
        elif "{{event." in template_value or "{{system." in template_value:
            return "system"
        elif "{{today}}" in template_value or "{{timezone}}" in template_value:
            return "system"
        else:
            return "template"

    def validate_required_inputs(
        self,
        resolved: ResolvedInputs,
        template: WorkflowTemplate,
        agent_fields: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Validate that all required inputs are present.

        Args:
            resolved: Resolved inputs
            template: Workflow template
            agent_fields: Optional map of agent_type -> required field names

        Returns:
            List of missing required fields (empty if all present)
        """
        missing = []

        # Check template-level required inputs (null means required)
        for key, template_value in template.inputs.items():
            if template_value is None and resolved.get(key) is None:
                missing.append(key)

        # If agent_fields provided, check each agent's requirements
        if agent_fields:
            for agent_type, fields in agent_fields.items():
                for field in fields:
                    if field not in resolved.values or resolved.values[field] is None:
                        if field not in missing:
                            missing.append(f"{agent_type}.{field}")

        return missing


class AgentInputMatcher:
    """
    Matches workflow inputs to agent required fields.

    This implements the "Auto-Parameter Matching" from Design Doc section 5.4.6.3:
    - Framework matches workflow.inputs.X to agent.fields.X by name
    - If agent field is required but missing → validation error
    - If agent field is optional and missing → agent receives null/default
    """

    def match_inputs(
        self,
        resolved_inputs: ResolvedInputs,
        agent_type: str,
        required_fields: List[str],
        optional_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Match resolved inputs to an agent's field requirements.

        Args:
            resolved_inputs: Fully resolved workflow inputs
            agent_type: Type of agent
            required_fields: List of required field names
            optional_fields: List of optional field names

        Returns:
            Dictionary of matched inputs for the agent

        Raises:
            ResolutionError: If required fields are missing
        """
        matched = {}
        missing = []

        # Match required fields
        for field in required_fields:
            if field in resolved_inputs:
                matched[field] = resolved_inputs[field]
            else:
                missing.append(field)

        if missing:
            raise ResolutionError(
                f"Agent '{agent_type}' requires fields not provided: {missing}"
            )

        # Match optional fields
        if optional_fields:
            for field in optional_fields:
                if field in resolved_inputs:
                    matched[field] = resolved_inputs[field]
                # Optional fields get None/default if not present

        return matched

    def get_all_matching_inputs(
        self,
        resolved_inputs: ResolvedInputs,
        agent_field_names: List[str]
    ) -> Dict[str, Any]:
        """
        Get all inputs that match any agent field names.

        This is a simpler version that just returns any inputs
        that match the given field names.

        Args:
            resolved_inputs: Fully resolved workflow inputs
            agent_field_names: All field names the agent can accept

        Returns:
            Dictionary of matching inputs
        """
        return {
            name: resolved_inputs[name]
            for name in agent_field_names
            if name in resolved_inputs
        }
