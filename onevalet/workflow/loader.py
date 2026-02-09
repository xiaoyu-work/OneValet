"""
OneValet Workflow Loader - Load and validate workflow templates from YAML

This module handles:
1. Loading workflow templates from YAML files
2. Parsing the simplified 4-keyword syntax (run, parallel, then, stages)
3. Validating workflow structure
4. Registering workflows for trigger matching
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

from .models import (
    WorkflowType,
    WorkflowTemplate,
    StageDefinition,
)


class WorkflowLoadError(Exception):
    """Raised when a workflow fails to load"""
    pass


class WorkflowValidationError(Exception):
    """Raised when a workflow fails validation"""
    pass


class WorkflowLoader:
    """
    Loads workflow templates from YAML configuration.

    Supports loading from:
    - Single YAML file
    - Directory of YAML files
    - Dictionary (for programmatic creation)

    Example usage:
        loader = WorkflowLoader()
        loader.load_from_file("workflows.yaml")
        loader.load_from_directory("config/workflows/")

        # Get a workflow by ID
        workflow = loader.get("morning_brief")

        # Find workflow matching a trigger
        workflow = loader.match_trigger("good morning")
    """

    def __init__(self):
        self._workflows: Dict[str, WorkflowTemplate] = {}
        self._trigger_index: Dict[str, str] = {}  # trigger pattern -> workflow_id

    def load_from_file(self, file_path: Union[str, Path]) -> List[WorkflowTemplate]:
        """
        Load workflows from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            List of loaded WorkflowTemplate objects

        Raises:
            WorkflowLoadError: If file cannot be read or parsed
            WorkflowValidationError: If workflow definition is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise WorkflowLoadError(f"Workflow file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise WorkflowLoadError(f"Invalid YAML in {file_path}: {e}")

        if not data:
            return []

        return self._load_from_dict(data, source=str(file_path))

    def load_from_directory(self, dir_path: Union[str, Path]) -> List[WorkflowTemplate]:
        """
        Load all workflow files from a directory.

        Loads files with .yaml or .yml extension.

        Args:
            dir_path: Path to directory containing YAML files

        Returns:
            List of all loaded WorkflowTemplate objects
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise WorkflowLoadError(f"Workflow directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise WorkflowLoadError(f"Not a directory: {dir_path}")

        workflows = []
        for file_path in sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml")):
            workflows.extend(self.load_from_file(file_path))

        return workflows

    def load_from_dict(
        self,
        data: Dict[str, Any],
        source: str = "<dict>"
    ) -> List[WorkflowTemplate]:
        """
        Load workflows from a dictionary.

        Args:
            data: Dictionary with 'workflows' key
            source: Source identifier for error messages

        Returns:
            List of loaded WorkflowTemplate objects
        """
        return self._load_from_dict(data, source)

    def _load_from_dict(
        self,
        data: Dict[str, Any],
        source: str = "<dict>"
    ) -> List[WorkflowTemplate]:
        """Internal method to load workflows from parsed YAML data"""
        workflows_data = data.get("workflows", {})

        if not workflows_data:
            return []

        loaded = []
        for workflow_id, workflow_data in workflows_data.items():
            try:
                workflow = self._parse_workflow(workflow_id, workflow_data)
                self._register_workflow(workflow)
                loaded.append(workflow)
            except Exception as e:
                raise WorkflowValidationError(
                    f"Error loading workflow '{workflow_id}' from {source}: {e}"
                )

        return loaded

    def _parse_workflow(
        self,
        workflow_id: str,
        data: Dict[str, Any]
    ) -> WorkflowTemplate:
        """Parse a single workflow definition into WorkflowTemplate"""

        # Parse workflow type
        type_str = data.get("type", "interactive")
        try:
            workflow_type = WorkflowType(type_str)
        except ValueError:
            raise WorkflowValidationError(
                f"Invalid workflow type: {type_str}. "
                f"Must be one of: {[t.value for t in WorkflowType]}"
            )

        # Parse triggers (for interactive workflows)
        triggers = data.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]

        # Parse trigger_config (for event-triggered workflows)
        trigger_config = data.get("trigger_config")

        # Parse schedule (for scheduled workflows)
        schedule = data.get("schedule")

        # Parse inputs
        inputs = data.get("inputs", {})

        # Parse execution definition - only 4 keywords
        run_agents = data.get("run")
        parallel_agents = data.get("parallel")
        then_agent = data.get("then")
        stages_data = data.get("stages")

        # Normalize agent lists
        if run_agents and isinstance(run_agents, str):
            run_agents = [run_agents]
        if parallel_agents and isinstance(parallel_agents, str):
            parallel_agents = [parallel_agents]

        # Parse stages
        stages = None
        if stages_data:
            stages = self._parse_stages(stages_data)

        # Create workflow template
        workflow = WorkflowTemplate(
            id=workflow_id,
            description=data.get("description", ""),
            type=workflow_type,
            triggers=triggers,
            trigger_config=trigger_config,
            schedule=schedule,
            inputs=inputs,
            run=run_agents,
            parallel=parallel_agents,
            then=then_agent,
            stages=stages,
            timeout_seconds=data.get("timeout_seconds", 3600),
            retry_on_failure=data.get("retry_on_failure", False),
            max_retries=data.get("max_retries", 3),
            tags=data.get("tags", []),
        )

        # Validate
        errors = workflow.validate()
        if errors:
            raise WorkflowValidationError(
                f"Workflow '{workflow_id}' validation failed: {'; '.join(errors)}"
            )

        return workflow

    def _parse_stages(self, stages_data: List[Dict[str, Any]]) -> List[StageDefinition]:
        """Parse stages definition"""
        stages = []

        for i, stage_data in enumerate(stages_data):
            run_agents = stage_data.get("run")
            parallel_agents = stage_data.get("parallel")
            then_agent = stage_data.get("then")
            condition = stage_data.get("condition")

            # Normalize agent lists
            if run_agents and isinstance(run_agents, str):
                run_agents = [run_agents]
            if parallel_agents and isinstance(parallel_agents, str):
                parallel_agents = [parallel_agents]

            # Validate stage has at least run or parallel
            if not run_agents and not parallel_agents:
                raise WorkflowValidationError(
                    f"Stage {i} must have either 'run' or 'parallel' defined"
                )

            stages.append(StageDefinition(
                run=run_agents,
                parallel=parallel_agents,
                then=then_agent,
                condition=condition
            ))

        return stages

    def _register_workflow(self, workflow: WorkflowTemplate) -> None:
        """Register a workflow and index its triggers"""
        # Store workflow
        self._workflows[workflow.id] = workflow

        # Index triggers for interactive workflows
        for trigger in workflow.triggers:
            # Normalize trigger for matching
            normalized = self._normalize_trigger(trigger)
            self._trigger_index[normalized] = workflow.id

    def _normalize_trigger(self, trigger: str) -> str:
        """Normalize a trigger string for matching"""
        return trigger.lower().strip()

    def get(self, workflow_id: str) -> Optional[WorkflowTemplate]:
        """Get a workflow by ID"""
        return self._workflows.get(workflow_id)

    def get_all(self) -> List[WorkflowTemplate]:
        """Get all loaded workflows"""
        return list(self._workflows.values())

    def match_trigger(self, message: str) -> Optional[WorkflowTemplate]:
        """
        Find a workflow that matches the given message trigger.

        Uses prefix matching: if any registered trigger is a prefix of the message,
        the corresponding workflow is returned.

        Args:
            message: User message to match

        Returns:
            Matching WorkflowTemplate or None
        """
        normalized_message = message.lower().strip()

        # Exact match first
        if normalized_message in self._trigger_index:
            workflow_id = self._trigger_index[normalized_message]
            return self._workflows.get(workflow_id)

        # Prefix match
        for trigger, workflow_id in self._trigger_index.items():
            if normalized_message.startswith(trigger):
                return self._workflows.get(workflow_id)

        return None

    def get_by_type(self, workflow_type: WorkflowType) -> List[WorkflowTemplate]:
        """Get all workflows of a specific type"""
        return [w for w in self._workflows.values() if w.type == workflow_type]

    def get_scheduled_workflows(self) -> List[WorkflowTemplate]:
        """Get all scheduled workflows"""
        return self.get_by_type(WorkflowType.SCHEDULED)

    def get_event_triggered_workflows(
        self,
        event_type: Optional[str] = None
    ) -> List[WorkflowTemplate]:
        """
        Get event-triggered workflows.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of matching workflows
        """
        workflows = self.get_by_type(WorkflowType.EVENT_TRIGGERED)

        if event_type:
            workflows = [
                w for w in workflows
                if w.trigger_config and w.trigger_config.get("event_type") == event_type
            ]

        return workflows

    def get_interactive_workflows(self) -> List[WorkflowTemplate]:
        """Get all interactive workflows"""
        return self.get_by_type(WorkflowType.INTERACTIVE)

    def unregister(self, workflow_id: str) -> bool:
        """
        Unregister a workflow.

        Args:
            workflow_id: ID of workflow to remove

        Returns:
            True if workflow was found and removed
        """
        if workflow_id not in self._workflows:
            return False

        workflow = self._workflows.pop(workflow_id)

        # Remove from trigger index
        for trigger in workflow.triggers:
            normalized = self._normalize_trigger(trigger)
            if normalized in self._trigger_index:
                del self._trigger_index[normalized]

        return True

    def clear(self) -> None:
        """Clear all loaded workflows"""
        self._workflows.clear()
        self._trigger_index.clear()

    def __len__(self) -> int:
        return len(self._workflows)

    def __contains__(self, workflow_id: str) -> bool:
        return workflow_id in self._workflows

    def __iter__(self):
        return iter(self._workflows.values())
