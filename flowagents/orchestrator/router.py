"""
FlowAgent Message Router - Routes messages to appropriate agents

This module provides:
- MessageRouter: Routes messages to agents based on triggers and context
- Trigger matching for workflows and agents
- Context extraction from messages
"""

import json
import re
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .models import RoutingAction, RoutingReason, RoutingDecision

if TYPE_CHECKING:
    from ..config import AgentRegistry
    from ..standard_agent import StandardAgent
    from ..protocols import LLMClientProtocol

logger = logging.getLogger(__name__)


class MessageRouter:
    """
    Routes messages to appropriate agents or workflows.

    The router uses multiple strategies:
    1. Check for active agents that should receive the message
    2. Match message against workflow triggers
    3. Match message against agent triggers
    4. Use LLM for intelligent routing (if configured)
    5. Route to DefaultAgent (when no other agent matches)

    Usage:
        router = MessageRouter(agent_registry=registry)
        decision = await router.route(
            tenant_id=tenant_id,
            message=message,
            active_agents=active_agents
        )
    """

    def __init__(
        self,
        agent_registry: Optional["AgentRegistry"] = None,
        llm_client: Optional["LLMClientProtocol"] = None,
        enable_llm_routing: bool = False,
        default_agent_type: str = ""
    ):
        """
        Initialize the router.

        Args:
            agent_registry: Registry containing agent and workflow configs
            llm_client: LLM client for intelligent routing
            enable_llm_routing: Whether to use LLM for routing decisions
            default_agent_type: Agent type to use when no other agent matches
        """
        self.agent_registry = agent_registry
        self.llm_client = llm_client
        self.enable_llm_routing = enable_llm_routing and llm_client is not None
        self.default_agent_type = default_agent_type

    async def route(
        self,
        tenant_id: str,
        message: str,
        active_agents: List["StandardAgent"],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a message to an appropriate handler.

        Args:
            tenant_id: Tenant/user identifier
            message: The message to route
            active_agents: List of currently active agents for this tenant
            metadata: Optional metadata from the message

        Returns:
            RoutingDecision indicating how to handle the message
        """
        # Step 1: Check for active agents in non-terminal state
        for agent in active_agents:
            if self._should_route_to_agent(agent, message):
                return RoutingDecision(
                    action=RoutingAction.ROUTE_TO_EXISTING,
                    agent_id=agent.agent_id,
                    confidence=1.0,
                    reason=RoutingReason.ACTIVE_AGENT_FOUND
                )

        # Step 2: Check for workflow triggers
        if self.agent_registry:
            workflow_match = self._match_workflow_trigger(message)
            if workflow_match:
                context = await self._extract_context(message, workflow_match)
                return RoutingDecision(
                    action=RoutingAction.EXECUTE_WORKFLOW,
                    workflow_id=workflow_match,
                    context_hints=context,
                    confidence=0.9,
                    reason=RoutingReason.WORKFLOW_TRIGGER_MATCHED
                )

        # Step 3: Check for agent triggers
        if self.agent_registry:
            agent_match = self._match_agent_trigger(message)
            if agent_match:
                context = await self._extract_context(message, agent_match)
                return RoutingDecision(
                    action=RoutingAction.CREATE_NEW,
                    agent_type=agent_match,
                    context_hints=context,
                    confidence=0.8,
                    reason=RoutingReason.AGENT_TRIGGER_MATCHED
                )

        # Step 4: Try LLM-based routing
        if self.enable_llm_routing:
            llm_decision = await self._route_with_llm(message)
            if llm_decision and llm_decision.action != RoutingAction.ROUTE_TO_DEFAULT:
                return llm_decision

        # Step 5: Route to DefaultAgent
        context = await self._extract_context(message, self.default_agent_type)
        return RoutingDecision(
            action=RoutingAction.ROUTE_TO_DEFAULT,
            agent_type=self.default_agent_type,
            context_hints=context,
            confidence=0.0,
            reason=RoutingReason.DEFAULT_FALLBACK
        )

    def _should_route_to_agent(
        self,
        agent: "StandardAgent",
        message: str
    ) -> bool:
        """
        Check if message should be routed to this agent.

        Routes to agent if it's not in a terminal state.
        """
        from ..result import AgentStatus
        return agent.status not in AgentStatus.terminal_states()

    def _match_workflow_trigger(self, message: str) -> Optional[str]:
        """
        Match message against workflow triggers.

        Returns:
            Workflow ID if matched, None otherwise
        """
        if not self.agent_registry:
            return None

        # TODO: Implement workflow trigger matching when AgentRegistry supports workflows
        # For now, workflows must be triggered explicitly via routing decision
        return None

    def _match_agent_trigger(self, message: str) -> Optional[str]:
        """
        Match message against agent triggers.

        Returns:
            Agent type name if matched, None otherwise
        """
        if not self.agent_registry:
            return None

        return self.agent_registry.find_agent_by_trigger(message)

    def _matches_trigger(self, message: str, trigger: str) -> bool:
        """
        Check if message matches a trigger pattern.

        Supports:
        - Exact match (case insensitive)
        - Prefix match (trigger at start)
        - Contains match (trigger anywhere in message)
        - Regex patterns (if trigger starts with ^)
        """
        trigger_lower = trigger.lower().strip()

        # Regex pattern
        if trigger.startswith('^') or trigger.endswith('$'):
            try:
                return bool(re.search(trigger, message, re.IGNORECASE))
            except re.error:
                pass

        # Exact match
        if message == trigger_lower:
            return True

        # Prefix match
        if message.startswith(trigger_lower):
            return True

        # Contains match
        if trigger_lower in message:
            return True

        return False

    async def _extract_context(
        self,
        message: str,
        agent_or_workflow: str
    ) -> Dict[str, Any]:
        """
        Extract context hints from the message.

        This can be used to pre-fill fields or provide initial context
        to the agent/workflow.
        """
        context: Dict[str, Any] = {
            "original_message": message,
        }

        # If LLM is available, use it to extract structured data
        if self.enable_llm_routing and self.llm_client:
            try:
                extracted = await self._extract_with_llm(message, agent_or_workflow)
                if extracted:
                    context.update(extracted)
            except Exception as e:
                logger.warning(f"Failed to extract context with LLM: {e}")

        return context

    async def _route_with_llm(self, message: str) -> Optional[RoutingDecision]:
        """
        Use LLM to decide routing when no trigger matches.

        Override this method to implement LLM-based routing.
        """
        # Default: no LLM routing. Subclasses implement their own logic.
        return None

    async def _extract_with_llm(
        self,
        message: str,
        agent_or_workflow: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract structured data from message.

        Override this method to implement LLM-based context extraction.
        """
        # Default: no LLM extraction. Subclasses implement their own logic.
        return None


class TriggerMatcher:
    """
    Utility class for matching triggers.

    Supports multiple matching strategies:
    - exact: Exact string match
    - prefix: Message starts with trigger
    - contains: Trigger appears anywhere in message
    - regex: Regular expression match
    - semantic: Semantic similarity (requires embeddings)
    """

    def __init__(
        self,
        strategy: str = "contains",
        case_sensitive: bool = False
    ):
        self.strategy = strategy
        self.case_sensitive = case_sensitive

    def matches(self, message: str, trigger: str) -> bool:
        """Check if message matches trigger"""
        if not self.case_sensitive:
            message = message.lower()
            trigger = trigger.lower()

        if self.strategy == "exact":
            return message == trigger
        elif self.strategy == "prefix":
            return message.startswith(trigger)
        elif self.strategy == "contains":
            return trigger in message
        elif self.strategy == "regex":
            try:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(trigger, message, flags))
            except re.error:
                return False

        return False

    def find_match(
        self,
        message: str,
        triggers: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        Find the first matching trigger from a dict of name -> triggers.

        Returns the name of the matching entry, or None.
        """
        for name, trigger_list in triggers.items():
            for trigger in trigger_list:
                if self.matches(message, trigger):
                    return name
        return None
