"""
Model Router - Complexity-based LLM routing for cost optimization.

Routes each user request to the most appropriate LLM provider based on
task complexity.  A lightweight classifier model scores the request (1-100),
then a set of rules maps the score to a registered LLM provider.

Inspired by the ModelRouterService in Google's gemini-cli, adapted for
OneValet's personal-assistant domain.

Usage::

    from onevalet.llm.router import ModelRouter, RoutingRule

    router = ModelRouter(
        registry=llm_registry,
        classifier_provider="fast",
        rules=[
            RoutingRule(1, 30, "cheap"),
            RoutingRule(31, 70, "fast"),
            RoutingRule(71, 100, "strong"),
        ],
    )

    decision = await router.route(messages)
    client = registry.get(decision.provider)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier system prompt — tuned for personal-assistant use cases
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """\
You are a Task Complexity Classifier for a personal AI assistant that \
manages emails, calendars, travel, smart home, todos, and more.

Analyze the user's request and assign a **complexity score** from 1 to 100.

# Scoring Rubric

**1-20  Trivial / Conversational**
Simple greetings, casual chat, quick factual Q&A, time/date questions, \
simple translations, or single-turn acknowledgements.
- "Hey, good morning!"
- "What time is it in Tokyo?"
- "Thanks, that's all for now."

**21-50  Standard / Single-Agent**
Clear, bounded requests handled by one agent with 1-3 tool calls. \
The intent, target service, and parameters are obvious.
- "Do I have any unread emails?"
- "What's on my calendar tomorrow?"
- "Turn off the living room lights."
- "Add 'buy groceries' to my todo list."
- "Send an email to John saying I'll be 10 minutes late."

**51-80  Complex / Multi-Agent or Analytical**
Requests that span multiple services, require investigation, involve \
ambiguous intent, or need the assistant to plan before acting.
- "Check my emails for any flight confirmations and add them to my calendar."
- "What meetings do I have this week that conflict with my dentist appointment?"
- "Find a good Italian restaurant near my office and send the link to Sarah."
- "Summarize all unread emails from my boss this week."
- "Why didn't my reminder fire yesterday?"

**81-100  Extreme / Strategic or Multi-Step Workflow**
Highly ambiguous requests, complex trip planning, large-scale operations, \
workflows spanning many agents, or requests requiring deep reasoning.
- "Plan a 5-day trip to Japan next month — flights, hotels, and a daily itinerary."
- "Go through my inbox, unsubscribe from all newsletters, and summarize what's left."
- "Reorganize my calendar for next week to free up Wednesday afternoon."
- "Help me prepare for my performance review — pull my recent project emails, \
  meeting notes, and todo completions."

# Rules
- Score based on the ACTUAL work required, not how the user phrases it.
- "What's the best way to remind me about X?" is just a reminder (score ~25), \
  not a strategic question.
- If the conversation history shows the request is a follow-up to an already \
  running task, score LOWER (the context is already loaded).
- When in doubt, round UP — it's better to use a stronger model than to fail.

# Output Format
Respond ONLY with a JSON object. No other text.

{"reasoning": "<brief explanation>", "score": <integer 1-100>}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """Maps a complexity score range to an LLM provider name."""

    min_score: int
    max_score: int
    provider: str

    def matches(self, score: int) -> bool:
        return self.min_score <= score <= self.max_score


@dataclass
class RoutingDecision:
    """The result of a routing decision."""

    provider: str
    score: int
    reasoning: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """Routes requests to different LLM providers based on complexity.

    The router uses a cheap/fast classifier model to score the user's request,
    then picks the appropriate provider from a list of rules.

    Args:
        registry: ``LLMRegistry`` instance with providers already registered.
        classifier_provider: Name of the provider used for classification
            (should be fast and cheap, e.g. ``"fast"``).
        rules: Ordered list of ``RoutingRule``.  First match wins.
        default_provider: Fallback provider when no rule matches or the
            classifier fails.
    """

    def __init__(
        self,
        registry: Any,
        classifier_provider: str = "fast",
        rules: Optional[List[RoutingRule]] = None,
        default_provider: str = "fast",
    ):
        self.registry = registry
        self.classifier_provider = classifier_provider
        self.default_provider = default_provider
        self.rules: List[RoutingRule] = rules or [
            RoutingRule(1, 30, "cheap"),
            RoutingRule(31, 70, "fast"),
            RoutingRule(71, 100, "strong"),
        ]

    # Number of recent conversation turns sent to the classifier for context.
    HISTORY_TURNS = 4

    async def route(
        self,
        messages: List[Dict[str, Any]],
    ) -> RoutingDecision:
        """Classify the request complexity and pick a provider.

        Args:
            messages: The full message list that will be sent to the main LLM.
                Only the last ``HISTORY_TURNS`` user/assistant messages are
                forwarded to the classifier.

        Returns:
            A ``RoutingDecision`` with the chosen provider, score, and reasoning.
        """
        start = time.monotonic()

        try:
            # Build a lightweight context for the classifier: system prompt
            # plus the tail of the conversation (skip tool-call messages).
            recent = [
                m for m in messages
                if m.get("role") in ("user", "assistant")
            ][-self.HISTORY_TURNS:]

            classifier_messages: List[Dict[str, Any]] = [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                *recent,
            ]

            classifier_client = self.registry.get(self.classifier_provider)
            if classifier_client is None:
                raise RuntimeError(
                    f"Classifier provider '{self.classifier_provider}' "
                    f"not found in LLMRegistry"
                )

            resp = await classifier_client.chat_completion(
                messages=classifier_messages,
                config={"temperature": 0, "max_tokens": 150},
            )

            parsed = self._parse_response(resp.content or "")
            score = parsed["score"]
            reasoning = parsed.get("reasoning", "")

            # Match against rules (first match wins)
            provider = self.default_provider
            for rule in self.rules:
                if rule.matches(score):
                    provider = rule.provider
                    break

            # Verify the provider actually exists; fall back if not
            if self.registry.get(provider) is None:
                logger.warning(
                    f"[ModelRouter] Provider '{provider}' not in registry, "
                    f"falling back to '{self.default_provider}'"
                )
                provider = self.default_provider

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                f"[ModelRouter] score={score} -> {provider} "
                f"({elapsed:.0f}ms) | {reasoning}"
            )
            return RoutingDecision(
                provider=provider,
                score=score,
                reasoning=reasoning,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning(
                f"[ModelRouter] Classification failed ({e}), "
                f"falling back to '{self.default_provider}'"
            )
            return RoutingDecision(
                provider=self.default_provider,
                score=-1,
                reasoning=f"fallback: {e}",
                latency_ms=elapsed,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """Extract the JSON object from the classifier response."""
        text = text.strip()

        # Try direct parse first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "score" in obj:
                obj["score"] = int(obj["score"])
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: find first { ... } block (handles markdown fences)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict) and "score" in obj:
                    obj["score"] = int(obj["score"])
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"Could not parse classifier response: {text!r}")
