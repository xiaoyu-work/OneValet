"""Intent Analyzer - Classifies user messages into domains and detects multi-intent.

The Intent Analyzer runs as a lightweight LLM call before the ReAct loop,
determining:
1. Whether the message contains single or multiple independent intents
2. Which domain(s) are relevant (communication, productivity, lifestyle, travel)
3. For multi-intent: sub-task decomposition with dependency ordering
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A single sub-task extracted from a multi-intent message."""

    id: int
    description: str
    domain: str
    depends_on: List[int] = field(default_factory=list)


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""

    intent_type: str  # "single" or "multi"
    domains: List[str]  # Domains needed (e.g., ["communication", "productivity"])
    sub_tasks: List[SubTask] = field(default_factory=list)
    raw_message: str = ""  # Original user message


VALID_DOMAINS = {"communication", "productivity", "lifestyle", "travel", "general"}

INTENT_ANALYZER_SYSTEM_PROMPT = """\
You are an intent classifier for a personal AI assistant.

Given a user message, determine:
1. Whether it contains a single intent or multiple independent intents
2. Which domain(s) are needed

Domains:
- communication: email, slack, discord, twitter/X, linkedin messaging
- productivity: calendar, todo/reminders, briefing, notion, google workspace, cloud storage, github, cron/scheduling
- lifestyle: expenses/budget, smart home, package tracking, spotify/music, youtube, image generation, important dates
- travel: trip planning, maps/directions/navigation, nearby places, air quality
- general: greeting, chitchat, creative writing, general knowledge (no agent needed)

Rules:
- For conditional logic ("if X then Y", "check X and based on that do Y"), classify as SINGLE intent — the execution engine handles conditions natively.
- For multiple items targeting the same agent ("lunch $15, uber $12, coffee $5"), classify as SINGLE intent.
- Only classify as MULTI when there are genuinely independent tasks that require DIFFERENT agents.
- When classifying as MULTI, identify dependencies: if task B needs the result of task A, mark B as depending on A.
- Each sub-task's description should be a complete, self-contained instruction.

Return strict JSON only:
{
  "intent_type": "single" | "multi",
  "domains": ["domain1", ...],
  "sub_tasks": []
}

For single intent, sub_tasks must be an empty array.
For multi intent, sub_tasks must have at least 2 entries:
{
  "intent_type": "multi",
  "domains": ["domain1", "domain2"],
  "sub_tasks": [
    {"id": 1, "description": "...", "domain": "...", "depends_on": []},
    {"id": 2, "description": "...", "domain": "...", "depends_on": [1]}
  ]
}
"""


class IntentAnalyzer:
    """Lightweight LLM-based intent analyzer.

    Classifies user messages into domains and detects multi-intent requests.
    Falls back gracefully to all-domains on any failure.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    async def analyze(self, user_message: str) -> IntentAnalysis:
        """Analyze user message intent and domain classification.

        Uses a single lightweight LLM call (~200 tokens).
        Falls back to single-intent with all domains on failure.
        """
        messages = [
            {"role": "system", "content": INTENT_ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                config={"temperature": 0.0, "max_tokens": 400},
            )
            result_json = self._extract_json(response.content or "")
            if not result_json:
                logger.warning("[IntentAnalyzer] Failed to parse JSON from response")
                return self._fallback(user_message)
            return self._parse_result(result_json, user_message)
        except Exception as e:
            logger.warning(f"[IntentAnalyzer] LLM call failed: {e}")
            return self._fallback(user_message)

    def _fallback(self, user_message: str) -> IntentAnalysis:
        """Safe fallback: single intent with all domains."""
        return IntentAnalysis(
            intent_type="single",
            domains=["all"],
            sub_tasks=[],
            raw_message=user_message,
        )

    def _parse_result(self, data: dict, user_message: str) -> IntentAnalysis:
        """Parse and validate LLM response into IntentAnalysis."""
        intent_type = data.get("intent_type", "single")
        domains = data.get("domains", ["all"])

        # Validate domains
        domains = [d for d in domains if d in VALID_DOMAINS] or ["all"]

        sub_tasks: List[SubTask] = []
        if intent_type == "multi":
            for st in data.get("sub_tasks", []):
                domain = st.get("domain", "general")
                if domain not in VALID_DOMAINS:
                    domain = "general"
                sub_tasks.append(SubTask(
                    id=st.get("id", 0),
                    description=st.get("description", ""),
                    domain=domain,
                    depends_on=st.get("depends_on", []),
                ))
            # Downgrade to single if fewer than 2 sub-tasks
            if len(sub_tasks) < 2:
                intent_type = "single"
                sub_tasks = []

        return IntentAnalysis(
            intent_type=intent_type,
            domains=domains,
            sub_tasks=sub_tasks,
            raw_message=user_message,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from model output."""
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
