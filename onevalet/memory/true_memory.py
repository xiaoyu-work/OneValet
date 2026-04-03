"""Helpers for extracting canonical true-memory proposals from chat turns.

True Memory proposals are structured facts that the orchestrator emits for
an app-owned canonical fact store.  The app backend decides whether to
promote them to ``confirmed``, leave as ``candidate``, or discard.

The orchestrator never writes directly to the canonical store — it only
produces proposals that travel in ``AgentResult.metadata["true_memory_proposals"]``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cheap gate patterns — avoid an LLM call on every turn
# ---------------------------------------------------------------------------

_CANDIDATE_PATTERNS = (
    # Durable facts
    r"\bremember\b",
    r"\bfor future reference\b",
    r"\bkeep in mind\b",
    r"\bmy name is\b",
    r"\bcall me\b",
    r"\bi (?:am|work|live|prefer|like|love|hate|dislike|always|never)\b",
    r"\bmy (?:birthday|email|phone|favorite)\b",
    # Feedback signals — user correcting or confirming AI behavior
    r"\bdon'?t do that\b",
    r"\bstop (?:doing|asking|saying)\b",
    r"\bnext time\b",
    r"\bplease (?:always|never|stop|don'?t)\b",
    r"\bi (?:don'?t )?like (?:it )?when you\b",
    r"\bthat'?s (?:perfect|exactly|right|correct|great|wrong|not what)\b",
    r"\byes,? (?:exactly|perfect|that'?s? (?:it|right))\b",
    r"\bkeep doing\b",
    r"\bdon'?t (?:ask|bother|need to)\b",
)

_TASK_PREFIXES = (
    "can you ",
    "could you ",
    "please ",
    "what ",
    "when ",
    "where ",
    "how ",
    "find ",
    "book ",
    "send ",
    "show ",
    "check ",
)

# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_PROPOSAL_SYSTEM_PROMPT = """
You extract durable user facts AND behavioral feedback for a canonical true-memory store.

Return JSON only in this exact shape:
{
  "should_store": true,
  "proposals": [
    {
      "operation": "upsert",
      "namespace": "preference",
      "fact_key": "flight_seat",
      "value": {"seat": "aisle"},
      "summary": "User prefers aisle seats on flights.",
      "confidence": 0.97,
      "source_type": "user_direct",
      "reason": "The user directly stated a stable travel preference.",
      "why": "User finds aisle seats more comfortable on long flights.",
      "how_to_apply": "When booking flights, default to aisle seat selection.",
      "evidence": "Remember that I prefer aisle seats when I fly."
    }
  ]
}

## Two categories of memory

### 1. Durable facts (namespace: identity, work, relationship, lifestyle, travel, preference)
Personal facts, biographical info, and stable preferences.
- Store when the user states something about themselves that will remain true across sessions.
- source_type: "user_direct" or "user_correction".

### 2. Behavioral feedback (namespace: feedback)
How the user wants the AI to behave — both corrections AND confirmations.
- Corrections: "don't do that", "stop asking me to confirm", "next time just do it"
- Confirmations: "yes exactly", "perfect, keep doing that", "that's the right approach"
- Store BOTH corrections and confirmations. Corrections prevent repeating mistakes;
  confirmations prevent drifting away from approaches the user already validated.
- source_type: "user_correction" for corrections, "user_confirmation" for confirmations.
- ALWAYS include "why" (the reason behind the feedback) and "how_to_apply" (when this guidance kicks in).

## Rules
- Store only things that will matter in future turns.
- Prefer direct user statements over inferences.
- Do NOT store transient tasks, temporary plans, questions, or assistant claims.
- If the user corrects or replaces a prior fact/feedback, reuse the same namespace/fact_key with operation "upsert".
- Use operation "revoke" only when the user explicitly invalidates something without replacing it.
- Keep fact_key stable and snake_case.
- Keep summary to one sentence starting with "User ...".
- "why" should explain the user's reason or motivation (one sentence).
- "how_to_apply" should describe when/where this memory should influence behavior (one sentence).
- If nothing qualifies, return {"should_store": false, "proposals": []}.
""".strip()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def looks_like_true_memory_candidate(message: str) -> bool:
    """Cheap gate so we only run proposal extraction on likely durable facts."""
    normalized = (message or "").strip().lower()
    if len(normalized) < 8:
        return False
    if normalized.endswith("?") and "remember" not in normalized:
        return False
    if any(normalized.startswith(p) for p in _TASK_PREFIXES) and "remember" not in normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _CANDIDATE_PATTERNS)


def format_true_memory_for_prompt(
    true_memory: Optional[List[Dict[str, Any]]],
    *,
    limit: int = 20,
) -> str:
    """Format canonical app-owned memory facts for prompt injection.

    Output is pure natural-language text — no internal identifiers (namespace,
    fact_key) are exposed.  Feedback memories include Why/How-to-apply lines
    so the LLM can reason about edge cases rather than blindly following rules.
    """
    if not true_memory:
        return ""

    lines: List[str] = []
    for fact in true_memory[:limit]:
        if not isinstance(fact, dict):
            continue
        summary = str(fact.get("summary") or "").strip()
        if not summary:
            continue

        line = f"- {summary}"

        # Append Why + How so the LLM can generalize
        why = str(fact.get("why") or "").strip()
        how = str(fact.get("how_to_apply") or "").strip()
        if why:
            line += f" Why: {why}"
        if how:
            line += f" Apply: {how}"

        lines.append(line)

    return "\n".join(lines)


async def extract_true_memory_proposals(
    llm_client: Any,
    *,
    user_message: str,
    assistant_response: str = "",
    existing_true_memory: Optional[List[Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract structured canonical-memory proposals from a completed turn.

    Returns a list of proposal dicts ready to be placed in
    ``AgentResult.metadata["true_memory_proposals"]``.
    """
    if not looks_like_true_memory_candidate(user_message):
        return []

    payload = {
        "user_message": user_message,
        "assistant_response": assistant_response,
        "existing_true_memory": existing_true_memory or [],
        "user_profile": user_profile or {},
    }

    try:
        response = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": _PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            config={"temperature": 0.0, "max_tokens": 700},
        )
        parsed = _parse_json_payload(response.content or "")
        raw_proposals = parsed.get("proposals", []) if isinstance(parsed, dict) else []

        proposals: List[Dict[str, Any]] = []
        for item in raw_proposals:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_proposal(item, fallback_evidence=user_message)
            if normalized:
                proposals.append(normalized)

        if proposals:
            return _dedupe_proposals(proposals)
    except Exception as exc:
        logger.warning("True-memory extraction failed, falling back to rules: %s", exc)

    return _fallback_extract(user_message)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_json_payload(text: str) -> Dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _normalize_proposal(
    item: Dict[str, Any],
    *,
    fallback_evidence: str,
) -> Optional[Dict[str, Any]]:
    namespace = _normalize_slug(str(item.get("namespace") or "preference"))
    fact_key = _normalize_slug(str(item.get("fact_key") or "user_fact"))
    summary = " ".join(str(item.get("summary") or "").split()).strip()
    if not namespace or not fact_key or not summary:
        return None

    operation = _normalize_slug(str(item.get("operation") or "upsert"))
    if operation not in {"upsert", "revoke"}:
        operation = "upsert"
    source_type = _normalize_slug(str(item.get("source_type") or "user_direct"))

    try:
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "operation": operation,
        "namespace": namespace,
        "fact_key": fact_key,
        "value": item.get("value"),
        "summary": summary,
        "confidence": round(confidence, 4),
        "source_type": source_type,
        "reason": item.get("reason"),
        "why": " ".join(str(item.get("why") or "").split()).strip() or None,
        "how_to_apply": " ".join(str(item.get("how_to_apply") or "").split()).strip() or None,
        "evidence": item.get("evidence") or fallback_evidence,
    }


def _dedupe_proposals(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for proposal in proposals:
        key = (
            proposal.get("operation", "upsert"),
            proposal.get("namespace", ""),
            proposal.get("fact_key", ""),
        )
        current = deduped.get(key)
        if current is None or proposal.get("confidence", 0.0) >= current.get("confidence", 0.0):
            deduped[key] = proposal
    return list(deduped.values())


def _fallback_extract(user_message: str) -> List[Dict[str, Any]]:
    """Rule-based fallback when LLM extraction fails or is unavailable."""
    text = (user_message or "").strip()
    lowered = text.lower()
    proposals: List[Dict[str, Any]] = []

    name_match = re.search(
        r"\b(?:my name is|call me)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})",
        text,
        re.IGNORECASE,
    )
    if name_match:
        captured = name_match.group(1).strip()
        fact_key = "preferred_name" if "call me" in lowered else "full_name"
        proposals.append({
            "operation": "upsert",
            "namespace": "identity",
            "fact_key": fact_key,
            "value": captured,
            "summary": f"User's {'preferred name' if fact_key == 'preferred_name' else 'name'} is {captured}.",
            "confidence": 0.88,
            "source_type": "user_direct",
            "reason": "Matched a direct self-identification pattern.",
            "evidence": text,
        })

    seat_match = re.search(r"\b(?:prefer|like)\s+(aisle|window)\s+seats?\b", lowered)
    if seat_match:
        seat = seat_match.group(1)
        proposals.append({
            "operation": "upsert",
            "namespace": "travel",
            "fact_key": "flight_seat",
            "value": {"seat": seat},
            "summary": f"User prefers {seat} seats on flights.",
            "confidence": 0.9,
            "source_type": "user_direct",
            "reason": "Matched an explicit travel seating preference.",
            "evidence": text,
        })

    location_match = re.search(r"\bi live in\s+([^.!?]+)", text, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip()
        proposals.append({
            "operation": "upsert",
            "namespace": "identity",
            "fact_key": "home_location",
            "value": {"text": location},
            "summary": f"User lives in {location}.",
            "confidence": 0.84,
            "source_type": "user_direct",
            "reason": "Matched an explicit home location statement.",
            "evidence": text,
        })

    # Feedback: user correcting AI behavior
    stop_match = re.search(
        r"\b(?:stop|don'?t|quit|please don'?t)\s+(doing|asking|saying|sending|checking)\s+(.{5,60})",
        lowered,
    )
    if stop_match:
        action = stop_match.group(1)
        detail = stop_match.group(2).rstrip(" .,!").strip()
        fact_key = _normalize_slug(f"stop_{action}_{detail[:30]}")
        proposals.append({
            "operation": "upsert",
            "namespace": "feedback",
            "fact_key": fact_key,
            "value": {"correction": text},
            "summary": f"User wants assistant to stop {action} {detail}.",
            "confidence": 0.85,
            "source_type": "user_correction",
            "reason": "User explicitly corrected AI behavior.",
            "why": f"User said to stop {action} {detail}.",
            "how_to_apply": f"Avoid {action} {detail} in future interactions.",
            "evidence": text,
        })

    # Feedback: user confirming AI behavior
    confirm_match = re.search(
        r"\b(?:yes,? ?exactly|perfect|that'?s (?:right|correct|it|great)|keep doing (?:that|this))\b",
        lowered,
    )
    if confirm_match and not stop_match:
        fact_key = _normalize_slug(f"confirmed_{text[:30]}")
        proposals.append({
            "operation": "upsert",
            "namespace": "feedback",
            "fact_key": fact_key,
            "value": {"confirmation": text},
            "summary": f"User confirmed current approach is correct.",
            "confidence": 0.75,
            "source_type": "user_confirmation",
            "reason": "User validated the assistant's approach.",
            "why": "User expressed approval of the current interaction style.",
            "how_to_apply": "Continue using this approach in similar situations.",
            "evidence": text,
        })

    return _dedupe_proposals(proposals)
