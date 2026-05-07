"""WeeklyReflector — the single LLM call that anchors Koi's memory growth.

Cron: Monday 03:00 in user's local tz. Reads the last 7 daily_log episodes
from Momex (written by :mod:`daily_log_aggregator`) and produces:

  * 0-5 episode entries (notable events, milestones, routine breaks),
    each written back to Momex as ``subkind="behavioral_pattern"``.
  * 1 weekly_reflection episode capturing the overall highlight + mood
    trend + top topics.
  * N fact proposals (durable observations) returned via the caller's
    metadata pipe — persisted by the outer agent runner.

Why weekly, not daily:
  1. Mobile conversation is often sparse; a daily reflection would hallucinate.
  2. Weekly aggregation catches patterns ("three nights of bad sleep") that
     daily reflection can't see.
  3. Cost — one LLM call per user per week is predictable.

Prompt philosophy:
  * Feed pre-aggregated daily_log episodes, never raw messages.
  * Force structured JSON output.
  * Clamp output counts to prevent runaway episode growth.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


MAX_EPISODES_PER_WEEK = 5
MAX_FACTS_PER_WEEK = 8
MAX_EXISTING_TRUE_MEMORY = 30


LLMCall = Callable[[str, str], Awaitable[str]]


SYSTEM_PROMPT = """\
You are Koi's weekly memory reflector. You read a week of aggregated
activity logs about one user and run a bounded memory consolidation pass.
You NEVER invent details not present in the input.

This is NOT a full database scan. Only compare the current week against the
provided existing_true_memory candidates. Prefer no proposal over a weak one.

Output requirements:
1. Respond with a SINGLE JSON object. No prose, no code fences.
2. Shape:
   {
     "highlight": "1-2 sentence summary of the week",
     "mood_trend": "improving" | "stable" | "declining" | "mixed",
     "top_topics": ["string", ...],
     "episodes": [
       {
         "local_date": "YYYY-MM-DD",
         "title": "short title (<80 chars)",
         "summary": "2-3 sentences",
         "kind": "event" | "milestone" | "reflection" | "routine_break",
         "mood": "positive" | "neutral" | "negative" | null,
         "location": "string or null",
         "importance": 1..5,
         "entities": ["name", ...]
       }
     ],
      "facts": [
        {
          "operation": "upsert" | "revoke",
          "namespace": "identity" | "work" | "relationship" | "lifestyle" | "travel" | "preference" | "feedback" | "routine" | "health" | "project" | "habit",
          "fact_key": "snake_case_key",
          "value": <any JSON>,
          "summary": "human-readable sentence",
          "confidence": 0.0..1.0,
          "why": "why this should become or change durable memory",
          "how_to_apply": "tell the assistant how this should change behavior"
        }
      ]
   }
3. If the week is genuinely uneventful, return empty lists and
   highlight "Quiet week with no notable events."
4. Only mark an episode important (>=4) when the user explicitly engaged
   with it or it caused a change in state (routine break, milestone).
5. Do not repeat facts that are already trivial (time/weekday).
6. For facts:
   - Emit "upsert" only for stable preferences, routines, recurring patterns,
     explicit corrections, or meaningful project/relationship/health changes.
   - Reuse an existing namespace/fact_key when the week updates or replaces an
     existing memory. This is how memory stays small.
   - Emit "revoke" only when the week clearly invalidates an existing memory
     and there is no replacement value.
   - Do not emit a fact that is already present in existing_true_memory unless
     the value, confidence, why, or how_to_apply should change.
   - Avoid one-off tasks, temporary plans, and noisy observations.
"""


@dataclass
class WeeklyReflection:
    week_start: date
    week_end: date
    highlight: str
    mood_trend: str
    top_topics: List[str]
    episodes_written: int
    fact_proposals: List[Dict[str, Any]]
    raw_response: Dict[str, Any]


async def run_weekly_reflection(
    user_id: str,
    week_end: date,
    llm_call: LLMCall,
    episode_memory,
    *,
    existing_true_memory: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[WeeklyReflection]:
    """Run the full pipeline for a single user.

    Returns the reflection record (already persisted as episodes in Momex)
    or None if the week was empty / LLM failed.

    ``week_end`` is inclusive; the reflector reads days
    [week_end - 6, week_end].
    """
    week_start = week_end - timedelta(days=6)

    daily_episodes = await _fetch_daily_log_episodes(episode_memory, user_id, week_start, week_end)
    if not daily_episodes:
        logger.info("weekly_reflector: no daily_log episodes for %s..%s", week_start, week_end)
        return None

    user_prompt = _build_user_prompt(
        user_id,
        week_start,
        week_end,
        daily_episodes,
        existing_true_memory=existing_true_memory,
    )

    try:
        raw = await llm_call(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error("weekly_reflector LLM call failed: %s", e)
        return None

    try:
        parsed = _parse_response(raw)
    except Exception as e:
        logger.error("weekly_reflector JSON parse failed: %s; raw=%.500s", e, raw)
        return None

    episodes_written = 0
    for ep in parsed.get("episodes", [])[:MAX_EPISODES_PER_WEEK]:
        if await _write_episode(episode_memory, user_id, ep):
            episodes_written += 1

    fact_proposals = _clean_facts(parsed.get("facts", []))[:MAX_FACTS_PER_WEEK]

    reflection = WeeklyReflection(
        week_start=week_start,
        week_end=week_end,
        highlight=str(parsed.get("highlight", ""))[:500],
        mood_trend=str(parsed.get("mood_trend", "stable")),
        top_topics=[str(x)[:60] for x in (parsed.get("top_topics") or [])][:5],
        episodes_written=episodes_written,
        fact_proposals=fact_proposals,
        raw_response=parsed,
    )

    await _write_weekly_summary(episode_memory, user_id, reflection)
    return reflection


# ---------------------------------------------------------------- internals


async def _fetch_daily_log_episodes(
    episode_memory, user_id: str, start: date, end: date
) -> List[Dict[str, Any]]:
    """Pull the last ~7 daily_log episodes via Momex recall."""
    try:
        items = await episode_memory.recall_recent_episodes(
            user_id,
            subkind="daily_log",
            limit=14,
        )
    except Exception as e:
        logger.error("recall daily_log episodes failed: %s", e)
        return []

    # Keep only episodes whose metadata.local_date falls in the window.
    out: List[Dict[str, Any]] = []
    for item in items:
        meta = item.get("metadata") or {}
        ld_raw = meta.get("local_date")
        try:
            ld = date.fromisoformat(ld_raw) if ld_raw else None
        except Exception:
            ld = None
        if ld and (ld < start or ld > end):
            continue
        out.append(
            {
                "local_date": ld.isoformat() if ld else None,
                "text": item.get("text") or "",
                "payload": meta.get("payload") or {},
            }
        )
    out.sort(key=lambda x: x.get("local_date") or "")
    return out


def _build_user_prompt(
    user_id: str,
    start: date,
    end: date,
    rows: List[Dict[str, Any]],
    *,
    existing_true_memory: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    compact = []
    for r in rows:
        p = r.get("payload") or {}
        compact.append(
            {
                "date": r.get("local_date"),
                "text": r.get("text", "")[:500],
                "messages": p.get("messages", {}).get("total", 0),
                "tools": p.get("tools", {}),
                "calendar": [
                    e.get("title") for e in (p.get("calendar", {}) or {}).get("events", [])
                ][:10],
                "reminders_done": (p.get("reminders", {}) or {}).get("completed", [])[:10],
                "health": p.get("health", {}),
                "motion": p.get("motion", {}),
                "state": {
                    k: v
                    for k, v in (p.get("state") or {}).items()
                    if v is not None
                    and k
                    in (
                        "sleep_minutes",
                        "sleep_score",
                        "steps",
                        "activity_minutes",
                        "stress_score",
                        "mood",
                        "primary_location",
                        "flags",
                    )
                },
            }
        )
    return (
        f"Week: {start} to {end}\n"
        f"existing_true_memory candidates (bounded; use keys for updates/revokes):\n"
        f"{json.dumps(_compact_true_memory(existing_true_memory), default=str, indent=2)}\n\n"
        f"Daily activity logs (pre-aggregated, one entry per day):\n"
        f"{json.dumps(compact, default=str, indent=2)}\n\n"
        "Return the JSON specified in the system prompt. Output ONLY the JSON."
    )


def _parse_response(raw: str) -> Dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, s = s.split("\n", 1)
    return json.loads(s)


def _compact_true_memory(
    existing_true_memory: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for item in list(existing_true_memory or [])[:MAX_EXISTING_TRUE_MEMORY]:
        if not isinstance(item, dict):
            continue
        namespace = str(item.get("namespace") or "").strip()
        fact_key = str(item.get("fact_key") or "").strip()
        summary = " ".join(str(item.get("summary") or "").split()).strip()
        if not namespace or not fact_key or not summary:
            continue
        compact.append(
            {
                "namespace": namespace[:60],
                "fact_key": fact_key[:100],
                "summary": summary[:300],
                "value": item.get("value"),
                "confidence": item.get("confidence"),
                "why": str(item.get("why") or "")[:240] or None,
                "how_to_apply": str(item.get("how_to_apply") or "")[:240] or None,
            }
        )
    return compact


async def _write_episode(episode_memory, user_id: str, ep: Dict[str, Any]) -> bool:
    try:
        local_date = date.fromisoformat(ep["local_date"])
    except Exception:
        return False
    title = str(ep.get("title", "")).strip()[:200]
    summary_text = str(ep.get("summary", ""))[:1000]
    if not title or not summary_text:
        return False
    text = f"{title}. {summary_text}"
    extras = {
        "title": title,
        "local_date": local_date.isoformat(),
        "kind_hint": str(ep.get("kind", "event")),
        "mood": ep.get("mood"),
        "location": ep.get("location"),
        "importance": _clamp_int(ep.get("importance", 3), 1, 5),
        "entities": [str(x) for x in (ep.get("entities") or [])][:10],
    }
    try:
        await episode_memory.write_episode(
            tenant_id=user_id,
            summary=text,
            subkind="behavioral_pattern",
            start_ts=datetime.combine(local_date, datetime.min.time(), tzinfo=timezone.utc),
            end_ts=datetime.combine(local_date, datetime.max.time(), tzinfo=timezone.utc),
            source="weekly_reflection",
            extras=extras,
        )
        return True
    except Exception as e:
        logger.error("behavioural_pattern episode write failed: %s", e)
        return False


async def _write_weekly_summary(episode_memory, user_id: str, r: WeeklyReflection) -> None:
    fact_lines = []
    for f in r.fact_proposals[:MAX_FACTS_PER_WEEK]:
        summary = str(f.get("summary") or "").strip()
        if not summary:
            continue
        op = str(f.get("operation") or "upsert")
        ns = str(f.get("namespace") or "")
        key = str(f.get("fact_key") or "")
        fact_lines.append(f"{op} {ns}.{key}: {summary}")

    text = (
        f"Week {r.week_start.isoformat()} to {r.week_end.isoformat()}: "
        f"{r.highlight} "
        f"(mood trend: {r.mood_trend}; topics: {', '.join(r.top_topics) if r.top_topics else 'none'})"
    )
    if fact_lines:
        text += " Memory maintenance proposals: " + " | ".join(fact_lines)
    extras = {
        "week_start": r.week_start.isoformat(),
        "week_end": r.week_end.isoformat(),
        "mood_trend": r.mood_trend,
        "top_topics": r.top_topics,
        "fact_proposal_count": len(r.fact_proposals),
        "fact_proposal_keys": [
            {
                "operation": f.get("operation"),
                "namespace": f.get("namespace"),
                "fact_key": f.get("fact_key"),
            }
            for f in r.fact_proposals[:MAX_FACTS_PER_WEEK]
        ],
    }
    try:
        await episode_memory.write_episode(
            tenant_id=user_id,
            summary=text,
            subkind="weekly_reflection",
            start_ts=datetime.combine(r.week_start, datetime.min.time(), tzinfo=timezone.utc),
            end_ts=datetime.combine(r.week_end, datetime.max.time(), tzinfo=timezone.utc),
            source="weekly_reflection",
            extras=extras,
        )
    except Exception as e:
        logger.error("weekly_reflection episode write failed: %s", e)


def _clean_facts(raw_facts: List[Any]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
    valid_ns = {
        "feedback",
        "habit",
        "health",
        "identity",
        "lifestyle",
        "preference",
        "project",
        "relationship",
        "routine",
        "travel",
        "work",
    }
    for f in raw_facts or []:
        if not isinstance(f, dict):
            continue
        ns = _normalize_slug(str(f.get("namespace", "")))
        if ns not in valid_ns:
            continue
        key = _normalize_slug(str(f.get("fact_key", "")))
        if not key:
            continue
        operation = _normalize_slug(str(f.get("operation") or "upsert"))
        if operation not in {"upsert", "revoke"}:
            operation = "upsert"
        confidence = _clamp_float(f.get("confidence", 0.5), 0.0, 1.0)
        min_confidence = 0.75 if operation == "revoke" else 0.65
        if confidence < min_confidence:
            continue
        summary = " ".join(str(f.get("summary", "")).split()).strip()[:300]
        if not summary:
            continue
        proposal = {
            "operation": operation,
            "namespace": ns,
            "fact_key": key,
            "value": None if operation == "revoke" else f.get("value"),
            "summary": summary,
            "confidence": round(confidence, 4),
            "source_type": "weekly_reflection",
            "reason": "Inferred by weekly reflector from aggregated activity.",
            "why": " ".join(str(f.get("why") or "").split()).strip()[:400]
            or "Inferred by weekly reflector from aggregated activity.",
            "how_to_apply": " ".join(str(f.get("how_to_apply") or "").split()).strip()[:400]
            or None,
            "evidence": str(f.get("evidence") or "")[:500] or None,
        }
        key_tuple = (ns, key)
        current = deduped.get(key_tuple)
        if current is None or proposal["confidence"] >= current.get("confidence", 0.0):
            deduped[key_tuple] = proposal
    return list(deduped.values())


def _normalize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return lo
    return max(lo, min(hi, n))


def _clamp_float(v: Any, lo: float, hi: float) -> float:
    try:
        n = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, n))
