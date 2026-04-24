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
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


MAX_EPISODES_PER_WEEK = 5
MAX_FACTS_PER_WEEK = 8


LLMCall = Callable[[str, str], Awaitable[str]]


SYSTEM_PROMPT = """\
You are Koi's weekly memory reflector. You read a week of aggregated
activity logs about one user and distill them into structured long-term
memory. You NEVER invent details not present in the input.

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
         "namespace": "preference" | "routine" | "relationship" | "health" | "project",
         "fact_key": "snake_case_key",
         "value": <any JSON>,
         "summary": "human-readable sentence",
         "confidence": 0.0..1.0,
         "how_to_apply": "tell the assistant how this should change behavior"
       }
     ]
   }
3. If the week is genuinely uneventful, return empty lists and
   highlight "Quiet week with no notable events."
4. Only mark an episode important (>=4) when the user explicitly engaged
   with it or it caused a change in state (routine break, milestone).
5. Do not repeat facts that are already trivial (time/weekday).
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

    user_prompt = _build_user_prompt(user_id, week_start, week_end, daily_episodes)

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


def _build_user_prompt(user_id: str, start: date, end: date, rows: List[Dict[str, Any]]) -> str:
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
    text = (
        f"Week {r.week_start.isoformat()} to {r.week_end.isoformat()}: "
        f"{r.highlight} "
        f"(mood trend: {r.mood_trend}; topics: {', '.join(r.top_topics) if r.top_topics else 'none'})"
    )
    extras = {
        "week_start": r.week_start.isoformat(),
        "week_end": r.week_end.isoformat(),
        "mood_trend": r.mood_trend,
        "top_topics": r.top_topics,
        "fact_proposal_count": len(r.fact_proposals),
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
    out: List[Dict[str, Any]] = []
    valid_ns = {"preference", "routine", "relationship", "health", "project"}
    for f in raw_facts or []:
        if not isinstance(f, dict):
            continue
        ns = str(f.get("namespace", ""))
        if ns not in valid_ns:
            continue
        key = str(f.get("fact_key", "")).strip()
        if not key:
            continue
        out.append(
            {
                "operation": "upsert",
                "namespace": ns,
                "fact_key": key,
                "value": f.get("value"),
                "summary": str(f.get("summary", ""))[:300],
                "confidence": _clamp_float(f.get("confidence", 0.5), 0.0, 1.0),
                "source_type": "weekly_reflection",
                "how_to_apply": str(f.get("how_to_apply", ""))[:400],
                "why": "Inferred by weekly reflector from aggregated activity.",
            }
        )
    return out


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
