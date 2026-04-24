"""WeeklyReflector — the single LLM call that anchors Koi's memory growth.

Run cron: Monday 03:00 in user's local tz.  Consumes the previous 7 days'
``daily_logs`` rows and produces:

  * 0-5 ``episodes`` (notable events, milestones, routine breaks)
  * 1 ``weekly_reflections`` row (highlight, mood trend, top topics, facts)
  * N ``true_memory_proposals`` (observations about the user's evolving
    preferences / routines / relationships)

We do this weekly rather than daily for three reasons:
  1. Mobile conversation is often sparse; a daily reflection would hallucinate.
  2. Weekly aggregation catches patterns ("three nights of bad sleep") that
     daily reflection can't see.
  3. Cost — one LLM call per user per week is predictable.

Prompt philosophy:
  * Feed pre-aggregated data, not raw messages — the daily_logs already
    summarize everything.
  * Force structured JSON output.  No prose.
  * Clamp output counts to prevent runaway episode growth.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from koa.memory.lifecycle.episode_store import EpisodeDraft, create_episode

logger = logging.getLogger(__name__)


# Upper bound so a pathological LLM response doesn't spam the episodes table.
MAX_EPISODES_PER_WEEK = 5
MAX_FACTS_PER_WEEK = 8


# ``LLMCall`` takes (system_prompt, user_prompt) and returns a JSON string.
LLMCall = Callable[[str, str], Awaitable[str]]
Embedder = Callable[[str], Awaitable[Optional[List[float]]]]


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
     "top_topics": ["string", ...],  // max 5
     "episodes": [                   // max 5 notable events
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
     "facts": [                      // max 8 durable observations
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
    episodes_created: List[str]        # episode ids
    fact_proposals: List[Dict[str, Any]]
    raw_response: Dict[str, Any]


async def run_weekly_reflection(
    db,
    user_id: str,
    week_end: date,
    llm_call: LLMCall,
    embedder: Optional[Embedder] = None,
) -> Optional[WeeklyReflection]:
    """Run the full pipeline for a single user; returns the reflection
    record (already persisted) or None if the week was empty or LLM failed.

    ``week_end`` is inclusive; the reflector reads days
    [week_end - 6, week_end].
    """
    week_start = week_end - timedelta(days=6)
    daily_rows = await _fetch_daily_logs(db, user_id, week_start, week_end)
    if not daily_rows:
        logger.info("weekly_reflector: no daily_logs for %s..%s", week_start, week_end)
        return None

    user_prompt = _build_user_prompt(user_id, week_start, week_end, daily_rows)

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

    # Persist episodes (idempotent via (user_id, local_date, title)).
    episode_ids: List[str] = []
    for ep in parsed.get("episodes", [])[:MAX_EPISODES_PER_WEEK]:
        draft = _draft_from_payload(ep)
        if draft is None:
            continue
        eid = await create_episode(db, user_id, draft, embedder=embedder)
        if eid:
            episode_ids.append(eid)

    fact_proposals = _clean_facts(parsed.get("facts", []))[:MAX_FACTS_PER_WEEK]

    reflection = WeeklyReflection(
        week_start=week_start,
        week_end=week_end,
        highlight=str(parsed.get("highlight", ""))[:500],
        mood_trend=str(parsed.get("mood_trend", "stable")),
        top_topics=[str(x)[:60] for x in (parsed.get("top_topics") or [])][:5],
        episodes_created=episode_ids,
        fact_proposals=fact_proposals,
        raw_response=parsed,
    )

    await _persist_reflection(db, user_id, reflection)
    return reflection


async def _fetch_daily_logs(db, user_id: str, start: date, end: date) -> List[Dict[str, Any]]:
    try:
        rows = await db.fetch(
            """SELECT local_date, payload
               FROM daily_logs
               WHERE user_id = $1 AND local_date >= $2 AND local_date <= $3
               ORDER BY local_date""",
            user_id, start, end,
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            payload = r["payload"]
            if isinstance(payload, str):
                try: payload = json.loads(payload)
                except Exception: payload = {}
            out.append({"local_date": r["local_date"].isoformat(), "payload": payload or {}})
        return out
    except Exception as e:
        logger.error("_fetch_daily_logs failed: %s", e)
        return []


def _build_user_prompt(user_id: str, start: date, end: date, rows: List[Dict[str, Any]]) -> str:
    """Compact the week into a single prompt; strip redundant keys."""
    compact = []
    for r in rows:
        p = r["payload"] or {}
        compact.append({
            "date": r["local_date"],
            "messages": p.get("messages", {}).get("total", 0),
            "tools": p.get("tools", {}),
            "calendar": [e.get("title") for e in p.get("calendar", {}).get("events", [])][:10],
            "reminders_done": p.get("reminders", {}).get("completed", [])[:10],
            "state": {
                k: v for k, v in (p.get("state") or {}).items()
                if v is not None and k in (
                    "sleep_minutes", "sleep_score", "steps", "activity_minutes",
                    "stress_score", "mood", "primary_location", "flags",
                )
            },
        })
    return (
        f"Week: {start} to {end}\n"
        f"Daily activity logs (pre-aggregated):\n"
        f"{json.dumps(compact, default=str, indent=2)}\n\n"
        "Return the JSON specified in the system prompt. Output ONLY the JSON."
    )


def _parse_response(raw: str) -> Dict[str, Any]:
    # Strip code fences if the LLM disobeyed instructions.
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # Drop optional leading language tag
        if "\n" in s:
            _, s = s.split("\n", 1)
    return json.loads(s)


def _draft_from_payload(ep: Dict[str, Any]) -> Optional[EpisodeDraft]:
    try:
        local_date = date.fromisoformat(ep["local_date"])
    except Exception:
        return None
    title = str(ep.get("title", "")).strip()[:200]
    if not title:
        return None
    return EpisodeDraft(
        title=title,
        summary=str(ep.get("summary", ""))[:1000],
        local_date=local_date,
        kind=str(ep.get("kind", "event")),
        mood=ep.get("mood"),
        location=ep.get("location"),
        importance=_clamp_int(ep.get("importance", 3), 1, 5),
        source="weekly_reflection",
        entities=[str(x) for x in (ep.get("entities") or [])][:10],
    )


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
        out.append({
            "operation": "upsert",
            "namespace": ns,
            "fact_key": key,
            "value": f.get("value"),
            "summary": str(f.get("summary", ""))[:300],
            "confidence": _clamp_float(f.get("confidence", 0.5), 0.0, 1.0),
            "source_type": "weekly_reflection",
            "how_to_apply": str(f.get("how_to_apply", ""))[:400],
            "why": "Inferred by weekly reflector from aggregated activity.",
        })
    return out


async def _persist_reflection(db, user_id: str, r: WeeklyReflection):
    try:
        await db.execute(
            """INSERT INTO weekly_reflections
                 (user_id, week_start, week_end, highlight, mood_trend,
                  top_topics, episode_ids, fact_proposals, raw_response)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb)
               ON CONFLICT (user_id, week_start) DO UPDATE SET
                  week_end = EXCLUDED.week_end,
                  highlight = EXCLUDED.highlight,
                  mood_trend = EXCLUDED.mood_trend,
                  top_topics = EXCLUDED.top_topics,
                  episode_ids = EXCLUDED.episode_ids,
                  fact_proposals = EXCLUDED.fact_proposals,
                  raw_response = EXCLUDED.raw_response,
                  updated_at = NOW()""",
            user_id,
            r.week_start, r.week_end,
            r.highlight, r.mood_trend,
            json.dumps(r.top_topics),
            json.dumps(r.episodes_created),
            json.dumps(r.fact_proposals),
            json.dumps(r.raw_response, default=str),
        )
    except Exception as e:
        logger.error("weekly_reflections upsert failed: %s", e)


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try: n = int(v)
    except Exception: return lo
    return max(lo, min(hi, n))


def _clamp_float(v: Any, lo: float, hi: float) -> float:
    try: n = float(v)
    except Exception: return lo
    return max(lo, min(hi, n))
