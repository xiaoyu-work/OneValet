"""
Habit Discovery — analyzes user behavior patterns and stores them as True Memory facts.

Run weekly via CronService. Examines:
1. Conversation timestamps → active hours
2. Tool usage patterns → feature preferences
3. Notification interactions → what they care about
4. Calendar patterns → routine detection

Writes discovered habits to True Memory (namespace="habit") so all agents
automatically adapt to the user's patterns.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated

from koa.tool_decorator import tool
from koa.models import AgentToolContext

logger = logging.getLogger(__name__)

# Habit categories and their proactive job mappings
HABIT_JOB_MAPPING = {
    "active_morning_time": "Proactive: Calendar Check",
    "active_evening_time": "Proactive: Evening Summary",
    "task_review_time": "Proactive: Task Check",
    "preferred_weather_time": "Proactive: Weather Alert",
}


@tool
async def analyze_user_habits(*, context: AgentToolContext) -> str:
    """Analyze recent user behavior and discover habit patterns.

    Examines conversation history, tool usage, and activity patterns
    from the past 7 days to identify habits.
    """
    tenant_id = context.tenant_id
    db = context.context_hints.get("db") if context.context_hints else None
    if not db:
        return "Database not available for habit analysis."

    discoveries = []

    # 1. Analyze conversation timestamps for active hours
    try:
        recent_messages = await db.fetch(
            """SELECT created_at AT TIME ZONE 'UTC' as ts
               FROM messages
               WHERE user_id = $1 AND role = 'user'
               AND created_at > NOW() - INTERVAL '7 days'
               ORDER BY created_at""",
            tenant_id,
        )
        if recent_messages and len(recent_messages) >= 5:
            hours = [row["ts"].hour for row in recent_messages]
            from collections import Counter

            hour_counts = Counter(hours)
            top_hours = hour_counts.most_common(3)

            morning_hours = [(h, c) for h, c in top_hours if 5 <= h <= 12]
            evening_hours = [(h, c) for h, c in top_hours if 18 <= h <= 23]

            if morning_hours:
                peak_morning = morning_hours[0][0]
                discoveries.append(
                    {
                        "namespace": "habit",
                        "fact_key": "active_morning_time",
                        "value": {
                            "hour": peak_morning,
                            "confidence_basis": f"{morning_hours[0][1]} messages",
                        },
                        "summary": f"User is typically most active around {peak_morning}:00 in the morning.",
                        "how_to_apply": f"Schedule morning notifications around {peak_morning}:00, not earlier.",
                    }
                )

            if evening_hours:
                peak_evening = evening_hours[0][0]
                discoveries.append(
                    {
                        "namespace": "habit",
                        "fact_key": "active_evening_time",
                        "value": {
                            "hour": peak_evening,
                            "confidence_basis": f"{evening_hours[0][1]} messages",
                        },
                        "summary": f"User is typically most active around {peak_evening}:00 in the evening.",
                        "how_to_apply": f"Schedule evening notifications around {peak_evening}:00.",
                    }
                )
    except Exception as e:
        logger.debug(f"Active hours analysis failed: {e}")

    # 2. Analyze tool usage patterns
    try:
        tool_usage = await db.fetch(
            """SELECT metadata->>'agent_name' as agent, COUNT(*) as cnt
               FROM activities
               WHERE user_id = $1 AND created_at > NOW() - INTERVAL '7 days'
               GROUP BY metadata->>'agent_name'
               ORDER BY cnt DESC LIMIT 5""",
            tenant_id,
        )
        if tool_usage:
            top_agents = [
                (row["agent"], row["cnt"]) for row in tool_usage if row["agent"]
            ]
            if top_agents:
                agent_summary = ", ".join(
                    f"{a} ({c}x)" for a, c in top_agents[:3]
                )
                discoveries.append(
                    {
                        "namespace": "habit",
                        "fact_key": "most_used_features",
                        "value": {
                            "agents": [
                                {"name": a, "count": c} for a, c in top_agents
                            ]
                        },
                        "summary": f"User's most used features this week: {agent_summary}.",
                        "how_to_apply": "Prioritize proactive updates for these features.",
                    }
                )
    except Exception as e:
        logger.debug(f"Tool usage analysis failed: {e}")

    # 3. Analyze calendar patterns (meeting density by day)
    try:
        cal_provider = (
            context.context_hints.get("calendar_provider")
            if context.context_hints
            else None
        )
        if cal_provider:
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            start = (now - timedelta(days=14)).isoformat()
            end = now.isoformat()
            events = await cal_provider.list_events(time_min=start, time_max=end)
            if events.get("success") and events.get("data"):
                from collections import Counter

                day_counts: Counter[str] = Counter()
                for ev in events["data"]:
                    start_str = ev.get("start", {}).get("dateTime", "")
                    if start_str:
                        try:
                            from dateutil.parser import parse as dateparse

                            dt = dateparse(start_str)
                            day_counts[dt.strftime("%A")] += 1
                        except Exception:
                            pass
                if day_counts:
                    busiest = day_counts.most_common(1)[0]
                    quietest = day_counts.most_common()[-1]
                    discoveries.append(
                        {
                            "namespace": "habit",
                            "fact_key": "weekly_rhythm",
                            "value": {
                                "busiest_day": busiest[0],
                                "quietest_day": quietest[0],
                                "by_day": dict(day_counts),
                            },
                            "summary": (
                                f"Busiest day is usually {busiest[0]} ({busiest[1]} events/week). "
                                f"Quietest is {quietest[0]}."
                            ),
                            "how_to_apply": (
                                f"On {busiest[0]}s, keep notifications minimal. "
                                f"On {quietest[0]}s, suggest catching up on tasks."
                            ),
                        }
                    )
    except Exception as e:
        logger.debug(f"Calendar pattern analysis failed: {e}")

    # 4. Analyze email notification interactions (tapped vs dismissed)
    try:
        interactions = await db.fetch(
            """SELECT sender, action, COUNT(*) as cnt
               FROM notification_interactions
               WHERE user_id = $1 AND created_at > NOW() - INTERVAL '14 days'
                 AND category = 'email_alert' AND sender != ''
               GROUP BY sender, action
               ORDER BY cnt DESC LIMIT 20""",
            tenant_id,
        )
        if interactions:
            from collections import defaultdict
            sender_stats: dict = defaultdict(lambda: {"tapped": 0, "dismissed": 0})
            for row in interactions:
                sender_stats[row["sender"]][row["action"]] = row["cnt"]

            # Senders the user consistently ignores (dismissed >= 3, never tapped)
            ignored = [s for s, stats in sender_stats.items()
                       if stats["dismissed"] >= 3 and stats["tapped"] == 0]
            # Senders the user always taps (tapped >= 3, never dismissed)
            priority = [s for s, stats in sender_stats.items()
                        if stats["tapped"] >= 3 and stats["dismissed"] == 0]

            if ignored:
                discoveries.append({
                    "namespace": "feedback",
                    "fact_key": "email_ignore_senders",
                    "value": {"senders": ignored},
                    "summary": f"User consistently ignores email notifications from: {', '.join(ignored[:5])}",
                    "how_to_apply": "Do NOT notify user about emails from these senders.",
                })
            if priority:
                discoveries.append({
                    "namespace": "feedback",
                    "fact_key": "email_priority_senders",
                    "value": {"senders": priority},
                    "summary": f"User always reads email notifications from: {', '.join(priority[:5])}",
                    "how_to_apply": "ALWAYS notify user about emails from these senders, even if content seems routine.",
                })
    except Exception as e:
        logger.debug(f"Email interaction analysis failed: {e}")

    if not discoveries:
        return "Not enough data yet to discover habits (need at least a week of usage)."

    # 4. Convert discoveries into true_memory_proposals format
    #    These will be picked up by koi-backend's _apply_true_memory_proposals()
    #    via the orchestrator's result.metadata["true_memory_proposals"] pipeline.
    proposals = []
    for d in discoveries:
        proposals.append({
            "operation": "upsert",
            "namespace": d["namespace"],
            "fact_key": d["fact_key"],
            "value": d["value"],
            "summary": d["summary"],
            "confidence": 0.7,
            "source_type": "system_inferred",
            "how_to_apply": d.get("how_to_apply", ""),
            "why": "Inferred from user behavior patterns over the past week.",
        })

    # Store proposals in context metadata so the orchestrator picks them up
    if context.metadata is None:
        context.metadata = {}
    existing = context.metadata.get("true_memory_proposals", [])
    context.metadata["true_memory_proposals"] = existing + proposals

    # 5. Adjust proactive job schedules based on discovered timing
    cron_service = (
        context.context_hints.get("cron_service")
        if context.context_hints
        else None
    )
    if cron_service:
        await _adjust_proactive_schedules(cron_service, tenant_id, discoveries)

    summary_lines = [f"Discovered {len(discoveries)} habit(s):"]
    for d in discoveries:
        summary_lines.append(f"  • {d['summary']}")

    return "\n".join(summary_lines)


async def _adjust_proactive_schedules(
    cron_service, tenant_id: str, discoveries: list
):
    """Adjust proactive cron job schedules based on discovered habits."""
    from koa.triggers.cron.models import CronJobPatch, CronScheduleSpec

    timing: dict[str, int] = {}
    for d in discoveries:
        if d["fact_key"] == "active_morning_time":
            timing["morning"] = d["value"]["hour"]
        elif d["fact_key"] == "active_evening_time":
            timing["evening"] = d["value"]["hour"]

    if not timing:
        return

    try:
        jobs = cron_service.list_jobs(user_id=tenant_id, include_disabled=False)
        for job in jobs:
            new_schedule = None

            if job.name == "Proactive: Calendar Check" and "morning" in timing:
                h = timing["morning"]
                new_schedule = CronScheduleSpec(expr=f"0 {h} * * *")

            elif job.name == "Proactive: Task Check" and "morning" in timing:
                h = timing["morning"]
                new_schedule = CronScheduleSpec(expr=f"0 {h} * * *")

            elif job.name == "Proactive: Evening Summary" and "evening" in timing:
                h = timing["evening"]
                new_schedule = CronScheduleSpec(expr=f"0 {h} * * *")

            elif job.name == "Proactive: Task Rollover" and "evening" in timing:
                h = max(timing.get("evening", 20) - 1, 18)
                new_schedule = CronScheduleSpec(expr=f"0 {h} * * *")

            if new_schedule:
                try:
                    await cron_service.update(
                        job.id, CronJobPatch(schedule=new_schedule)
                    )
                    logger.info(
                        f"Adjusted '{job.name}' schedule for tenant {tenant_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to adjust '{job.name}': {e}")
    except Exception as e:
        logger.warning(f"Schedule adjustment failed for {tenant_id}: {e}")
