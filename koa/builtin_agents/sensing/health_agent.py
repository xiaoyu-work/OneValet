"""HealthAgent — derives sleep / activity / stress state from HealthKit samples.

Runs once per user per day (cron: 05:00 local time).  Reads yesterday's
``health_samples`` rows and writes a ``user_state`` summary plus, when
appropriate, memory proposals about the user's sleep schedule or stress
baseline.

Scoring heuristics (chosen to be simple, explainable, and cheap):

  sleep_score (0-100):
    60 = 7h of sleep
    +5 per 30min over, -5 per 30min under
    clamped to [0, 100]

  stress_score (0-100):
    baseline 50
    +20 if resting_hr > trailing_median + 5
    -10 if hrv_avg > trailing_median + 10%
    +15 if mindful_minutes == 0 AND weekday
    clamped to [0, 100]

These are not medical signals.  They exist only to give downstream agents
a quick "how is the user doing?" scalar.  The assistant should never claim
clinical meaning from them.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from koa import valet

from .base import SensingAgent, SensingResult, make_proposal

logger = logging.getLogger(__name__)


@valet(domain="sensing", expose_as_tool=False)
class HealthAgent(SensingAgent):
    """Daily HealthKit analysis: sleep, activity, stress, mood scoring."""

    SOURCE_TABLE = "health_samples"

    async def analyze(
        self,
        db: Any,
        user_id: str,
        local_date: date,
        tz_name: str,
    ) -> SensingResult:
        day_start, day_end = _local_day_bounds(local_date, tz_name)
        samples = await _fetch_samples(db, user_id, day_start, day_end)

        if not samples:
            return SensingResult(notes=f"No health samples for {local_date}")

        sleep_minutes = _sleep_minutes(samples)
        sleep_score = _sleep_score(sleep_minutes)
        steps = int(_sum_type(samples, "steps"))
        active_energy_kcal = int(_sum_type(samples, "active_energy"))
        mindful_minutes = _duration_minutes(samples, "mindful")
        workout_minutes = _duration_minutes(samples, "workout")
        resting_hr = _avg_type(samples, "resting_hr")
        hrv = _avg_type(samples, "hrv")
        activity_minutes = _activity_minutes(samples)

        baseline = await _fetch_baselines(db, user_id, local_date)
        stress = _stress_score(
            resting_hr, hrv, mindful_minutes, local_date.weekday() < 5, baseline,
        )

        flags: List[str] = []
        if sleep_minutes is not None and sleep_minutes < 5 * 60:
            flags.append("low_sleep")
        if stress is not None and stress >= 75:
            flags.append("elevated_stress")
        if workout_minutes >= 30:
            flags.append("worked_out")

        user_state_fields: Dict[str, Any] = {
            "sleep_minutes": sleep_minutes,
            "sleep_score": sleep_score,
            "hrv_ms": hrv,
            "resting_hr": resting_hr,
            "steps": steps,
            "activity_minutes": activity_minutes,
            "stress_score": stress,
            "source_data": {
                "active_energy_kcal": active_energy_kcal,
                "mindful_minutes": mindful_minutes,
                "workout_minutes": workout_minutes,
                "sample_count": len(samples),
            },
        }

        proposals: List[Dict[str, Any]] = []
        # Only promote to true_memory when a pattern has stabilized.
        recent_low_sleep = await _count_recent_flag(db, user_id, local_date, "low_sleep", window_days=7)
        if recent_low_sleep >= 3:
            proposals.append(make_proposal(
                namespace="health",
                fact_key="recurring_low_sleep",
                value=True,
                summary=f"User has slept <5h on {recent_low_sleep} of the past 7 days.",
                how_to_apply=(
                    "Be gentler and shorter in morning briefings. Avoid suggesting "
                    "high-energy commitments before noon. Ask once whether to "
                    "defer noncritical reminders."
                ),
                confidence=0.8,
                why="Observed via HealthKit sleep samples for 3+ of the past 7 days.",
            ))

        notes = (
            f"HealthAgent: sleep={sleep_minutes}min score={sleep_score} "
            f"steps={steps} stress={stress} flags={flags}"
        )
        return SensingResult(
            user_state_fields=user_state_fields,
            proposals=proposals,
            flags=flags,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Pure helpers — trivially unit-testable without instantiating the agent.
# ---------------------------------------------------------------------------

def _local_day_bounds(local_date: date, tz_name: str):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name) if tz_name and tz_name != "UTC" else timezone.utc
    except Exception:
        tz = timezone.utc
    start_local = datetime.combine(local_date, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


async def _fetch_samples(db, user_id: str, start_utc: datetime, end_utc: datetime) -> List[Dict[str, Any]]:
    try:
        rows = await db.fetch(
            """SELECT type, started_at, ended_at, value, unit, metadata
               FROM health_samples
               WHERE user_id = $1 AND started_at >= $2 AND started_at < $3""",
            user_id, start_utc, end_utc,
        )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("_fetch_samples failed: %s", e)
        return []


async def _fetch_baselines(db, user_id: str, local_date: date) -> Dict[str, Any]:
    try:
        rows = await db.fetch(
            """SELECT resting_hr, hrv_ms FROM user_state
               WHERE user_id = $1 AND local_date < $2
               ORDER BY local_date DESC LIMIT 14""",
            user_id, local_date,
        )
        rhrs = [r["resting_hr"] for r in rows if r["resting_hr"] is not None]
        hrvs = [r["hrv_ms"] for r in rows if r["hrv_ms"] is not None]
        return {
            "resting_hr_median": statistics.median(rhrs) if rhrs else None,
            "hrv_median": statistics.median(hrvs) if hrvs else None,
        }
    except Exception as e:
        logger.debug("_fetch_baselines failed: %s", e)
        return {}


async def _count_recent_flag(db, user_id: str, local_date: date, flag: str, window_days: int) -> int:
    try:
        row = await db.fetchrow(
            """SELECT COUNT(*) AS c FROM user_state
               WHERE user_id = $1
                 AND local_date >= $2 AND local_date <= $3
                 AND flags @> ARRAY[$4]::text[]""",
            user_id,
            local_date - timedelta(days=window_days),
            local_date,
            flag,
        )
        return int(row["c"]) if row else 0
    except Exception as e:
        logger.debug("_count_recent_flag failed: %s", e)
        return 0


def _sum_type(samples: List[Dict[str, Any]], t: str) -> float:
    return sum(float(s.get("value") or 0) for s in samples if s.get("type") == t)


def _avg_type(samples: List[Dict[str, Any]], t: str) -> Optional[float]:
    vals = [s["value"] for s in samples if s.get("type") == t and s.get("value") is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _duration_minutes(samples: List[Dict[str, Any]], t: str) -> int:
    total = 0
    for s in samples:
        if s.get("type") != t:
            continue
        st, en = s.get("started_at"), s.get("ended_at")
        if not (st and en):
            continue
        total += int((en - st).total_seconds() // 60)
    return total


def _sleep_minutes(samples: List[Dict[str, Any]]) -> Optional[int]:
    """Count only "asleep" category stages, not "inBed" or "awake"."""
    sleep_stages = {"asleep", "core", "deep", "rem"}
    total = 0
    found = False
    for s in samples:
        if s.get("type") != "sleep":
            continue
        stage = (s.get("metadata") or {}).get("stage") or ""
        if stage not in sleep_stages:
            continue
        st, en = s.get("started_at"), s.get("ended_at")
        if not (st and en):
            continue
        total += int((en - st).total_seconds() // 60)
        found = True
    return total if found else None


def _sleep_score(minutes: Optional[int]) -> Optional[int]:
    if minutes is None:
        return None
    delta_halfhours = (minutes - 420) / 30
    score = 60 + 5 * delta_halfhours
    return max(0, min(100, int(round(score))))


def _activity_minutes(samples: List[Dict[str, Any]]) -> int:
    ex = _duration_minutes(samples, "exercise")
    wo = _duration_minutes(samples, "workout")
    return max(ex, wo)


def _stress_score(
    resting_hr: Optional[float],
    hrv: Optional[float],
    mindful_minutes: int,
    is_weekday: bool,
    baseline: Dict[str, Any],
) -> Optional[int]:
    if resting_hr is None and hrv is None:
        return None
    score = 50
    rhr_base = baseline.get("resting_hr_median")
    if resting_hr is not None and rhr_base is not None and resting_hr > rhr_base + 5:
        score += 20
    hrv_base = baseline.get("hrv_median")
    if hrv is not None and hrv_base is not None and hrv > hrv_base * 1.1:
        score -= 10
    if mindful_minutes == 0 and is_weekday:
        score += 15
    return max(0, min(100, int(score)))
