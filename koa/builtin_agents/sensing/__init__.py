"""Sensing agents — Koi's perception layer.

Unlike the majority of Koi's builtin agents (which *operate* external services
like email, calendar, smart-home), sensing agents *perceive* the user
themselves.  They run as background cron jobs, do not appear in the ReAct
tool list, and produce two kinds of output:

  1. ``user_state`` rows — per-day derived scalars (sleep_score, stress_score,
     activity_minutes, mood).  Fast to query, consumed by the briefing agent
     and the chat prefetch layer.

  2. ``true_memory_proposals`` in context.metadata — higher-level facts
     that survive past a single day (e.g., "regularly sleeps <6h on Mondays").

The sensing layer is intentionally *read-only* to the user-facing conversation
surface; it informs other agents but never speaks directly.  Reflection and
proactive notification are the jobs of the reflection/proactive packages.
"""
