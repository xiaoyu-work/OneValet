"""Memory lifecycle — daily aggregation + weekly reflection pipeline.

This package is invoked by cron jobs and by the WeeklyReflectorAgent.  It
is deliberately *not* exposed as tools to the LLM; all behavior here is
mechanical bookkeeping or an explicit reflection step.

Modules:
  daily_log_aggregator  — pure-SQL roll-up of a user's day into ``daily_logs``
  episode_store         — CRUD for ``episodes`` + ``entities``
  weekly_reflector      — the LLM-driven reflection step itself
"""
