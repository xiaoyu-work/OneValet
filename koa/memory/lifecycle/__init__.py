"""Memory lifecycle — daily aggregation + weekly reflection pipeline.

This package is invoked by cron jobs and by the WeeklyReflectorAgent. It
is deliberately *not* exposed as tools to the LLM; all behavior here is
mechanical bookkeeping or an explicit reflection step.

Modules:
  daily_log_aggregator  — pure-SQL roll-up of a user's day into a daily_log
                          episode (written to Momex via EpisodeMemory).
  episode_memory        — thin adapter over MomexMemory for episodic writes
                          and reads (kind=episode metadata convention).
  weekly_reflector      — the LLM-driven reflection step itself, also
                          persisting its output as an episode.
"""

from .episode_memory import EPISODE_KIND, EpisodeMemory

__all__ = ["EpisodeMemory", "EPISODE_KIND"]
