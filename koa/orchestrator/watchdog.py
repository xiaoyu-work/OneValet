"""Noticing when a run has stopped making progress.

A model that calls the same tool with the same arguments and gets the
same answer is not going to reach a different conclusion by doing it a
fourth time. Left alone it will spend the whole turn budget that way, so
the run is stopped and whatever was gathered is handed back.

The three checks are ordered by how much they tell us. Identical
name-plus-arguments is unambiguous. Identical results from an identical
call is the same thing seen from the other side, and catches tools whose
arguments serialise differently between turns. A repeating name pattern
is the weakest: A-B-A-B can be legitimate work, so it only counts when
the arguments repeat too or when there is nothing better to go on.
"""

from typing import Any, List, Optional

from .run_state import RunState

STUCK_MESSAGE = (
    "I noticed I was repeating the same actions without making progress. "
    "Let me provide what I have so far."
)

#: Separator between the steps of a detected cycle, as shown in the log.
_CYCLE_ARROW = "\u2194"


def detect_loop(
    names: List[str],
    fingerprints: Optional[List[str]] = None,
    result_hashes: Optional[List[str]] = None,
) -> Optional[str]:
    """Describe the repetition this run is stuck in, or None if it is moving."""
    # Same tool, same arguments, three times: nothing else needs checking.
    if fingerprints and len(fingerprints) >= 3 and len(set(fingerprints[-3:])) == 1:
        return f"Exact repeat: {fingerprints[-1]} called 3 times with same args"

    # The same call returning the same bytes twice is spending tokens to
    # learn something it already knows.
    if (
        result_hashes
        and len(result_hashes) >= 2
        and result_hashes[-1] == result_hashes[-2]
        and fingerprints
        and len(fingerprints) >= 2
        and fingerprints[-1] == fingerprints[-2]
    ):
        return "No progress: same tool returned identical results twice"

    # Same name three times running. Only conclusive when the arguments
    # agree, or when no argument history was recorded to check against.
    if len(names) >= 3 and len(set(names[-3:])) == 1:
        if not fingerprints or len(set(fingerprints[-3:])) == 1:
            return f"Loop detected: {names[-1]} called 3 times consecutively"

    for cycle_len in range(2, 5):
        needed = cycle_len * 2
        if len(names) < needed:
            continue
        tail = names[-needed:]
        cycle = tail[:cycle_len]
        if all(tail[i] == cycle[i % cycle_len] for i in range(needed)):
            return (
                f"Cycle detected: {_CYCLE_ARROW.join(cycle)} "
                f"repeated {needed // cycle_len} times"
            )

    return None


def verdict(state: RunState, tool_calls: List[Any], timed_results: List[Any]) -> Optional[str]:
    """Record this turn's calls and report whether the run is going in circles."""
    for tc, timed in zip(tool_calls, timed_results):
        state.observe_for_watchdog(tc.name, tc.arguments, getattr(timed, "result", timed))
    return detect_loop(state.recent_names, state.recent_fingerprints, state.recent_result_hashes)
