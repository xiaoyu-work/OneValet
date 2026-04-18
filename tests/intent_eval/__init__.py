"""Intent classification eval harness.

Two modes of operation:

1. **Structural / deterministic tests** (this directory's ``test_*`` files,
   run with plain ``pytest``): exercise the fast-path classifier, the
   embedding router (with a deterministic fake backend), confidence/slot
   parsing, and feedback store semantics.  No LLM calls, no flakes, CI-safe.

2. **Semantic accuracy eval** (``test_llm_accuracy.py``): runs the full
   :class:`IntentAnalyzer` against a labelled fixture set using a real
   LLM.  Gated behind the ``KOA_RUN_INTENT_EVAL=1`` environment variable
   so CI doesn't pay API costs on every PR.

Fixture format
--------------

Fixtures live in ``tests/intent_eval/fixtures/*.jsonl``.  Each line is a
JSON object with keys:

- ``message`` (str, required) — user utterance
- ``expected_domains`` (list[str], required) — accepted domain sets; the
  classifier passes if its ``domains`` is any of the listed options
- ``expected_type`` (str, optional) — "single" or "multi"; default "single"
- ``expected_clarify`` (bool, optional) — whether ``needs_clarification``
  should be True; default False
- ``tags`` (list[str], optional) — free-form labels for slicing results
  (e.g., "zh", "ambiguous", "multi-tool")
- ``notes`` (str, optional) — human commentary

Add new fixtures by appending to the appropriate file; the harness auto-
discovers every ``.jsonl`` under ``fixtures/``.
"""
