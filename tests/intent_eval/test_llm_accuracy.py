"""Semantic accuracy eval — requires a real LLM, runs only when
``KOA_RUN_INTENT_EVAL=1``.

Run manually:

    KOA_RUN_INTENT_EVAL=1 uv run pytest tests/intent_eval/test_llm_accuracy.py -s

The harness loads every fixture from ``fixtures/*.jsonl``, runs the
full :class:`IntentAnalyzer`, and prints a summary + confusion matrix.
The test **fails only** if overall accuracy drops below a configurable
floor (default 0.80) — we want visibility into gradual regressions
without making CI flaky on every prompt tweak.

Configure the floor via ``KOA_INTENT_EVAL_MIN_ACCURACY=0.85``.
"""

from __future__ import annotations

import asyncio
import json
import os
import pytest

from koa.orchestrator.intent_analyzer import IntentAnalyzer

from .harness import domain_match, load_fixtures, summarize

_GATE = os.environ.get("KOA_RUN_INTENT_EVAL", "").lower() in {"1", "true", "yes"}
_MIN_ACCURACY = float(os.environ.get("KOA_INTENT_EVAL_MIN_ACCURACY", "0.80"))

pytestmark = pytest.mark.skipif(
    not _GATE, reason="intent eval requires KOA_RUN_INTENT_EVAL=1 + live LLM"
)


def _make_llm_client():
    """Resolve an LLM client from the environment.

    Uses the same LLM client factory as the main app so the eval reflects
    the exact classifier the user sees.  Override via
    ``KOA_INTENT_EVAL_MODEL`` to target a specific model.
    """
    try:
        from koa.llm.litellm_client import LiteLLMClient
    except ImportError as exc:
        pytest.skip(f"LLM client unavailable: {exc}")

    model = os.environ.get("KOA_INTENT_EVAL_MODEL") or os.environ.get(
        "KOA_DEFAULT_MODEL", "gpt-4o-mini"
    )
    provider = os.environ.get("KOA_INTENT_EVAL_PROVIDER", "openai")
    api_version = os.environ.get("KOA_INTENT_EVAL_API_VERSION")
    kwargs = {"model": model, "provider_name": provider}
    if api_version:
        kwargs["api_version"] = api_version
    return LiteLLMClient(**kwargs)


@pytest.mark.asyncio
async def test_intent_accuracy_over_fixtures():
    fixtures = load_fixtures()
    assert fixtures, "no fixtures found"

    client = _make_llm_client()
    analyzer = IntentAnalyzer(client, fast_path=None)  # force LLM path

    results = []
    # Run with bounded concurrency so we don't burn rate limit.
    sem = asyncio.Semaphore(int(os.environ.get("KOA_INTENT_EVAL_CONCURRENCY", "4")))

    async def _run(fx):
        async with sem:
            try:
                intent = await analyzer.analyze(fx.message)
            except Exception as exc:
                return {
                    "fixture": fx.source_file,
                    "message": fx.message,
                    "tags": fx.tags,
                    "expected_domains": fx.expected_domains,
                    "predicted_domains": [],
                    "correct": False,
                    "error": str(exc),
                }
            domains_ok = domain_match(intent.domains, fx.expected_domains)
            type_ok = (intent.intent_type == fx.expected_type) or (
                # Multi-intent is a strict superset — accept single when multi expected
                # only if the expected-set listing includes a single option.
                fx.expected_type == "multi"
                and intent.intent_type == "single"
                and any(len(opt) == 1 for opt in fx.expected_domains)
            )
            clarify_ok = (
                (not fx.expected_clarify)
                or intent.needs_clarification
            )
            return {
                "fixture": fx.source_file,
                "message": fx.message,
                "tags": fx.tags,
                "expected_domains": fx.expected_domains,
                "predicted_domains": list(intent.domains),
                "predicted_type": intent.intent_type,
                "predicted_confidence": intent.confidence,
                "predicted_clarify": intent.needs_clarification,
                "correct": bool(domains_ok and type_ok and clarify_ok),
            }

    results = await asyncio.gather(*[_run(fx) for fx in fixtures])

    summary = summarize(results)
    # Emit in a machine-readable form so CI can scrape it.
    print("\n=== INTENT EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== FAILURES ===")
    for r in results:
        if not r["correct"]:
            print(
                f"  [{r['fixture']}] {r['message']!r} "
                f"expected={r['expected_domains']} got={r['predicted_domains']}"
            )

    assert summary["accuracy"] >= _MIN_ACCURACY, (
        f"Intent classification accuracy {summary['accuracy']:.2%} "
        f"below floor {_MIN_ACCURACY:.2%}"
    )
