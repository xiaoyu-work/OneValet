"""Structural intent classifier tests — fast, deterministic, CI-safe.

Covers:
- Fast-path regex short-circuit on trivial utterances
- Fixture schema validity
- Embedding router with deterministic fake backend
- IntentAnalysis field population (confidence, clarification, slots)
- Feedback store semantics
"""

import pytest

from koa.orchestrator.intent_analyzer import (
    CLARIFY_CONFIDENCE_THRESHOLD,
    FastPathClassifier,
    IntentAnalyzer,
)
from koa.orchestrator.intent_embedding import EmbeddingRouter
from koa.orchestrator.intent_feedback import InMemoryIntentFeedbackStore

from .harness import domain_match, load_fixtures

# ---------------------------------------------------------------------------
# Fast-path classifier
# ---------------------------------------------------------------------------


def test_fast_path_matches_greetings_acks_cancel():
    fp = FastPathClassifier()
    for msg in [
        "hi",
        "hello",
        "hey!",
        "thanks",
        "ok",
        "好的",
        "你好",
        "谢谢",
        "算了",
        "cancel",
        "bye",
    ]:
        out = fp.classify(msg)
        assert out is not None, f"fast-path should classify {msg!r}"
        assert out.domains == ["general"]
        assert out.confidence >= 0.95
        assert out.source == "fast_path"


def test_fast_path_rejects_long_messages():
    fp = FastPathClassifier()
    assert fp.classify("hi, can you send an email to alice?") is None


def test_fast_path_rejects_unrelated_short_messages():
    fp = FastPathClassifier()
    assert fp.classify("flight tokyo") is None
    assert fp.classify("remind me") is None


def test_fast_path_empty_and_none_safe():
    fp = FastPathClassifier()
    assert fp.classify("") is None
    assert fp.classify("   ") is None


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------


def test_fixtures_load_successfully():
    fixtures = load_fixtures()
    assert len(fixtures) >= 40, f"expected >= 40 fixtures, got {len(fixtures)}"
    for fx in fixtures:
        assert fx.message, f"fixture {fx.source_file} has empty message"
        assert fx.expected_domains, f"fixture {fx.source_file} has no expected_domains"
        assert fx.expected_type in {"single", "multi"}


def test_fast_path_covers_tagged_fixtures():
    """Every fixture tagged `fast_path` must actually be classified by fast-path."""
    fp = FastPathClassifier()
    missed = []
    for fx in load_fixtures(tag_filter="fast_path"):
        out = fp.classify(fx.message)
        if out is None or not domain_match(out.domains, fx.expected_domains):
            missed.append(fx.message)
    assert not missed, f"fast-path failed for tagged fixtures: {missed}"


# ---------------------------------------------------------------------------
# Embedding router with fake backend
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic fake embedder.

    Maps each word in a text to a one-hot dimension in a small vocabulary.
    Texts sharing vocabulary produce similar vectors (cosine > 0).
    """

    def __init__(self, vocab):
        self._vocab = {w: i for i, w in enumerate(vocab)}
        self._dim = len(vocab)

    async def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * self._dim
            for w in t.lower().split():
                idx = self._vocab.get(w)
                if idx is not None:
                    vec[idx] += 1.0
            out.append(vec)
        return out


@pytest.mark.asyncio
async def test_embedding_router_classifies_known_centroid():
    vocab = ["email", "send", "meeting", "lights", "flight", "tokyo", "turn", "off"]
    backend = _FakeEmbedder(vocab)
    router = EmbeddingRouter(backend, hit_threshold=0.5, margin=0.0)
    from koa.orchestrator.intent_embedding import Exemplar

    await router.fit(
        [
            Exemplar("send email", "communication"),
            Exemplar("email meeting", "communication"),
            Exemplar("turn off lights", "lifestyle"),
            Exemplar("turn lights off", "lifestyle"),
            Exemplar("flight tokyo", "travel"),
            Exemplar("tokyo flight", "travel"),
        ]
    )
    result = await router.classify("send email")
    assert result is not None
    assert result.domains == ["communication"]
    assert result.source == "embedding"


@pytest.mark.asyncio
async def test_embedding_router_refuses_low_confidence():
    vocab = ["a", "b", "c", "d"]
    backend = _FakeEmbedder(vocab)
    router = EmbeddingRouter(backend, hit_threshold=0.95)
    from koa.orchestrator.intent_embedding import Exemplar

    await router.fit(
        [
            Exemplar("a", "communication"),
            Exemplar("b", "communication"),
            Exemplar("c", "travel"),
            Exemplar("d", "travel"),
        ]
    )
    # "a b c d" is equidistant from both centroids — must refuse.
    assert await router.classify("a b c d") is None


@pytest.mark.asyncio
async def test_embedding_router_requires_fit():
    backend = _FakeEmbedder(["x"])
    router = EmbeddingRouter(backend)
    assert await router.classify("anything") is None


# ---------------------------------------------------------------------------
# Analyzer parsing (confidence, clarification, slots)
# ---------------------------------------------------------------------------


class _StubLLM:
    def __init__(self, content):
        self._content = content

    async def chat_completion(self, messages, config=None):
        class _R:
            content = self._content

        return _R()


@pytest.mark.asyncio
async def test_analyzer_parses_confidence_and_slots():
    stub = _StubLLM(
        '{"intent_type":"single","domains":["communication"],"needs_memory":false,'
        '"confidence":0.92,"needs_clarification":false,'
        '"slots":{"recipient":"alice@acme.com","target":"Q3 report"},"sub_tasks":[]}'
    )
    analyzer = IntentAnalyzer(stub, fast_path=None)
    out = await analyzer.analyze("email alice the Q3 report")
    assert out.confidence == pytest.approx(0.92)
    assert out.slots == {"recipient": "alice@acme.com", "target": "Q3 report"}
    assert out.source == "llm"
    assert out.needs_clarification is False


@pytest.mark.asyncio
async def test_analyzer_auto_trips_clarify_on_low_confidence():
    stub = _StubLLM(
        '{"intent_type":"single","domains":["general"],"needs_memory":false,'
        '"confidence":0.3,"needs_clarification":false,"sub_tasks":[]}'
    )
    analyzer = IntentAnalyzer(stub, fast_path=None)
    out = await analyzer.analyze("handle that")
    assert out.needs_clarification is True
    assert out.clarification_question  # non-empty
    assert out.confidence < CLARIFY_CONFIDENCE_THRESHOLD


@pytest.mark.asyncio
async def test_analyzer_falls_back_on_exception():
    class _Boom:
        async def chat_completion(self, messages, config=None):
            raise RuntimeError("no network")

    analyzer = IntentAnalyzer(_Boom(), fast_path=None)
    out = await analyzer.analyze("anything")
    assert out.source == "fallback"
    assert out.domains == ["all"]
    assert out.confidence == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_analyzer_skips_fast_path_with_history():
    stub = _StubLLM(
        '{"intent_type":"single","domains":["communication"],"needs_memory":true,'
        '"confidence":0.9,"needs_clarification":false,"sub_tasks":[]}'
    )
    analyzer = IntentAnalyzer(stub)
    # "ok" in isolation is fast-pathed, but with history it must hit LLM
    # because it may be confirming a pending action.
    out = await analyzer.analyze(
        "ok",
        conversation_history=[
            {"role": "user", "content": "draft an email to alice"},
            {"role": "assistant", "content": "Shall I send it now?"},
        ],
    )
    assert out.source == "llm"
    assert out.domains == ["communication"]


# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_store_records_and_summarizes():
    store = InMemoryIntentFeedbackStore()
    await store.record(
        tenant_id="t1",
        user_message="hi",
        intent_type="single",
        domains=["general"],
        confidence=0.99,
        source="fast_path",
        outcome="completed",
    )
    await store.record(
        tenant_id="t1",
        user_message="cancel that",
        intent_type="single",
        domains=["general"],
        confidence=0.5,
        source="llm",
        outcome="cancelled",
    )
    await store.record(
        tenant_id="t2",
        user_message="send email",
        intent_type="single",
        domains=["communication"],
        confidence=0.9,
        source="llm",
        outcome="completed",
    )

    summary = await store.accuracy_summary()
    assert summary["total"] == 3
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["positive_ratio"] == pytest.approx(2 / 3)

    # Tenant-scoped
    t1 = await store.accuracy_summary(tenant_id="t1")
    assert t1["total"] == 2


@pytest.mark.asyncio
async def test_feedback_store_rejects_invalid_outcome():
    store = InMemoryIntentFeedbackStore()
    with pytest.raises(ValueError):
        await store.record(
            tenant_id="t1",
            user_message="x",
            intent_type="single",
            domains=["general"],
            confidence=1.0,
            source="llm",
            outcome="not-a-real-outcome",
        )
