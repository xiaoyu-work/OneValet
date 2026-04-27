"""Embedding-based L1 intent router.

A lightweight nearest-centroid classifier that short-circuits the full
LLM intent analyzer when the user's message is semantically close to a
previously-seen example.

Design
------

At initialization time, callers register "exemplars" — short labelled
utterances representing each domain.  The router embeds them once and
averages per-domain centroids.

At classify-time, the router embeds the user message and compares (cosine
similarity) to every centroid.  A classification is returned only when:

- top-1 similarity ≥ ``hit_threshold`` (default 0.82), AND
- gap between top-1 and top-2 ≥ ``margin`` (default 0.05) so we don't
  emit low-confidence classifications that the LLM would handle better.

All other cases return ``None``, letting the caller fall through to the
LLM classifier.

Embedding backend is pluggable: any object with

.. code-block:: python

    async def embed(texts: list[str]) -> list[list[float]]

will work.  Dimensionality only needs to be consistent across calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .intent_analyzer import VALID_DOMAINS, IntentAnalysis

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


@dataclass
class Exemplar:
    text: str
    domain: str
    intent_type: str = "single"
    needs_memory: bool = False


#: Conservative defaults.  Tuned to be **high-precision** — a miss costs
#: one LLM call; a wrong classification can pick the wrong agent.
DEFAULT_HIT_THRESHOLD = 0.82
DEFAULT_MARGIN = 0.05


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _avg(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]


class EmbeddingRouter:
    """Nearest-centroid domain router.

    Args:
        backend: Embedding backend (see :class:`EmbeddingBackend`).
        hit_threshold: Minimum cosine similarity for a classification hit.
        margin: Minimum gap between top-1 and top-2 for a confident hit.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        hit_threshold: float = DEFAULT_HIT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        self.backend = backend
        self.hit_threshold = float(hit_threshold)
        self.margin = float(margin)
        self._centroids: Dict[str, List[float]] = {}
        #: Number of exemplars that contributed to each centroid; used to
        #: refuse classification from under-trained domains.
        self._exemplar_counts: Dict[str, int] = {}
        self._initialized = False

    async def fit(self, exemplars: Iterable[Exemplar]) -> None:
        """Compute per-domain centroids from the exemplar set.

        Must be called before :meth:`classify`.  Safe to call again with
        an updated exemplar set; state is fully replaced.
        """
        grouped: Dict[str, List[str]] = {}
        for ex in exemplars:
            if ex.domain not in VALID_DOMAINS:
                logger.warning("Skipping exemplar with invalid domain: %s", ex.domain)
                continue
            grouped.setdefault(ex.domain, []).append(ex.text)

        if not grouped:
            raise ValueError("No valid exemplars provided")

        # Embed all texts in one backend call to amortize overhead.
        flat_texts: List[str] = []
        boundaries: List[Tuple[str, int, int]] = []
        for domain, texts in grouped.items():
            start = len(flat_texts)
            flat_texts.extend(texts)
            boundaries.append((domain, start, len(flat_texts)))

        try:
            vectors = await self.backend.embed(flat_texts)
        except Exception as exc:
            raise RuntimeError(f"Embedding backend failed during fit: {exc}") from exc
        if len(vectors) != len(flat_texts):
            raise RuntimeError(
                f"Embedding backend returned {len(vectors)} vectors for {len(flat_texts)} texts"
            )

        centroids: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for domain, start, end in boundaries:
            centroids[domain] = _avg(vectors[start:end])
            counts[domain] = end - start

        self._centroids = centroids
        self._exemplar_counts = counts
        self._initialized = True
        logger.info(
            "EmbeddingRouter fit complete: %s",
            ", ".join(f"{d}={n}" for d, n in counts.items()),
        )

    async def classify(self, user_message: str) -> Optional[IntentAnalysis]:
        """Classify a message using the fitted centroids.

        Returns ``None`` when the router is not confident enough — the
        caller should fall through to the LLM classifier.
        """
        if not self._initialized or not self._centroids:
            return None
        if not user_message or not user_message.strip():
            return None
        try:
            vectors = await self.backend.embed([user_message])
        except Exception as exc:
            logger.debug("EmbeddingRouter embed failed: %s", exc)
            return None
        if not vectors:
            return None
        vec = vectors[0]

        scored: List[Tuple[str, float]] = []
        for domain, centroid in self._centroids.items():
            scored.append((domain, _cosine(vec, centroid)))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_domain, top_score = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else 0.0

        if top_score < self.hit_threshold:
            return None
        if (top_score - runner_up) < self.margin:
            return None
        if self._exemplar_counts.get(top_domain, 0) < 2:
            # Single-exemplar domain isn't statistically meaningful.
            return None

        return IntentAnalysis(
            intent_type="single",
            domains=[top_domain],
            needs_memory=(top_domain != "general"),
            raw_message=user_message,
            confidence=min(0.95, top_score),
            needs_clarification=False,
            source="embedding",
        )


# ---------------------------------------------------------------------------
# Default starter exemplars
# ---------------------------------------------------------------------------


#: A small seed set operators can use to bootstrap an EmbeddingRouter.
#: Expand with tenant-specific exemplars over time (see
#: :mod:`koa.orchestrator.intent_feedback` for the raw signal source).
DEFAULT_EXEMPLARS: List[Exemplar] = [
    # communication
    Exemplar("send an email to alice about the project", "communication"),
    Exemplar("reply to the last slack message", "communication"),
    Exemplar("dm bob on discord", "communication"),
    Exemplar("tweet about our launch", "communication"),
    # productivity
    Exemplar("add a meeting tomorrow at 3pm", "productivity"),
    Exemplar("remind me to call mom in an hour", "productivity"),
    Exemplar("create a todo for the Q3 review", "productivity"),
    Exemplar("show my briefing for today", "productivity"),
    Exemplar("schedule a cron job to sync at midnight", "productivity"),
    # lifestyle
    Exemplar("log expense $15 lunch", "lifestyle"),
    Exemplar("turn off the living room lights", "lifestyle"),
    Exemplar("track my amazon package", "lifestyle"),
    Exemplar("play my focus playlist on spotify", "lifestyle"),
    Exemplar("generate an image of a sunset", "lifestyle"),
    # travel
    Exemplar("find flights to tokyo next week", "travel"),
    Exemplar("how do I get to the airport from here", "travel"),
    Exemplar("coffee shops nearby", "travel"),
    Exemplar("what's the air quality in Beijing", "travel"),
    # general
    Exemplar("hi", "general"),
    Exemplar("thanks!", "general"),
    Exemplar("write me a haiku about spring", "general"),
    Exemplar("what is the capital of France", "general"),
]
