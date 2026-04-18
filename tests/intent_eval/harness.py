"""Shared utilities for intent eval fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class IntentFixture:
    message: str
    expected_domains: List[List[str]]  # any-of
    expected_type: str = "single"
    expected_clarify: bool = False
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    source_file: str = ""


def load_fixtures(
    tag_filter: Optional[str] = None,
) -> List[IntentFixture]:
    """Load every ``*.jsonl`` fixture file, optionally filtering by tag."""
    fixtures: List[IntentFixture] = []
    if not FIXTURES_DIR.exists():
        return fixtures
    for path in sorted(FIXTURES_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                fx = IntentFixture(
                    message=data["message"],
                    expected_domains=data["expected_domains"],
                    expected_type=data.get("expected_type", "single"),
                    expected_clarify=bool(data.get("expected_clarify", False)),
                    tags=list(data.get("tags", []) or []),
                    notes=data.get("notes", ""),
                    source_file=f"{path.name}:{lineno}",
                )
                if tag_filter is None or tag_filter in fx.tags:
                    fixtures.append(fx)
    return fixtures


def domain_match(actual: List[str], accepted: List[List[str]]) -> bool:
    """True iff ``actual`` exactly matches any of the accepted sets."""
    actual_set = set(actual)
    for option in accepted:
        if actual_set == set(option):
            return True
    return False


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fixture results into a dashboard-style summary."""
    total = len(results)
    if total == 0:
        return {"total": 0}
    correct = sum(1 for r in results if r["correct"])
    by_tag: Dict[str, Dict[str, int]] = {}
    for r in results:
        for tag in r.get("tags", []):
            agg = by_tag.setdefault(tag, {"total": 0, "correct": 0})
            agg["total"] += 1
            if r["correct"]:
                agg["correct"] += 1
    # Confusion matrix: expected-domain -> predicted-domain
    confusion: Dict[str, Dict[str, int]] = {}
    for r in results:
        exp = ",".join(sorted(r["expected_domains"][0])) if r["expected_domains"] else "?"
        pred = ",".join(sorted(r["predicted_domains"])) if r["predicted_domains"] else "?"
        confusion.setdefault(exp, {}).setdefault(pred, 0)
        confusion[exp][pred] += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "by_tag": {
            tag: {
                "accuracy": v["correct"] / v["total"],
                "total": v["total"],
            }
            for tag, v in sorted(by_tag.items())
        },
        "confusion": confusion,
    }


def iter_fixtures_by_tag(tag: str) -> Iterator[IntentFixture]:
    for fx in load_fixtures():
        if tag in fx.tags:
            yield fx
