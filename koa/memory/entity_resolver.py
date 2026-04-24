"""Entity resolution — map free-text mentions to records in ``entities`` /
``device_contacts``.

Used by EmailAgent, SlackAgent, TodoAgent to turn "remind Sarah" into a
canonical contact (phone/email) so downstream tools don't need a fuzzy match.

Phase-3 stub.  Strategy:
  1. Exact match on name (case-insensitive) in entities where kind='person'.
  2. Fallback to device_contacts with trigram similarity.
  3. If multiple candidates, return all with confidence scores and let the
     caller's LLM disambiguate via a follow-up question.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


async def resolve_person(
    db,
    user_id: str,
    mention: str,
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Return ranked candidate person entities for a text mention.

    Each candidate: { entity_id, display_name, emails, phones, confidence }.
    """
    mention = (mention or "").strip()
    if not mention:
        return []

    # Step 1: exact entity match
    rows = await db.fetch(
        """SELECT id, display_name, aliases, metadata
           FROM entities
           WHERE user_id = $1 AND kind = 'person'
             AND (LOWER(display_name) = LOWER($2) OR LOWER($2) = ANY (
                 SELECT LOWER(a) FROM unnest(COALESCE(aliases, ARRAY[]::text[])) a
             ))
           LIMIT $3""",
        user_id, mention, limit,
    )
    results = [
        {
            "entity_id": r["id"],
            "display_name": r["display_name"],
            "emails": (r["metadata"] or {}).get("emails", []),
            "phones": (r["metadata"] or {}).get("phones", []),
            "confidence": 0.95,
        }
        for r in rows
    ]
    if len(results) >= limit:
        return results

    # Step 2: fuzzy against device contacts (ILIKE fallback; trigram optional).
    try:
        contact_rows = await db.fetch(
            """SELECT id, display_name, phones, emails
               FROM device_contacts
               WHERE user_id = $1 AND display_name ILIKE $2
               ORDER BY length(display_name) ASC
               LIMIT $3""",
            user_id, f"%{mention}%", limit - len(results),
        )
    except Exception:
        contact_rows = []
    for r in contact_rows:
        results.append({
            "entity_id": None,
            "display_name": r["display_name"],
            "emails": r["emails"] or [],
            "phones": r["phones"] or [],
            "confidence": 0.6,
        })
    return results[:limit]


# TODO(phase-3): add a "write-back" that creates an entities row the first
# time a contact is actually used via Koi (learning from usage).
