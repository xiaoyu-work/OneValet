"""Delivering asks to wherever the user already is.

An ask that only exists in the database helps nobody. This mirrors it to the
channels the user has -- SMS, push, an app webhook -- so the question reaches
them without their having to come looking.

Delivery is best-effort and never blocks the run: the ask is already durable,
so a failed text means the user sees it next time they open the app rather
than the request being lost. Channels are tried in order and stop at the first
success, so a user with both push and SMS gets one message, not two.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Keep mirrored text short -- an SMS is 160 characters and a lock-screen
#: notification shows less than that.
_MAX_BODY = 140


def format_ask(ask: Any) -> str:
    """Render an ask as a single message a person can act on.

    Leads with what is being asked rather than which agent asked, and states
    how to answer, since a text message has no buttons.
    """
    title = (ask.title or "").strip()
    body = (ask.body or "").strip()

    if body and len(body) > _MAX_BODY:
        body = body[: _MAX_BODY - 1].rstrip() + "…"

    parts = [p for p in (title, body) if p]
    text = "\n".join(parts) if parts else "Koa needs your approval."

    if ask.options:
        text += f"\n\nReply {' or '.join(ask.options)}."
    expires_at = getattr(ask, "expires_at", None)
    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        remaining = max(0, (expires_at - datetime.now(timezone.utc)).total_seconds())
        if remaining < 2 * 24 * 60 * 60:
            text += f" Expires in {max(1, math.ceil(remaining / 3600))}h."
        else:
            text += f" Expires in {math.ceil(remaining / 86400)}d."
    return text


def parse_reply(text: str, options: Optional[List[str]] = None) -> Optional[str]:
    """Map a free-text reply onto one of an ask's options.

    People answer a text message with "yes", "ok", "go ahead", or "no" rather
    than the literal option word, so accept the obvious synonyms. Returns None
    when the reply is not recognisable as a decision -- better to leave the ask
    open than to act on a guess.

    Deliberately absent from the synonyms: "stop", "cancel", "don't". Those
    are how people interrupt an assistant, not how they answer a question that
    may have been raised by a background job days ago. Reading them as a
    refusal would consume a real instruction and record a decision the user
    did not make, and the record cannot be taken back.
    """
    if not text:
        return None
    reply = text.strip().lower()
    opts = [o.lower() for o in (options or [])]

    # An exact option always wins.
    for o in opts:
        if reply == o:
            return o

    affirmative = {"yes", "y", "ok", "okay", "sure", "approve", "approved",
                   "go ahead", "do it", "confirm", "yep", "yeah", "是", "好", "确认"}
    negative = {"no", "n", "nope", "reject", "rejected", "deny",
                "no thanks", "不", "不要"}

    if reply in affirmative:
        return "approve" if "approve" in opts or not opts else opts[0]
    if reply in negative:
        return "reject" if "reject" in opts or not opts else (opts[-1] if opts else "reject")

    # A reply that merely starts with an option ("approve it", "no thanks").
    for o in opts:
        if reply.startswith(o):
            return o
    return None


def parse_explicit_reply(text: str, options: Optional[List[str]] = None) -> Optional[str]:
    """Parse a tenant-wide chat reply without guessing its intent.

    A chat message is not tied to an ask id. Even when the tenant has exactly
    one ask open, "ok", "yes", or "sure" can be a response to the conversation
    happening now rather than permission for a background action raised days
    ago. The mirrored prompt explicitly says "Reply approve or reject", so the
    chat path requires one of those option words. Surfaces that identify the
    ask directly may use the friendlier synonym parser above.
    """
    reply = (text or "").strip().lower()
    for option in (options or []):
        canonical = option.lower()
        if reply == canonical or reply.startswith(canonical + " "):
            return canonical
    return None


class AskMirror:
    """Fans an ask out to the user's channels.

    Args:
        channels: Objects with ``async send(tenant_id, message, metadata)``,
            in priority order. The first success wins.
    """

    def __init__(self, channels: Optional[List[Any]] = None) -> None:
        self._channels = list(channels or [])

    @property
    def enabled(self) -> bool:
        return bool(self._channels)

    def add_channel(self, channel: Any) -> None:
        if channel is not None:
            self._channels.append(channel)

    async def mirror(self, ask: Any) -> bool:
        """Deliver an ask. Returns True once a channel accepts it.

        Failures are logged rather than raised: the ask is already durable, so
        an undelivered notification degrades to the user finding it in the app
        instead of the run being lost.
        """
        if not self._channels:
            return False

        message = format_ask(ask)
        metadata: Dict[str, Any] = {
            "category": "approval",
            "priority": "high",
            "ask_id": ask.id,
            "run_id": ask.run_id,
            "options": list(ask.options or []),
        }

        for channel in self._channels:
            name = type(channel).__name__
            try:
                if await channel.send(ask.tenant_id, message, metadata):
                    logger.info(f"[Inbox] Ask {ask.id} delivered via {name}")
                    return True
                logger.debug(f"[Inbox] {name} declined ask {ask.id}")
            except Exception as e:
                logger.warning(f"[Inbox] {name} failed to deliver ask {ask.id}: {e}")

        logger.warning(
            f"[Inbox] Ask {ask.id} not delivered on any channel; "
            "it remains pending in the Inbox"
        )
        return False
