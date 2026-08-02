"""Resuming a run that stopped mid-flight.

A run stops before finishing for two reasons: the process died, or the agent
asked for something only a human can give. Either way the transcript records
exactly how far it got, and the tool calls with no matching result are the
work left to do.

Resuming replays that transcript with those calls answered -- from the user's
reply, or by re-executing them after a crash -- and hands control back to the
normal ReAct loop. It never re-runs a tool whose result is already recorded,
so a resume is safe to attempt more than once.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..streaming.models import AgentEvent, EventType
from .ask_mirror import parse_reply
from .transcript_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)

logger = logging.getLogger(__name__)

#: An answer to a yes/no question is short. Longer than this and the message
#: is a request in its own right, whatever words it happens to contain.
_MAX_REPLY_WORDS = 6


class ResumeMixin:
    """Mixin providing resumption of suspended runs.

    Expects on ``self`` (provided by Orchestrator):
    - ``_transcript_store``, ``inbox``
    - ``_react_loop_events()`` (ReactLoopMixin)
    - ``_build_tool_schemas_with_domain_fallback()``
    - ``_build_result_from_exec_data()``
    - ``post_process()``, ``prepare_context()``
    """

    async def resume_run(
        self,
        run_id: str,
        *,
        answers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Continue a suspended run, optionally answering what it was waiting on.

        ``answers`` maps a tool_call id to the result to record for it -- the
        user's decision, arriving long after the run that asked for it ended.
        Calls left unanswered are simply dropped with an explanatory result, so
        the model can see they did not happen and adapt.

        Yields the same event stream as a fresh request, so callers handle a
        resumed run exactly like a new one.
        """
        store = self._transcript_store
        if not store.enabled:
            logger.warning("[Resume] No transcript store configured")
            return

        transcript = await store.get(run_id)
        if transcript is None:
            logger.warning(f"[Resume] No transcript for run {run_id}")
            return

        if transcript.status in (STATUS_COMPLETED, STATUS_FAILED):
            logger.info(f"[Resume] Run {run_id} already {transcript.status}; nothing to do")
            return

        messages = list(transcript.messages)
        pending = store.unanswered_tool_calls(messages)

        # Answers that arrived through the Inbox since this run stopped.
        resolved, open_asks = await self._collect_inbox_answers(run_id)
        merged = {**resolved, **(answers or {})}

        if open_asks:
            # Waking now would answer the questions we have and record the rest
            # as "never ran", so the answers still coming would arrive to a run
            # that had already moved past them. One wake, once everything is in.
            logger.info(
                f"[Resume] Run {run_id} still waiting on {open_asks} ask(s); not resuming yet"
            )
            return

        if not pending:
            # Nothing was outstanding -- the run died between rounds rather
            # than mid-tool. Replaying the loop from here is still correct.
            logger.info(f"[Resume] Run {run_id} has no pending tool calls; continuing loop")
        else:
            self._answer_pending_calls(messages, pending, merged)

        await store.save(
            run_id,
            transcript.tenant_id,
            messages,
            user_message=transcript.user_message,
            metadata=transcript.metadata,
            turn=transcript.turn,
            status=STATUS_RUNNING,
        )

        context = await self.prepare_context(
            transcript.tenant_id,
            transcript.user_message,
            transcript.metadata,
        )
        context["request_id"] = run_id
        context["resumed"] = True

        intent = context.get("intent_analysis")
        domains = getattr(intent, "domains", None) or ["all"]
        tool_schemas = await self._build_tool_schemas_with_domain_fallback(
            transcript.tenant_id, domains=domains
        )

        logger.info(
            f"[Resume] Continuing run {run_id} "
            f"({len(messages)} messages, {len(pending)} answered)"
        )

        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages,
            tool_schemas,
            transcript.tenant_id,
            context=context,
            user_message=transcript.user_message,
            metadata=transcript.metadata,
            request_tools=list(getattr(self, "builtin_tools", [])),
        ):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data
            yield event

        result = self._build_result_from_exec_data(
            exec_data,
            final_response=exec_data.get("final_response", ""),
            context=context,
            total_tool_count=len(tool_schemas),
        )
        result.metadata["resumed_run_id"] = run_id
        await self.post_process(result, context)

    async def answer_from_message(
        self,
        tenant_id: str,
        message: str,
    ) -> Optional[str]:
        """Resolve a pending ask if this message answers one. Returns its run id.

        We ask people things on surfaces that have no buttons -- a text
        message, a pair of glasses -- and tell them to reply. "yes" arriving
        afterwards is an answer to that question, not a new instruction, and
        running it through the agent as a fresh request would produce a
        bewildered response instead of the action they just approved.

        Only when exactly one ask is open. With none there is nothing to
        answer; with several, a bare "yes" does not say which, so it is left
        for the user to answer explicitly.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return None

        text = (message or "").strip()
        # Count the words before touching the database. Almost every message
        # is a real request, and those must not pay for a query that could
        # only ever have matched a short reply.
        if not text or len(text.split()) > _MAX_REPLY_WORDS:
            return None

        try:
            open_asks = await inbox.pending(tenant_id, limit=2)
        except Exception as e:
            logger.warning(f"[Resume] Could not check pending asks for {tenant_id}: {e}")
            return None
        if len(open_asks) != 1:
            return None

        ask = open_asks[0]
        decision = parse_reply(text, ask.options)
        if decision is None:
            return None
        if not await inbox.resolve(ask.id, decision, resolved_by="reply"):
            return None
        logger.info(f"[Resume] Reply {decision!r} answered ask {ask.id} (run={ask.run_id})")
        return ask.run_id

    async def _collect_inbox_answers(self, run_id: str) -> Tuple[Dict[str, str], int]:
        """Answers given for this run's asks, and how many are still open.

        Answers are keyed by tool_call_id so they slot straight into the
        transcript. An approval becomes an instruction the model can act on
        rather than a bare "approve", since by the time it resumes the model
        has to re-decide what to do.

        The open count matters as much as the answers: a run with a question
        still outstanding is not ready to continue.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return {}, 0
        out: Dict[str, str] = {}
        open_count = 0
        for ask in await inbox.for_run(run_id):
            if ask.is_open:
                open_count += 1
                continue
            if not ask.resolution:
                continue
            tool_name = (ask.data or {}).get("tool", "the action")
            if ask.kind == "approval":
                if ask.resolution.strip().lower() in ("approve", "approved", "yes", "y"):
                    out[ask.tool_call_id] = (
                        f"The user approved this. Call {tool_name} again with the "
                        "same arguments to carry it out."
                    )
                else:
                    out[ask.tool_call_id] = (
                        f"The user declined this. Do not run {tool_name}; "
                        "continue without it and say so."
                    )
            else:
                out[ask.tool_call_id] = ask.resolution
        return out, open_count

    async def answer_ask(
        self,
        ask_id: str,
        resolution: str,
        *,
        resolved_by: str = "",
    ) -> Optional[str]:
        """Record a person's answer and return the run it unblocks, if any.

        Returns None when the ask was already answered elsewhere -- the caller
        should not resume a run on the strength of a lost race.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return None
        ask = await inbox.get(ask_id)
        if ask is None:
            return None
        if not await inbox.resolve(ask_id, resolution, resolved_by=resolved_by):
            return None
        return ask.run_id

    @staticmethod
    def _answer_pending_calls(
        messages: List[Dict[str, Any]],
        pending: List[Dict[str, Any]],
        answers: Dict[str, str],
    ) -> None:
        """Record a result for every tool call still waiting on one.

        Each call must end up with a result, answered or not: an assistant
        message whose tool_calls have no replies is rejected outright by
        several providers, and would make the transcript unresumable.
        """
        for call in pending:
            call_id = call.get("id")
            if not call_id:
                continue
            name = (call.get("function") or {}).get("name", "tool")
            if call_id in answers:
                content = answers[call_id]
            else:
                content = (
                    f"'{name}' was not completed -- the run was interrupted before it ran. "
                    "Continue without it, or try again if it is still needed."
                )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )

    async def list_resumable_runs(
        self,
        tenant_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Runs for this tenant that stopped before finishing."""
        return await self._transcript_store.list_resumable(tenant_id, limit)
