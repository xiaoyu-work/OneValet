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
from typing import Any, AsyncIterator, Dict, List, Optional

from ..streaming.models import AgentEvent, EventType
from .agent_tool import build_agent_hints
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

#: Slack on top of the longest a turn can take, before a quiet run is assumed
#: dead. Half an hour: long enough that nothing legitimate is mistaken for a
#: corpse, short enough that a real crash is recoverable the same day.
_RESUME_LEASE_MARGIN = 1800


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

        # A run wakes once, when nothing is left for the user to answer.
        open_asks = await self._open_ask_count(run_id)
        if open_asks:
            # Waking now would act on the answers we have and record the rest
            # as "never ran", so the answers still coming would arrive to a run
            # that had already moved past them. One wake, once everything is in.
            logger.info(
                f"[Resume] Run {run_id} still waiting on {open_asks} ask(s); not resuming yet"
            )
            return

        # And only one continuation at a time. Answers can land together and a
        # resume can be asked for by hand, and two loops replaying the same
        # transcript would repeat each other's work and overwrite the record.
        if not await store.claim(run_id, self._resume_lease_seconds()):
            logger.info(f"[Resume] Run {run_id} is already being continued; leaving it alone")
            return

        # Two different things can be outstanding, and they are answered in
        # different currencies. A tool call the process died in the middle of
        # needs a tool result under its own id. A decision the user made needs
        # acting on and then telling, because the id it was recorded against
        # belongs to an inner loop this transcript has never seen.
        if pending:
            self._answer_pending_calls(messages, pending, answers or {})
        else:
            logger.info(f"[Resume] Run {run_id} has no pending tool calls; continuing loop")

        context = await self.prepare_context(
            transcript.tenant_id,
            transcript.user_message,
            transcript.metadata,
        )
        context["request_id"] = run_id
        context["resumed"] = True

        # Before the model gets a turn: the approved actions happen, from the
        # arguments the user saw, and their outcomes go in as turns it reads.
        decisions = await self._honour_decisions(run_id, transcript.tenant_id, context)
        for note in decisions:
            messages.append({"role": "user", "content": note})

        await store.save(
            run_id,
            transcript.tenant_id,
            messages,
            user_message=transcript.user_message,
            metadata=transcript.metadata,
            turn=transcript.turn,
            status=STATUS_RUNNING,
        )

        intent = context.get("intent_analysis")
        domains = getattr(intent, "domains", None) or ["all"]
        tool_schemas = await self._build_tool_schemas_with_domain_fallback(
            transcript.tenant_id, domains=domains
        )

        logger.info(
            f"[Resume] Continuing run {run_id} ({len(messages)} messages, "
            f"{len(pending)} tool call(s) answered, {len(decisions)} decision(s) applied)"
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

    def _resume_lease_seconds(self) -> int:
        """How long a run may go quiet before it counts as abandoned.

        A live run writes its transcript after each round of tools, so the
        longest it can legitimately be silent is one round: a model call with
        its retries, then the tools that round asked for. Both are bounded by
        configuration, so the lease is derived from them rather than picked --
        raise a timeout and the lease follows.

        The margin is deliberately wide. Being too generous means a crashed
        run waits longer before anyone can take it over; being too tight means
        taking over a run that is still working, and running its tools twice.
        """
        cfg = self._react_config
        llm_budget = (cfg.llm_max_retries + 1) * cfg.llm_retry_base_delay * 60
        return int(cfg.agent_tool_execution_timeout + llm_budget + _RESUME_LEASE_MARGIN)

    def _agent_type_for_class(self, class_name: str) -> Optional[str]:
        """The registry name for an agent, given the class that recorded an ask.

        Asks written before the registry name was recorded carry only the
        Python class name, which the registry does not answer to. Without this
        those decisions could never be routed, and a decision that cannot be
        routed strands its run for good.
        """
        registry = getattr(self, "_agent_registry", None)
        if registry is None or not class_name:
            return None
        try:
            for name in registry.get_all_agent_names():
                cls = registry.get_agent_class(name)
                if cls is not None and cls.__name__ == class_name:
                    return name
        except Exception as e:
            logger.warning(f"[Resume] Could not resolve agent class {class_name}: {e}")
        return None

    async def _honour_decisions(
        self,
        run_id: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Carry out every decision this run is still owed, and describe them.

        Done here rather than left to the replayed loop because the loop
        cannot be relied on: the model would have to choose, unprompted, to go
        back to the agent that owns the action. When it does not, the user's
        answer is never acted on and nothing notices -- the ask is answered,
        so no amount of waking the run again would help.

        Each owning agent is asked directly instead, built with the same
        context the ReAct loop would have given it, because the approved tool
        is the same tool and needs the same database and settings.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return []
        try:
            asks = await inbox.for_run(run_id)
        except Exception as e:
            logger.warning(f"[Resume] Could not read decisions for run {run_id}: {e}")
            return []

        notes_direct: List[str] = []
        owners: List[str] = []
        for ask in asks:
            if not ask.awaits_execution:
                continue
            if ask.kind != "approval":
                # Nothing to carry out, but the answer is still owed a place in
                # the run. Left unstamped it would keep the run listed as owing
                # something with nothing able to clear it.
                if await inbox.claim_execution(ask.id):
                    notes_direct.append(f"The user answered: {ask.resolution}")
                continue
            data = ask.data or {}
            owner = data.get("agent_type") or self._agent_type_for_class(data.get("agent", ""))
            if owner and owner not in owners:
                owners.append(owner)
            elif not owner:
                logger.error(
                    f"[Resume] Ask {ask.id} names no agent that can carry it out; "
                    "the user's decision cannot be honoured"
                )
        if not owners:
            return notes_direct

        notes: List[str] = list(notes_direct)
        for agent_type in owners:
            hints = build_agent_hints(
                self,
                agent_type,
                tenant_id,
                request_context=dict(context or {}, resumed=True, request_id=run_id),
            )
            agent = await self.create_agent(tenant_id, agent_type, context_hints=hints)
            if agent is None:
                logger.warning(f"[Resume] Cannot reach {agent_type} to honour its approvals")
                continue
            try:
                notes.extend(await agent._carry_out_approved_actions())
            except Exception as e:
                logger.error(f"[Resume] {agent_type} failed honouring approvals: {e}")
            finally:
                # Created only to honour these decisions; leaving it in the
                # pool would count against the tenant's agent limit and could
                # be picked up as a live conversation.
                await self.agent_pool.remove_agent(tenant_id, agent.agent_id)
        return notes

    async def _open_ask_count(self, run_id: str) -> int:
        """How many of this run's questions the user has not answered yet.

        A run wakes once, when the count reaches zero. Waking earlier would
        act on the answers in hand and record the rest as never having run, so
        the answers still on their way would arrive to a run that had already
        moved past them.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return 0
        try:
            return sum(1 for ask in await inbox.for_run(run_id) if ask.is_open)
        except Exception as e:
            # Resuming on a guess could act on a decision the user has not
            # made, so an unreadable Inbox holds the run rather than waking it.
            logger.warning(f"[Resume] Could not count open asks for {run_id}: {e}")
            return 1

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

        Raises ValueError on an answer that is neither one of the ask's own
        options nor a recognisable yes or no. Anything we do not understand
        counts as a refusal further down, so accepting it silently would let a
        client saying "allow" have the action recorded as declined.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return None
        ask = await inbox.get(ask_id)
        if ask is None:
            return None

        decided = parse_reply(resolution, ask.options)
        if decided is None:
            raise ValueError(
                f"{resolution!r} is not an answer to this question. "
                f"Expected one of: {', '.join(ask.options) or 'approve, reject'}"
            )
        if not await inbox.resolve(ask_id, decided, resolved_by=resolved_by):
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
        """Runs for this tenant that stopped before finishing.

        Includes runs holding a decision the user made that was never acted
        on. Those are invisible to every other path -- both ways of waking a
        run begin with answering a question that is still open, and theirs has
        been answered -- so without this they would sit forever.
        """
        runs = await self._transcript_store.list_resumable(tenant_id, limit)
        seen = {r.get("run_id") for r in runs}

        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return runs
        for run_id in await inbox.runs_awaiting_execution(tenant_id, limit):
            if run_id not in seen:
                runs.append({"run_id": run_id, "status": "awaiting_action"})
        return runs
