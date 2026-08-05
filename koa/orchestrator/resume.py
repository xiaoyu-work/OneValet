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

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..streaming.models import AgentEvent, EventType
from .agent_tool import build_agent_hints
from .ask_mirror import parse_reply
from .transcript_store import (
    RunLeaseLost,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUSPENDED,
)

logger = logging.getLogger(__name__)

#: An answer to a yes/no question is short. Longer than this and the message
#: is a request in its own right, whatever words it happens to contain.
_MAX_REPLY_WORDS = 6

#: Slack on top of the longest a turn can take, before a quiet run is assumed
#: dead. Half an hour: long enough that nothing legitimate is mistaken for a
#: corpse, short enough that a real crash is recoverable the same day.
_RESUME_LEASE_MARGIN = 1800

#: The longest a single model request can take before the client gives up.
#: Streaming is the slower of the two ceilings in LLMConfig, so it is the one
#: a lease has to survive.
_LLM_REQUEST_CEILING = 120

#: How many times to wait for a busy run to finish before giving up on
#: continuing it, and how long each successive wait grows by. Together these
#: span several minutes -- longer than a turn, so a run that is merely
#: finishing is caught, while one that is genuinely wedged is left to the lease.
_RESUME_ATTEMPTS = 6
_RESUME_RETRY_DELAY = 20


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
    ) -> AsyncIterator[AgentEvent]:
        """Continue a suspended run from its durable transcript.

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
        claim_token = await store.claim(run_id, self._resume_lease_seconds())
        if not claim_token:
            logger.info(f"[Resume] Run {run_id} is already being continued; leaving it alone")
            return

        # The first read was only enough to decide whether claiming made
        # sense. Another process may have completed a round and handed the run
        # back between that read and our claim; continuing its older message
        # list would replay work the row now says is done. Read under the
        # fencing token and use only that copy.
        claimed = await store.get(run_id)
        if claimed is None or claimed.claim_token != claim_token:
            logger.warning(f"[Resume] Could not read the transcript claimed for run {run_id}")
            await store.release(run_id, claim_token)
            return
        transcript = claimed
        messages = list(transcript.messages)
        pending = store.unanswered_tool_calls(messages)

        # Two different things can be outstanding, and they are answered in
        # different currencies. A tool call the process died in the middle of
        # needs a tool result under its own id. A decision the user made needs
        # acting on and then telling, because the id it was recorded against
        # belongs to an inner loop this transcript has never seen.
        try:
            async for event in self._continue_claimed_run(
                run_id, transcript, messages, pending, claim_token
            ):
                yield event
        except BaseException:
            # The claim outlives us otherwise, and the run would sit unreachable
            # until the lease expired. Cancellation and shutdown come through
            # here too, which is exactly when a process stops without finishing.
            # Conditional, because the loop sets the run's final status before
            # this returns: a failure in the tail must not reopen a run that
            # actually completed.
            logger.warning(f"[Resume] Run {run_id} did not finish; releasing it")
            await store.release(run_id, claim_token)
            raise

    async def _continue_claimed_run(
        self,
        run_id: str,
        transcript: Any,
        messages: List[Dict[str, Any]],
        pending: List[Dict[str, Any]],
        claim_token: str,
    ) -> AsyncIterator[AgentEvent]:
        """Do the work of a resume, once this caller owns the run."""
        store = self._transcript_store

        if pending:
            self._answer_pending_calls(messages, pending)
        else:
            logger.info(f"[Resume] Run {run_id} has no pending tool calls; continuing loop")

        context = await self.prepare_context(
            transcript.tenant_id,
            transcript.user_message,
            transcript.metadata,
        )
        context["request_id"] = run_id
        context["resumed"] = True
        context["_claim_token"] = claim_token

        # Before the model gets a turn: the approved actions happen, from the
        # arguments the user saw, and their outcomes go in as turns it reads.
        decisions = await self._honour_decisions(run_id, transcript.tenant_id, context)
        for note in decisions:
            messages.append({"role": "user", "content": note})

        saved = await store.save(
            run_id,
            transcript.tenant_id,
            messages,
            user_message=transcript.user_message,
            metadata=transcript.metadata,
            turn=transcript.turn,
            status=STATUS_RUNNING,
            claim_token=claim_token,
        )
        if not saved:
            raise RunLeaseLost(f"Run {run_id} was taken over before it could continue")
        context["_transcript_owned"] = True

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
        try:
            await self.post_process(result, context)
        finally:
            # Everything this invocation held is released; only now is it safe
            # to let a continuation claim the run. In a finally because a
            # failure here does not make the user's decision any less pending.
            self.hand_off_unfinished(context)

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

        A live run says so at the top of every turn, so the window to cover is
        one turn: up to two model calls -- the summarizer that trims history,
        then the one that decides -- each with its retries, plus the tools that
        turn asked for. Tools run concurrently, so that term is the slowest
        single call rather than their sum.

        Derived from the timeouts it depends on, so raising one raises this.
        The margin is wide on purpose: being too generous means a crashed run
        waits longer before anyone takes it over, which nobody notices; being
        too tight means taking a run away from the process still working on it
        and running its tools a second time.
        """
        cfg = self._react_config
        attempts = cfg.llm_max_retries + 1
        per_call = _LLM_REQUEST_CEILING * attempts + cfg.llm_retry_base_delay * (2**attempts)
        return int(cfg.agent_tool_execution_timeout + 2 * per_call + _RESUME_LEASE_MARGIN)

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

        owners: List[str] = []
        for ask in asks:
            if not ask.awaits_execution:
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
            return []

        notes: List[str] = []
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

    async def resume_when_free(self, run_id: str) -> None:
        """Continue a run, waiting for it to finish if it is still going.

        The user is told about an ask while the run that raised it is still
        working -- raising one does not stop the run -- so a fast reply arrives
        to a run holding its own claim. Refusing outright would end it there.

        Waiting is the first attempt, not the guarantee. How long the run has
        left is not knowable from here, so when the attempts run out the run
        itself takes over: it ends knowing it owes something and hands the
        continuation on. This only has to cover the common case of a reply
        landing moments before the run finishes anyway.
        """
        for attempt in range(_RESUME_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_RESUME_RETRY_DELAY * attempt)

            if not await self._still_owed(run_id):
                logger.info(f"[Resume] Run {run_id} has nothing outstanding; nothing to continue")
                self._handoff_attempts.pop(run_id, None)
                return
            if await self._open_ask_count(run_id):
                # Another question is still unanswered. Waiting cannot change
                # that, and answering it will bring its own continuation.
                logger.info(f"[Resume] Run {run_id} is still waiting on the user")
                return

            produced = False
            try:
                async for _ in self.resume_run(run_id):
                    produced = True
            except Exception as e:
                # Worth another go: a transient failure is exactly what the
                # remaining attempts are for.
                logger.warning(f"[Resume] Attempt at run {run_id} failed: {e}")
                continue
            if produced:
                if not await self._still_owed(run_id):
                    self._handoff_attempts.pop(run_id, None)
                return
        logger.info(
            f"[Resume] Run {run_id} stayed busy; leaving it to hand itself on when it ends"
        )

    async def _still_owed(self, run_id: str) -> bool:
        """Whether this run has anything left to do for the user."""
        inbox = getattr(self, "inbox", None)
        if inbox is None or not inbox.enabled:
            return True
        try:
            return any(
                ask.is_open or ask.awaits_execution for ask in await inbox.for_run(run_id)
            )
        except Exception:
            return True

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
    ) -> None:
        """Close tool calls that the process died before completing.

        Each call must end up with a result, answered or not: an assistant
        message whose tool_calls have no replies is rejected outright by
        several providers, and would make the transcript unresumable.
        """
        for call in pending:
            call_id = call.get("id")
            if not call_id:
                continue
            name = (call.get("function") or {}).get("name", "tool")
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
