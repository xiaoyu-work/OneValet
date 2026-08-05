"""DAG execution mixin — runs multi-intent requests as a dependency graph.

When the intent analyzer splits a request into independent sub-tasks, they run
here: topologically sorted into levels, each level executed concurrently, and
the results synthesized into a single answer.

Each sub-task gets a deep-copied context and its own ReAct loop, so two
sub-tasks in the same level cannot observe each other's state.
"""

import asyncio
import copy
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

from ..result import AgentResult, AgentStatus
from ..streaming.models import AgentEvent, EventType
from .dag_executor import (
    SubTaskResult,
    aggregate_token_usage,
    get_runnable_tasks,
    topological_sort,
)

if TYPE_CHECKING:
    from .intent_analyzer import IntentAnalysis, SubTask

logger = logging.getLogger(__name__)

#: Joins a run id to a sub-task id. Unreserved in a URL path (RFC 3986), so
#: the composite survives a round trip through the resume route intact.
_SUB_RUN_SEPARATOR = "~"


def _sub_run_id(context: Optional[Dict[str, Any]], sub_task_id: str) -> str:
    """A transcript id of a sub-task's own.

    Sub-tasks each build their own message list and run their own loop, but
    they used to persist under the request's id -- one row, several writers,
    last one wins. What survived was one sub-task's context standing in for
    the whole run, and a status that flapped between running and finished
    while siblings were still going.

    Giving each its own id makes the transcript describe the work that
    produced it, which is what a resume needs: an approval raised inside a
    sub-task comes back to that sub-task's messages, not to whichever sibling
    happened to write last.

    The separator is unreserved in a URL path. These ids reach clients -- in
    the Inbox listing and the resumable listing -- and come back in the path
    of the resume route, so a character with meaning in a URL would be cut off
    before the request was even sent, and the run would look like the parent.
    """
    return f"{(context or {}).get('request_id', 'run')}{_SUB_RUN_SEPARATOR}{sub_task_id}"


class DagLoopMixin:
    """Mixin providing multi-intent DAG execution.

    Expects the following on ``self`` (provided by Orchestrator):
    - ``llm_client``
    - ``_react_config``
    - ``_react_loop_events()`` (ReactLoopMixin)
    - ``_build_llm_messages()`` (Orchestrator)
    - ``_build_tool_schemas_with_domain_fallback()`` (Orchestrator)
    """

    async def _execute_dag(
        self,
        intent: "IntentAnalysis",
        tenant_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute multi-intent sub-tasks in DAG order.

        Delegates to ``_stream_dag`` (the single implementation) and
        silently consumes its events, mirroring the pattern used by
        ``handle_message`` with ``_react_loop_events``.
        """
        exec_data: Dict[str, Any] = {}
        async for event in self._stream_dag(intent, tenant_id, context, metadata):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data

        final_response = exec_data.get("final_response", "")
        pending_approvals = exec_data.get("pending_approvals", [])

        status = AgentStatus.COMPLETED
        if pending_approvals:
            status = AgentStatus.WAITING_FOR_APPROVAL

        return AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
            metadata={
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "levels": exec_data.get("levels", 0),
                "duration_ms": exec_data.get("duration_ms", 0),
                "token_usage": exec_data.get("token_usage", {}),
                "pending_approvals": pending_approvals,
            },
        )

    async def _stream_dag(
        self,
        intent: "IntentAnalysis",
        tenant_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream events during DAG execution.

        This is the single DAG implementation.  ``_execute_dag`` consumes
        this generator silently, so all fixes live here only.

        Fixes applied:
        - Skip sub-tasks whose dependencies failed (get_runnable_tasks)
        - Propagate pending_approvals from sub-task ReAct loops
        - Aggregate token usage, turns, and tool_calls across sub-tasks
        - DAG-level timeout guard
        - Collect and yield events from parallel sub-tasks
        """
        start_time = time.monotonic()
        levels = topological_sort(intent.sub_tasks)
        all_results: Dict[int, SubTaskResult] = {}
        all_pending_approvals: list = []
        total_turns = 0
        total_tool_calls = 0

        # DAG-level budget: independent of max_turns (which is a model-iteration
        # count, not a time unit).
        deadline = start_time + self._react_config.dag_timeout

        yield AgentEvent(
            type=EventType.WORKFLOW_START,
            data={"sub_tasks": len(intent.sub_tasks), "levels": len(levels)},
        )

        for level_idx, level in enumerate(levels):
            # Fix 5: check DAG-level timeout before each level
            if time.monotonic() > deadline:
                logger.warning(
                    f"[DAG] Timeout exceeded ({self._react_config.dag_timeout}s) "
                    f"at level {level_idx}"
                )
                for st in level:
                    all_results[st.id] = SubTaskResult(
                        sub_task_id=st.id,
                        description=st.description,
                        response="Skipped: DAG timeout exceeded",
                        status="skipped",
                    )
                # Also mark tasks in remaining levels
                for remaining_level in levels[level_idx + 1 :]:
                    for st in remaining_level:
                        all_results[st.id] = SubTaskResult(
                            sub_task_id=st.id,
                            description=st.description,
                            response="Skipped: DAG timeout exceeded",
                            status="skipped",
                        )
                break

            # Fix 1: split level into runnable / skipped tasks
            runnable, skipped = get_runnable_tasks(level, all_results)
            for st in skipped:
                all_results[st.id] = SubTaskResult(
                    sub_task_id=st.id,
                    description=st.description,
                    response="Skipped: dependency failed",
                    status="skipped",
                )
                yield AgentEvent(
                    type=EventType.STAGE_START,
                    data={
                        "sub_task_id": st.id,
                        "description": st.description,
                        "domain": st.domain,
                    },
                )
                yield AgentEvent(
                    type=EventType.STAGE_END,
                    data={"sub_task_id": st.id, "status": "skipped"},
                )

            if not runnable:
                continue

            for st in runnable:
                yield AgentEvent(
                    type=EventType.STAGE_START,
                    data={
                        "sub_task_id": st.id,
                        "description": st.description,
                        "domain": st.domain,
                    },
                )

            if len(runnable) == 1:
                # Single task in level: stream events in real-time
                st = runnable[0]
                augmented_message = self._build_dag_augmented_message(
                    st,
                    all_results,
                )
                tool_schemas = await self._build_tool_schemas_with_domain_fallback(
                    tenant_id,
                    domains=[st.domain],
                )
                task_context = copy.deepcopy(context)
                task_context["request_id"] = _sub_run_id(context, st.id)
                messages = await self._build_llm_messages(task_context, augmented_message)

                exec_data: Dict[str, Any] = {}
                try:
                    async for event in self._react_loop_events(
                        messages,
                        tool_schemas,
                        tenant_id,
                        context=task_context,
                        user_message=augmented_message,
                        metadata=metadata,
                    ):
                        if event.type == EventType.EXECUTION_END:
                            exec_data = event.data
                        yield event
                except BaseException:
                    await self._fail_transcript(task_context)
                    raise
                finally:
                    # This sub-task owns its own transcript, so it is the one
                    # that has to pass on anything the user answered while it
                    # ran -- including when it is failing or being abandoned,
                    # since the decision is no less real for that.
                    self.hand_off_unfinished(task_context)

                # Fix 2: collect pending_approvals from sub-task
                sub_approvals = exec_data.get("pending_approvals", [])
                if sub_approvals:
                    all_pending_approvals.extend(sub_approvals)

                # Fix 3: accumulate turns / tool_calls
                total_turns += exec_data.get("turns", 0)
                total_tool_calls += exec_data.get("tool_calls_count", 0)

                waiting = await self._owes_the_user(task_context["request_id"])
                all_results[st.id] = SubTaskResult(
                    sub_task_id=st.id,
                    description=st.description,
                    response=exec_data.get("final_response", ""),
                    status="waiting" if waiting else "completed",
                    duration_ms=exec_data.get("duration_ms", 0),
                    token_usage=exec_data.get("token_usage", {}),
                )
                yield AgentEvent(
                    type=EventType.STAGE_END,
                    data={"sub_task_id": st.id},
                )
            else:
                # Multiple parallel tasks — each gets an isolated context
                # manager and a deepcopy of context to prevent shared-state
                # race conditions across concurrent sub-tasks.
                _agent_pool_lock = asyncio.Lock()

                async def _run_collecting(sub_task):
                    aug_msg = self._build_dag_augmented_message(
                        sub_task,
                        all_results,
                    )
                    t_schemas = await self._build_tool_schemas_with_domain_fallback(
                        tenant_id,
                        domains=[sub_task.domain],
                    )
                    # Fully isolated context per sub-task
                    task_context = copy.deepcopy(context)
                    task_context["request_id"] = _sub_run_id(context, sub_task.id)
                    msgs = await self._build_llm_messages(task_context, aug_msg)
                    exec_d: Dict[str, Any] = {}
                    events: list = []
                    try:
                        async for ev in self._react_loop_events(
                            msgs,
                            t_schemas,
                            tenant_id,
                            context=task_context,
                            user_message=aug_msg,
                            metadata=metadata,
                        ):
                            if ev.type == EventType.EXECUTION_END:
                                exec_d = ev.data
                            else:
                                events.append(ev)
                    except BaseException:
                        await self._fail_transcript(task_context)
                        raise
                    finally:
                        # Gathered with return_exceptions=True, so a sub-task
                        # that fails here is swallowed by its siblings. The
                        # handoff has to survive that or the decision is lost
                        # with nothing said about it.
                        self.hand_off_unfinished(task_context)

                    waiting = await self._owes_the_user(task_context["request_id"])
                    sub_result = SubTaskResult(
                        sub_task_id=sub_task.id,
                        description=sub_task.description,
                        response=exec_d.get("final_response", ""),
                        status="waiting" if waiting else "completed",
                        duration_ms=exec_d.get("duration_ms", 0),
                        token_usage=exec_d.get("token_usage", {}),
                    )
                    return sub_result, events, exec_d

                level_results = await asyncio.gather(
                    *[_run_collecting(st) for st in runnable],
                    return_exceptions=True,
                )

                for st, result in zip(runnable, level_results):
                    if isinstance(result, BaseException):
                        logger.warning(f"[DAG] Sub-task {st.id} failed: {result}")
                        all_results[st.id] = SubTaskResult(
                            sub_task_id=st.id,
                            description=st.description,
                            response=f"Error: {result}",
                            status="error",
                        )
                    else:
                        sub_result, events, exec_d = result
                        # Fix 6: yield collected events
                        for ev in events:
                            yield ev
                        # Fix 2: collect pending_approvals
                        sub_approvals = exec_d.get("pending_approvals", [])
                        if sub_approvals:
                            all_pending_approvals.extend(sub_approvals)
                        # Fix 3: accumulate turns / tool_calls
                        total_turns += exec_d.get("turns", 0)
                        total_tool_calls += exec_d.get("tool_calls_count", 0)
                        all_results[st.id] = sub_result
                    yield AgentEvent(
                        type=EventType.STAGE_END,
                        data={"sub_task_id": st.id},
                    )

        # Synthesis stage
        yield AgentEvent(
            type=EventType.STAGE_START,
            data={"sub_task_id": -1, "description": "Synthesizing results"},
        )
        final_response = await self._synthesize_dag_results(
            intent.raw_message,
            all_results,
            context,
        )
        yield AgentEvent(type=EventType.MESSAGE_START, data={})
        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
        yield AgentEvent(type=EventType.MESSAGE_END, data={})
        yield AgentEvent(
            type=EventType.STAGE_END,
            data={"sub_task_id": -1},
        )

        yield AgentEvent(
            type=EventType.WORKFLOW_END,
            data={"sub_tasks_completed": len(all_results)},
        )

        # Fix 3: aggregate token usage across all sub-tasks
        aggregated_usage = aggregate_token_usage(all_results)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "final_response": final_response,
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "levels": len(levels),
                "pending_approvals": all_pending_approvals,
                "result_status": (
                    "WAITING_FOR_APPROVAL"
                    if any(result.status == "waiting" for result in all_results.values())
                    else None
                ),
                "turns": total_turns,
                "token_usage": aggregated_usage,
                "duration_ms": duration_ms,
                "tool_calls_count": total_tool_calls,
                "tool_calls": [],
            },
        )

    async def _synthesize_dag_results(
        self,
        original_message: str,
        results: Dict[int, "SubTaskResult"],
        context: Dict[str, Any],
    ) -> str:
        """Synthesize multiple sub-task results into a unified response.

        Includes user profile and language context so the synthesis
        matches the user's communication style and language preference.
        """
        # If this truly was a one-task DAG, avoid an unnecessary synthesis call.
        # A single success alongside waiting/skipped work is not the whole
        # answer and must not hide what remains incomplete.
        successful = [r for r in results.values() if r.status == "completed"]
        if len(results) == 1 and len(successful) == 1:
            return successful[0].response

        result_parts = []
        for sub_id in sorted(results.keys()):
            r = results[sub_id]
            result_parts.append(
                f"## Sub-task {sub_id}: {r.description} "
                f"[status: {r.status}]\n{r.response}"
            )

        synthesis_message = (
            f'The user asked: "{original_message}"\n\n'
            "Here are the results from each sub-task:\n\n"
            + "\n\n".join(result_parts)
            + "\n\nSynthesize these into a single, coherent response for the user. "
            "Preserve all specific data points. Clearly say which work is "
            "waiting or skipped; never present it as completed. Be concise."
        )

        # Build a context-aware system prompt for synthesis
        synthesis_system_parts = [
            "You synthesize multiple task results into a unified response.",
        ]

        # Inject user profile if available for personalized tone
        user_profile = context.get("user_profile")
        if user_profile:
            profile_str = user_profile if isinstance(user_profile, str) else str(user_profile)
            if len(profile_str) < 500:
                synthesis_system_parts.append(f"\n[User Profile]\n{profile_str}")

        # Inject language preference so synthesis matches user's language
        language = context.get("language") or context.get("locale")
        if language:
            synthesis_system_parts.append(
                f"\nRespond in the same language as the user's original message. "
                f"User locale hint: {language}"
            )
        else:
            synthesis_system_parts.append(
                "\nRespond in the same language as the user's original message."
            )

        messages = [
            {
                "role": "system",
                "content": "\n".join(synthesis_system_parts),
            },
            {"role": "user", "content": synthesis_message},
        ]

        try:
            response = await self.llm_client.chat_completion(messages=messages)
            return response.content or ""
        except Exception as e:
            logger.warning(f"[DAG] Synthesis failed: {e}")
            # Fallback: concatenate results
            return "\n\n".join(
                f"**{r.description}:**\n{r.response}"
                for r in sorted(results.values(), key=lambda r: r.sub_task_id)
            )

    @staticmethod
    def _build_dag_augmented_message(
        sub_task: "SubTask",
        prior_results: Dict[int, "SubTaskResult"],
    ) -> str:
        """Build an augmented user message with predecessor results injected."""
        if not sub_task.depends_on:
            return sub_task.description

        predecessor_context = []
        for dep_id in sub_task.depends_on:
            if dep_id in prior_results:
                pred = prior_results[dep_id]
                predecessor_context.append(
                    f"[Result from previous step: {pred.description}]\n{pred.response}"
                )

        if not predecessor_context:
            return sub_task.description

        return (
            "\n\n".join(predecessor_context)
            + f"\n\nBased on the above, please: {sub_task.description}"
        )
