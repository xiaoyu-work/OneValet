# OneValet ReAct Orchestrator — Implementation Plan

> Based on: `docs/design-react-orchestrator.md`
> Design principle: No ABC interfaces. Pick the technology, write the implementation directly. One class per concern.

---

## Phase 1: ReAct Loop + Agent-Tool + Context Management + Approval (Design §2-§7, §10-§14)

**Goal**: Complete ReAct loop with concurrent execution, error recovery, context management, structured approval, and streaming. One integrated system.

---

### Step 1.1 — ReactLoopConfig + Data Types (§14, §3.1)

**New file**: `onevalet/orchestrator/react_config.py`

**ReactLoopConfig** — per §14:

- **Loop control**: `max_turns`
- **Tool execution**: `tool_execution_timeout`, `agent_tool_execution_timeout`, `max_tool_result_share`, `max_tool_result_chars`
- **Context management**: `context_token_limit`, `context_trim_threshold`, `max_history_messages`
- **LLM calls**: `llm_max_retries`, `llm_retry_base_delay`
- **Approval**: `approval_timeout_minutes`

**ReactLoopResult** — per §3.1: response, turns, tool_calls, token_usage, duration_ms, pending_approvals

**ToolCallRecord** — per §3.1: name, args_summary, duration_ms, success, result_status (Agent-Tool only: COMPLETED/WAITING_FOR_INPUT/WAITING_FOR_APPROVAL/ERROR), result_chars, token_attribution

**TokenUsage** — input_tokens, output_tokens, total

**Depends on**: Nothing.

---

### Step 1.2 — ContextManager (§12)

**New file**: `onevalet/orchestrator/context_manager.py`

Full three-line-of-defense system per §12.2:

- **Defense 1 — truncate_tool_result()**: Called immediately after each tool execution. max_chars = min(context_token_limit * max_tool_result_share * 4, max_tool_result_chars). Truncate at newline boundary if possible, append `[...truncated]` marker.
- **Defense 2 — trim_if_needed()**: Called before each loop iteration. Triggered when estimate_tokens(messages) exceeds context_token_limit * context_trim_threshold. Keep system prompt + most recent max_history_messages messages.
- **Defense 3 — force_trim()**: Triggered after context overflow error. Keep system prompt + most recent 5 messages.
- **truncate_all_tool_results()**: Walk all tool_result messages, apply truncate_tool_result to each. Step 2 of the context overflow recovery chain.
- **estimate_tokens()**: ~4 chars/token approximation.

**Depends on**: Step 1.1 (ReactLoopConfig).

---

### Step 1.3 — Agent-Tool Schema Auto-Generation + Enhancement (§4)

**Modify**: `onevalet/agents/decorator.py`

`@valet` decorator additions:
- `expose_as_tool: bool = True` — whether this agent participates in the ReAct loop (§4.4)
- `schema_version: int` — auto-computed from InputField definitions (§5.3)
- Store `validator_description: str | None` on InputField specs for schema enhancement

New functions:
- **generate_tool_schema()** (§4.1): Map agent_cls docstring to tool description, InputField to JSON Schema property (name, type, description, required), add task_instruction parameter.
- **enhance_agent_tool_schema()** (§4.3): Inject validator constraints into parameter descriptions; if needs_approval() overridden, append `[Requires user confirmation before execution]`; add task_instruction usage guidance.
- **get_schema_version()** (§5.3): Hash of InputField names, types, required flags. Changes when fields are added/removed/retyped.

**Modify**: `onevalet/config/registry.py` (AgentRegistry)

New methods:
- **get_all_agent_tool_schemas()**: Return enhanced tool schemas for all agents with expose_as_tool=True.
- **get_schema_version(agent_type)**: Return schema version for a registered agent type.

**Depends on**: Nothing.

---

### Step 1.4 — Agent-Tool Execution Logic (§4.1, §6, §3.1)

**New file**: `onevalet/orchestrator/agent_tool.py`

**AgentToolResult** dataclass: completed (bool), result_text, agent (non-None when in WAITING state), approval_request (non-None when WAITING_FOR_APPROVAL).

**execute_agent_tool()** function, per §4.1 + §6:

1. Create Agent via orchestrator.create_agent(), tool_call_args passed as context_hints
2. Agent.__init__ receives context_hints → pre-populate collected_fields (§6: extracted and valid → directly adopted)
3. Build Message from task_instruction (or empty)
4. Call agent.reply(msg)
5. Return based on AgentResult.status:
   - COMPLETED → completed=True, result_text
   - WAITING_FOR_INPUT → completed=False, agent instance
   - WAITING_FOR_APPROVAL → completed=False, agent instance + ApprovalRequest
   - ERROR → completed=True, error message

**Depends on**: Step 1.3.

---

### Step 1.5 — Structured Approval Flow (§13)

**New file**: `onevalet/orchestrator/approval.py`

**ApprovalRequest** dataclass (§13.2): agent_name, action_summary, details (dict), options (["approve", "edit", "cancel"]), timeout_minutes, allow_modification.

**build_approval_request()**: Build from agent.get_approval_prompt() + agent.collected_fields.

**collect_batch_approvals()** (§13.3): When multiple Agent-Tools in one turn all need approval, collect all ApprovalRequests and present to user at once.

**Approval result flow-back** (§13.4): User's approval response appended to ReAct loop messages as tool_result. Cancellation sent as is_error=true tool_result. Handled in check_pending_agents (Step 1.7).

**Depends on**: Step 1.4.

---

### Step 1.6 — react_loop() Full Implementation (§3, §11, §12, §14)

**Modify**: `onevalet/orchestrator/orchestrator.py`

New **react_loop()** method implementing the complete §3 logic:

#### Main loop (§3.1 Step 3)

Each iteration:
1. **Context guard (§12.2 Defense 2)**: trim_if_needed()
2. **LLM call + error recovery (§3.3)**: _llm_call_with_retry()
3. **No tool_calls → final answer**: Return ReactLoopResult
4. **Has tool_calls → execute all concurrently (§11)**: asyncio.gather, each call with independent timeout (regular Tool uses tool_execution_timeout, Agent-Tool uses agent_tool_execution_timeout)
5. **Process results**:
   - Exception → error message as tool_result appended to messages, continue loop (§3.3 Tool Execution Errors → delegated to LLM)
   - AgentToolResult.completed → truncate result, append to messages, continue loop
   - AgentToolResult not completed → Agent stored in Pool, collect ApprovalRequest, break loop and return
   - Regular Tool result → truncate, append to messages, continue loop

#### max_turns reached (§3.1)

Inject instruction asking LLM to summarize, final LLM call with no tools (force text response).

#### LLM call retry (§3.3)

Strategy-pattern error handling:
- RateLimitError → exponential backoff retry
- ContextOverflowError → three-step recovery chain (trim_if_needed → truncate_all_tool_results → force_trim → give up)
- AuthError → raise directly
- TimeoutError → retry once

#### Concurrent execution (§11.2)

**_execute_with_timeout()**: asyncio.wait_for wrapper, timeout returns TimeoutError as result (does not break loop).

**_execute_single()**: Dispatch to execute_agent_tool() or tool_registry.execute() based on _is_agent_tool().

#### Batch Agent-Tool approval (§11.3)

Multiple Agent-Tools in one turn all execute concurrently to their approval points → collect all ApprovalRequests → store all waiting agents in Pool → return combined pending_approvals list → break loop.

**Depends on**: Steps 1.1–1.5.

---

### Step 1.7 — handle_message() Rewrite (§2, §3.1 Step 1-2, §3.2)

**Modify**: `onevalet/orchestrator/orchestrator.py`

**__init__ new parameters**:
- `system_prompt: str = ""` — persona injection point (§3.1 Step 2)
- `react_config: ReactLoopConfig = None` — defaults to ReactLoopConfig()

**Rewrite handle_message()** flow:

1. **prepare_context()** — unchanged (§2)
2. **should_process()** — unchanged (§2)
3. **_check_pending_agents()** — new (§3.1 Step 1): Check Pool for WAITING_FOR_INPUT / WAITING_FOR_APPROVAL agents. Route current message to that agent. Agent completes → remove from Pool, result as context enters ReAct loop. Agent still waiting → return Agent's prompt directly, skip ReAct loop. (§3.2 interruption/resumption)
4. **_build_llm_messages()** — new (§3.1 Step 2): Assemble system prompt (persona + recalled memories + current time) + conversation history + Agent result (if any) + current user message
5. **_build_tool_schemas()** — new (§4.2): Merge regular Tools + Agent-Tools into unified tool list
6. **react_loop()** — new (§3.1 Step 3)
7. **post_process()** — unchanged (§2)

**Depends on**: Step 1.6, Step 1.3.

---

### Step 1.8 — stream_message() ReAct Version (§10)

**Modify**: `onevalet/orchestrator/orchestrator.py`

Rewrite stream_message(), same flow as handle_message() but yielding streaming events:

| Loop Phase | Event |
|---|---|
| LLM starts output | MESSAGE_START |
| LLM outputs text token | MESSAGE_CHUNK |
| LLM output ends | MESSAGE_END |
| LLM returns tool_call | TOOL_CALL_START (tool name + args) |
| Tool/Agent execution complete | TOOL_RESULT (result summary) |
| Agent enters WAITING_FOR_INPUT | STATE_CHANGE |
| Agent enters WAITING_FOR_APPROVAL | STATE_CHANGE |
| Loop ends | EXECUTION_END |

Implementation: LLM streaming enabled. Each ReAct turn: stream LLM response → if tool_calls, execute concurrently and emit TOOL_RESULT → if Agent WAITING, emit STATE_CHANGE and break.

Reuse existing StreamEngine and StandardAgent emit_xxx methods.

**Depends on**: Step 1.7.

---

### Step 1.9 — AgentPoolEntry Schema Version Guard (§5.3)

**Modify**: `onevalet/orchestrator/models.py`

AgentPoolEntry new field: `schema_version: int`, set from agent class at creation time.

**Modify**: `onevalet/orchestrator/pool.py`

In restore_tenant_session(): iterate entries, compare entry.schema_version with agent_registry.get_schema_version(entry.agent_type). Mismatch → remove entry from Pool, log warning, skip restore. Match → restore normally.

**Depends on**: Step 1.3 (get_schema_version).

---

### Step 1.10 — Remove MessageRouter Dependency (§16, §21)

**Modify**: `onevalet/orchestrator/orchestrator.py`
- Remove route_message() method
- Remove MessageRouter import and initialization
- Remove _route_with_llm(), _extract_with_llm() references

**Deprecate** (do not delete): `onevalet/orchestrator/router.py`
- Add module-level deprecation warning

**Depends on**: Step 1.7.

---

## Phase 2: CredentialStore (Design §9)

**Goal**: Per-tenant credential storage. One class, Postgres backend, no ABC, no CredentialScheme.

**Can run in parallel with Phase 1.**

---

### Step 2.1 — CredentialStore (§9.2, §9.4, §9.5)

**New file**: `onevalet/credentials/store.py`

**CredentialStore** — single class, Postgres backend directly:
- save(tenant_id, service, credentials, account_name="primary")
- get(tenant_id, service, account_name="primary") → dict | None
- list(tenant_id, service=None) → list[dict]
- delete(tenant_id, service, account_name="primary")

Table: `credentials`
Columns: tenant_id, service, account_name, credentials_json, created_at, updated_at
Primary key: (tenant_id, service, account_name)

Uses asyncpg. Auto-creates table on first access.

No CredentialScheme, no CredentialStatus, no CredentialEntry wrapper. The framework stores and retrieves `dict`. Each agent/provider knows what's inside.

### Step 2.2 — Integration (§9.6)

**Modify**: `onevalet/tools/models.py` — ToolExecutionContext new field: `credentials: CredentialStore | None = None`

**Modify**: `onevalet/orchestrator/orchestrator.py` — __init__ new parameter: `credential_store`, passed to ToolExecutionContext when executing tools/agents in react_loop

**New file**: `onevalet/credentials/__init__.py` — export CredentialStore

---

## Phase 3: Momex Integration (Design §8)

**Goal**: Direct Momex integration for conversation history + long-term knowledge. No MemoryProvider ABC.

**Can run in parallel with Phase 1.**

---

### Step 3.1 — Momex Wrapper (§8.2, §8.3)

**New file**: `onevalet/memory/momex.py`

Thin wrapper around typeagent-py's Memory + ShortTermMemory API. Not an abstract class — direct Momex calls:

- get_history(tenant_id, session_id, limit) → ShortTermMemory.get_session_messages()
- save_history(tenant_id, session_id, messages) → ShortTermMemory.add_messages()
- search(tenant_id, query, limit) → Memory.search()
- add(tenant_id, messages, infer=True) → Memory.add_messages() (Momex internally handles entity extraction, contradiction detection, index updates)

Init params: collection (str), config (MomexConfig)

### Step 3.2 — Integration (§8.3, §8.4)

**Modify**: `onevalet/orchestrator/orchestrator.py`

__init__ new parameter: `momex = None`

In **prepare_context()** (§8.3 "Before request"): if momex → load conversation_history (get_history) + recall relevant memories (search)

In **post_process()** (§8.3 "After request"): if momex → save conversation history (save_history) + long-term knowledge extraction (add with infer=True)

If momex not provided → no memory, no conversation history (stateless).

### Step 3.3 — Deprecate Existing MemoryManager

**Modify**: `onevalet/memory/manager.py`
- Add deprecation warning
- Keep functional for backward compatibility

---

## Phase 4: TriggerEngine (Design §15)

**Goal**: Proactive trigger system migrated from KoiAI, integrated with ReAct loop. Supports both LLM-driven and deterministic execution paths.

**Depends on**: Phase 1 (react_loop needed for OrchestratorExecutor).

---

### Step 4.1 — Core Engine + Models (§15.2, §15.5)

**New directory**: `onevalet/triggers/`

**New file**: `onevalet/triggers/models.py`
- Migrated from `koiai/core/triggers/models.py`
- Task, TriggerConfig, ActionConfig, TaskStatus (ACTIVE / PAUSED / DISABLED / COMPLETED / PENDING_APPROVAL / EXPIRED)
- TriggerContext, ActionResult

**New file**: `onevalet/triggers/engine.py`
- Migrated from `koiai/core/triggers/engine.py`
- Remove KoiAI-specific dependencies (Supabase client, KoiAI config imports)
- Keep: task CRUD, trigger evaluation loop, executor dispatch, task persistence

### Step 4.2 — Trigger Types (§15.5)

**New file**: `onevalet/triggers/schedule.py`
- Migrated from KoiAI: cron (via croniter), interval, one-time
- Logic unchanged

**New file**: `onevalet/triggers/event.py`
- Migrated from KoiAI: source + type + filter matching
- Logic unchanged

**New file**: `onevalet/triggers/condition.py`
- Full implementation (KoiAI only has placeholder)
- Periodic polling + condition expression evaluation

### Step 4.3 — EventBus (§15.7)

**New file**: `onevalet/triggers/event_bus.py`

**EventBus** — Redis Streams, single class, direct implementation. Migrated from `koiai/core/events/bus.py`:
- publish(event)
- subscribe(pattern, callback)
- unsubscribe(pattern)

No ABC, no InMemoryEventBus. One class, Redis Streams.

### Step 4.4 — Executors + Dual Execution Path (§15.3, §15.4)

**New file**: `onevalet/triggers/executor.py`

**OrchestratorExecutor** (§15.3) — default executor:
- Converts trigger event to TriggerMessage (tenant_id, content, metadata with source/trigger_type/task_id)
- Calls orchestrator.handle_message(message) to enter ReAct loop
- Pushes result to user via notification channels
- _build_message() builds context message by trigger type (schedule → instruction, event → event data, condition → condition expression)

**Executor registry** (§15.4) — dict-based:
- register(name, executor_instance)
- get(name) → executor
- Built-in: "orchestrator" → OrchestratorExecutor (default)
- Application registers custom executors (e.g. "email_pipeline" → EmailPipelineExecutor)
- Task creation specifies `executor` field to select which executor

Two paths coexist:
- **Path 1: OrchestratorExecutor** (default) — LLM-driven, flexible
- **Path 2: Custom executors** — deterministic pipelines for high-frequency triggers

### Step 4.5 — Notification Channels (§15.6)

**New file**: `onevalet/triggers/notification.py`

Direct concrete classes, no ABC:
- **SMSNotification**: send(tenant_id, message, metadata) via Twilio/SignalWire
- **PushNotification**: send(tenant_id, message, metadata) via push service

TriggerEngine holds a list of notification instances, calls them directly.

### Step 4.6 — Triggered Task Approval Flow (§15.10)

**Modify**: `onevalet/triggers/models.py`

TaskStatus includes: PENDING_APPROVAL, EXPIRED (already in Step 4.1)

**Modify**: `onevalet/triggers/engine.py`

Full offline approval flow:

1. Triggered task enters ReAct loop → Agent-Tool needs approval → Agent stored in Pool (TTL = approval_timeout_minutes)
2. Triggered task marked as PENDING_APPROVAL
3. Notification pushed to user via notification channels ("An email awaits your confirmation")
4. User comes online:
   - Query list_pending_approvals(tenant_id)
   - Or reply directly via notification channel link
5. User confirms → Agent resumes execution → result pushed via notification channels
6. TTL expires without confirmation → Agent removed from Pool → task marked as EXPIRED

**TTL expiry cleanup**: TriggerEngine periodically scans PENDING_APPROVAL tasks, checks if associated Agent has been TTL-removed from Pool. If removed → update task status to EXPIRED.

### Step 4.7 — Integration (§15.8, §15.9)

**Modify**: `onevalet/orchestrator/orchestrator.py`
- __init__ new parameter: `trigger_engine: TriggerEngine = None`
- Start trigger engine on orchestrator startup
- Stop on shutdown

New method **list_pending_approvals(tenant_id)** (§15.10):
- Query Pool for WAITING_FOR_APPROVAL agents
- Return structured list: agent_name, action_summary, created_at, expires_at, source (user/trigger), task_id

**New file**: `onevalet/triggers/__init__.py` — export public types

---

## Phase 5: KoiAI Agent Migration (Design §18)

**Goal**: Migrate all KoiAI agents + providers into OneValet as built-in capabilities.

**Depends on**: Phase 1 + Phase 2 + Phase 3 + Phase 4.

---

### Step 5.1 — Provider Layer

**New directory**: `onevalet/providers/`

Migrate all providers from KoiAI. Every provider: Supabase credential queries → `CredentialStore.get()`, ProviderFactory.create(oauth_account) → ProviderFactory.create(creds_dict).

Directory structure:
- `email/` — base, gmail, outlook, resolver (uses CredentialStore.list), factory (receives credentials dict)
- `calendar/` — base, google, factory
- `travel/` — amadeus (extracted from flight_search_agent), weather (extracted from weather_agent)
- `maps/` — places (extracted from map_search_agent), directions (extracted from directions_agent)
- `shipment/` — tracking
- `sms/` — base, twilio, signalwire

### Step 5.2 — Built-in Agents

**New directory**: `onevalet/builtin_agents/`

Directory structure:
- `email/` — read, send, delete, reply, archive, mark_read, importance, preference, summary
- `calendar/` — query, create_event, update_event, delete_event
- `travel/` — flight_search, hotel_search, weather
- `maps/` — search, directions, air_quality
- `reminder/` — reminder, task_mgmt, planner
- `shipment/` — tracking
- `tools/` — google_search (registered as @tool), important_dates (registered as @tool)

### Step 5.3 — Per-Agent Migration Changes

| Before (KoiAI) | After (OneValet) |
|---|---|
| `self.orchestrator_callback("get_cache", key)` | Remove. ReAct loop messages + Momex history carry context naturally |
| `self.orchestrator_callback("cache_update", data)` | Remove. Same reason |
| `self.orchestrator_callback("send_sms", msg)` | SMSNotification.send() via context |
| Direct Supabase credential query | `self.context.credentials.get(tenant_id, service)` |
| `ProviderFactory.create(oauth_account)` | `ProviderFactory.create(creds_dict)` |
| `required_tier` in agent class | Move to agent_registry.yaml config, checked by `should_process()` |
| KoiAgent personality wrapping | Remove. `system_prompt` handles persona |
| `triggers=["send email", ...]` | Keep but optional. ReAct loop understands intent via description |

### Step 5.4 — KoiAgent Decomposition (§18)

| KoiAgent responsibility | Goes to |
|---|---|
| Personality definition (PERSONALITY_PROMPT) | `Orchestrator(system_prompt=PERSONALITY_PROMPT)` |
| Chat fallback (no agent match) | ReAct loop's LLM responds directly |
| google_search tool | `builtin_agents/tools/google_search.py` (registered as @tool) |
| important_dates tool | `builtin_agents/tools/important_dates.py` (registered as @tool) |
| Profile detection (_detect_and_update_profile) | `post_process()` as background task |
| 6 response wrapping methods (_wrap_completed_result, _generate_input_request, _generate_approval_request, _generate_tier_upgrade_message, _generate_error_response, _generate_clarification) | All removed. ReAct LLM generates all responses in system_prompt persona |

### Step 5.5 — KoiAI-Specific Cleanup (§18)

**Remove**:
- `routing_llm_provider` — routing replaced by ReAct loop, no separate routing LLM needed
- All `MessageRouter` code (Phase 1 Step 1.10 deprecated, Phase 5 full removal)
- Supabase `chat_history` table — conversation history unified into Momex short-term memory

**Modify**: `agent_registry.yaml`
- `triggers` field becomes optional (kept for compatibility but no longer a routing dependency)
- ReAct loop understands intent via agent description

### Step 5.6 — Credential Data Migration (§9.8)

Migrate from KoiAI's Supabase `oauth_accounts` table to CredentialStore:

- One-time migration script: read Supabase oauth_accounts → write to CredentialStore (Postgres)
- AccountResolver switches to CredentialStore.list() instead of Supabase queries
- ProviderFactory receives credentials dict instead of oauth_account object
- OAuth flow unchanged, only storage target changes from Supabase → credential_store.save()

### Step 5.7 — Notification Implementations (§15.6, §18)

**SMSNotification**: Built on existing SMS provider (Twilio/SignalWire). Direct implementation in product code.

**PushNotification**: Push notification implementation in product code.

### Step 5.8 — Final Orchestrator Shape

Orchestrator init: llm_client, system_prompt (= PERSONALITY_PROMPT), react_config, credential_store (Postgres), momex (direct Momex instance), trigger_engine (with EventBus, executor, notifications)

Customize should_process() — tier checking, guardrails, rate limiting
Customize post_process() — profile detection (background task), usage recording

---

## Phase Dependency Graph

```
Phase 1 (ReAct Loop — complete, not "minimum viable")
  1.1  ReactLoopConfig + Data Types                         ─┐
  1.2  ContextManager              ← 1.1                    │
  1.3  Agent-Tool Schema           (independent)             │
  1.4  Agent-Tool Execution        ← 1.3                    │
  1.5  Structured Approval         ← 1.4                    ├── Sequential
  1.6  react_loop()                ← 1.1–1.5                │
  1.7  handle_message()            ← 1.6, 1.3               │
  1.8  stream_message()            ← 1.7                    │
  1.9  Schema Version Guard        ← 1.3                    │
  1.10 Remove MessageRouter        ← 1.7                    ─┘

Phase 2 (CredentialStore)          ← Nothing (parallel with Phase 1)
Phase 3 (Momex)                    ← Nothing (parallel with Phase 1)
Phase 4 (TriggerEngine)            ← Phase 1 (needs react_loop)
Phase 5 (KoiAI Migration)          ← Phase 1 + 2 + 3 + 4
```

---

## Files Summary

### New Files (25+)

| File | Phase |
|------|-------|
| `orchestrator/react_config.py` | 1 |
| `orchestrator/context_manager.py` | 1 |
| `orchestrator/agent_tool.py` | 1 |
| `orchestrator/approval.py` | 1 |
| `credentials/__init__.py` | 2 |
| `credentials/store.py` | 2 |
| `memory/momex.py` | 3 |
| `triggers/__init__.py` | 4 |
| `triggers/engine.py` | 4 |
| `triggers/models.py` | 4 |
| `triggers/schedule.py` | 4 |
| `triggers/event.py` | 4 |
| `triggers/condition.py` | 4 |
| `triggers/event_bus.py` | 4 |
| `triggers/executor.py` | 4 |
| `triggers/notification.py` | 4 |
| `providers/` (directory, ~15 files) | 5 |
| `builtin_agents/` (directory, ~25 files) | 5 |
| Credential migration script | 5 |

### Modified Files (7)

| File | Phase |
|------|-------|
| `orchestrator/orchestrator.py` | 1, 2, 3, 4 |
| `agents/decorator.py` | 1 |
| `config/registry.py` | 1 |
| `orchestrator/models.py` | 1 |
| `orchestrator/pool.py` | 1 |
| `tools/models.py` | 2 |
| `agent_registry.yaml` | 5 |

### Deprecated / Removed

| File | Phase | Action |
|------|-------|--------|
| `orchestrator/router.py` | 1 | Deprecate (add warning) |
| `orchestrator/router.py` | 5 | Remove completely |
| `memory/manager.py` | 3 | Deprecate |
| KoiAI `routing_llm_provider` | 5 | Remove |
| Supabase `chat_history` table | 5 | Remove (replaced by Momex) |
