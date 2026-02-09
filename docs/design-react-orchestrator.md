# FlowAgents ReAct Orchestrator Design

> Goal: Build FlowAgents into an extensible personal assistant agent framework.
> Core change: Replace the Orchestrator's single-shot routing with a ReAct loop.

---

## 1. Current State and Problems

### Current Architecture

```
User Message
    -> Orchestrator.handle_message()
        -> prepare_context()
        -> should_process()
        -> route_message()         <- Single routing decision, picks one Agent
        -> create_agent()
        -> execute_agent()         <- Executes only one Agent
        -> post_process()
    -> Response
```

### Problems

1. **One message can only route to one Agent.** Requests like "check weather + book flight + send email" cannot be handled.
2. **Routing relies on trigger keyword matching or LLM classification** — a one-time decision that cannot dynamically adjust based on intermediate results.
3. **No free conversation ability.** When no Agent matches, it can only return fallback text.
4. **No inter-Agent collaboration.** One Agent's output cannot naturally flow into another Agent.

---

## 2. Core Change

**Only one change: replace `route_message -> execute_agent` with a ReAct loop.**

### Updated handle_message flow

```
User Message
    -> Orchestrator.handle_message()
        -> prepare_context()       <- Unchanged
        -> should_process()        <- Unchanged
        -> check_pending_agents()  <- New: check Pool for Agents awaiting user input
        -> react_loop()            <- New: replaces route + execute
        -> post_process()          <- Unchanged
    -> Response
```

All other components — Agent definitions, Tool system, Agent Pool, Streaming, Memory interface, Extension Points — remain unchanged.

---

## 3. ReAct Loop Full Logic

### 3.1 Overall Flow

For each incoming user message, the Orchestrator processes in the following order:

#### Step 1: Check Pool for Pending Agents

If the Pool contains Agents for this user in WAITING_FOR_INPUT or WAITING_FOR_APPROVAL state, it means a previous task is incomplete. The current message is routed to that Agent first.

After the Agent processes:
- **Agent completes** -> Remove from Pool. Use the execution result as context, enter Step 2's ReAct loop to let the LLM determine if there are follow-up actions.
- **Agent still waiting** -> Update Pool, return the Agent's prompt directly to the user. Do not enter the ReAct loop.

#### Step 2: Build LLM Input

Assemble the LLM messages:
- **System Prompt**: User-configured `system_prompt` (persona / personality definition) + framework-appended context (current time, user timezone, capabilities list)
- **Recalled memories**: Query the MemoryProvider for historical knowledge relevant to the user message
- **Conversation history**: Loaded from MemoryProvider's short-term memory (`memory.get_history()`)
- **Current user message**
- **(If applicable) Agent result from Step 1**

The `system_prompt` is the **persona injection point**. The ReAct loop's LLM speaks in this persona natively — all responses (including tool result summaries, error messages, approval requests) are generated in this voice. No separate persona wrapping step is needed.

```python
# Example: KoiAI injects Koi's personality
orchestrator = Orchestrator(
    llm_client=llm,
    system_prompt="""You are Koi. You're a person, not an AI.
Talk like you're texting a friend. Be natural, human, real.
Output is sent via SMS. Use PLAIN TEXT only.""",
)
```

Simultaneously build **tool schemas**: all registered regular Tools + all registered Agents auto-converted to Agent-Tools.

#### Step 3: ReAct Loop

```
while turns < max_turns:
    # 0. Context guard: check token usage, trim history if over limit
    messages = context_manager.trim_if_needed(messages)

    # 1. Call LLM (with retry and error recovery)
    try:
        llm_response = await llm_call_with_retry(messages, tool_schemas)
    except LLMError as e:
        -> Error handling (see Section 3.3)

    # 2. No tool_calls -> final answer
    If LLM has no tool_calls:
        -> This is the final answer, return to user, loop ends

    # 3. Has tool_calls -> execute all concurrently by default
    If LLM has tool_calls:
        # Multiple tool_calls returned in the same turn are inherently independent:
        #   - The LLM returns them together only when it believes they can run simultaneously
        #   - If B depends on A's result, the LLM returns them in separate turns (A first, then B after getting results)
        #   - Cross-turn sequencing is managed by the LLM itself; the framework needs no dependency analysis
        # Therefore, execute all concurrently by default
        results = await asyncio.gather(*[
            execute_with_timeout(tc) for tc in tool_calls
        ], return_exceptions=True)

        For each execution result:
            Regular Tool:
                -> Truncate oversized results (context_manager.truncate_tool_result)
                -> Append to messages, continue loop

            Agent-Tool:
                -> Agent completed?
                    -> Yes: Truncate result, append to messages, continue loop
                    -> No: Store Agent in Pool, break loop, return Agent's prompt to user

            Execution failed (timeout/exception):
                -> Error message appended as tool result to messages
                -> Continue loop, let LLM decide how to handle (instead of hard framework abort)
```

#### Loop Termination Conditions

1. **LLM returns final answer** (no tool_calls) -> Normal completion
2. **Agent requires user participation** -> Break loop, Agent stored in Pool
3. **max_turns reached** -> Force summary then end (see below)
4. **Context overflow recovery failed** -> Return error message

#### max_turns Reached Handling

```python
if turns >= max_turns:
    # Inject instruction asking the LLM to summarize based on available information
    messages.append({
        "role": "user",
        "content": "You have executed enough steps. Please provide a final answer based on the information gathered so far."
    })
    # Final LLM call with no tools (force text response, no further tool calls allowed)
    final_response = await llm_call(messages, tools=None)
    return final_response
```

#### Return Value

`react_loop()` returns a structured result for application-layer analytics and display:

```python
@dataclass
class ReactLoopResult:
    response: str                    # Final answer
    turns: int                       # Actual loop iterations
    tool_calls: list[ToolCallRecord] # Record of each tool call (name, args, duration, success)
    token_usage: TokenUsage          # input_tokens, output_tokens, total
    duration_ms: int                 # Total duration
    pending_approvals: list          # Pending approval requests (if any)
```

### 3.2 Interruption and Resumption

The ReAct loop may be interrupted at any step by an Agent. On resumption (user's next message), the flow starts from Step 1:

1. Discover a waiting Agent in the Pool
2. Route the user message to that Agent
3. After Agent completes, build new LLM input with the result and previous conversation history
4. ReAct loop continues, LLM decides if there are follow-up actions

Conversation history ensures the LLM knows what was done previously (which tools were called, what results were obtained), enabling seamless continuation.

### 3.3 Error Recovery Strategy

Errors in the ReAct loop fall into two categories with different handling:

#### LLM Call Errors (Framework Handled)

```python
error_handlers = {
    RateLimitError: retry_with_exponential_backoff,  # Retry with exponential backoff
    ContextOverflowError: context_overflow_recovery,  # Three-step recovery chain (see below)
    AuthError: raise_to_caller,                      # Unrecoverable, propagate up
    TimeoutError: retry_once,                        # Retry once
}
```

#### Context Overflow Three-Step Recovery Chain

When the LLM API returns a context overflow error, attempt recovery in order:

```
ContextOverflowError
  -> Step 1: trim_if_needed()     -- Trim history messages (keep most recent N) -> Retry LLM
    -> Step 2: truncate_all_tool_results() -- Compress all large tool results in messages -> Retry LLM
      -> Step 3: force_trim()     -- Keep only the most recent 5 messages -> Final retry
        -> Give up, return "Conversation too long, please start a new conversation"
```

After each recovery step, retry the LLM call. If successful, continue the loop; if failed, proceed to the next step. Maximum 3 retries.

#### Tool Execution Errors (Delegated to LLM)

When Tool/Agent-Tool execution fails, **do not break the loop**. Instead, return the error as a tool result to the LLM:

```
tool_result = {
    "tool_call_id": "xxx",
    "content": "Error: Gmail API timeout after 10s. Service may be temporarily unavailable.",
    "is_error": true
}
```

The LLM can autonomously decide: skip the tool and continue other tasks, use an alternative approach, or inform the user of partial failure.
This is more appropriate for personal agent scenarios than hard framework interruption — "email failed but weather and flight info can be provided."

---

## 4. Agent-Tool Mechanism

### 4.1 Concept

Agent-Tool is not a new type. It is simply a perspective: **automatically map @flowagent registered Agent's InputFields to a Tool parameter schema**, allowing the LLM to trigger Agents via function calling.

The framework performs this conversion automatically when building tool schemas:
- Agent's `description` (docstring) -> Tool's description
- Agent's `InputField` list -> Tool's parameters (name, type, description, required)
- Additionally adds a `task_instruction` parameter, allowing the LLM to pass natural language instructions

### 4.2 LLM Perspective

The LLM sees a unified tool list. It does not know and does not need to know which are regular functions and which are backed by stateful Agents.

For example, the tool list the LLM sees:

```
- get_weather(city, date)                              <- Regular Tool
- web_search(query)                                    <- Regular Tool
- SendEmailAgent(recipient, subject, body)             <- Agent-Tool
- FlightSearchAgent(origin, destination, date)         <- Agent-Tool
- CalendarAgent(action, date, title)                   <- Agent-Tool
```

The LLM selects as needed; the framework differentiates internally.

### 4.3 Configuration Control

Not all Agents need to be exposed as Agent-Tools. Controlled via configuration (agent_registry.yaml or @flowagent parameters):
- `expose_as_tool: true/false` — Whether to participate in the ReAct loop
- Unexposed Agents can still be triggered via explicit workflows

---

## 5. Agent Pool Role

### 5.1 Responsibilities Unchanged

The Pool's responsibility remains: **store Agent instances in non-terminal states, isolated by tenant_id, with TTL expiration.**

### 5.2 Entry Path Changes

| | Before | After |
|---|---|---|
| How Agents enter Pool | Orchestrator routes and creates Agent -> enters Pool | Agent-Tool in ReAct loop not completed -> enters Pool |
| How Agents are found | Matched during next message routing | Pool checked first on next incoming message |
| How Agents exit Pool | Execution complete / TTL expired / Manual cancel | Same |

### 5.3 All Preserved Features

- Isolation by tenant_id
- TTL auto-expiration
- Serialization / deserialization (Redis backend, session restoration)
- Max Agent count per user limit
- pause / resume / cancel API

---

## 6. LLM Field Extraction vs Agent Field Collection

### Collaboration

When the LLM calls an Agent-Tool, it has already extracted parameters from the user message (as tool_call arguments). These parameters are injected into the Agent as `context_hints`.

The Agent compares against its InputFields:
- **Extracted and valid** -> Directly adopted (skip collection step)
- **Extracted but invalid** -> Goes through validation failure flow
- **Missing but optional** -> Use default value
- **Missing and required** -> Agent enters WAITING_FOR_INPUT, stored in Pool

### Effect

In most cases, the LLM's extraction ability far exceeds existing trigger keyword matching + simple extraction. The probability of an Agent completing in one shot is greatly increased. Multi-turn collection is only needed when information is truly missing.

---

## 7. Multi-Agent Collaboration

This is a natural advantage of the ReAct loop.

### Example: "Search flights + check weather + send email"

```
Turn 1:
  LLM thinks: User needs three things, flights and weather can be checked simultaneously
  LLM returns tool_calls:
    - FlightSearchAgent(origin="SFO", dest="NYC", date="next Friday")
    - get_weather(city="NYC", date="next Friday")
  Concurrent execution -> Results: 3 flight options + Sunny 15C

Turn 2:
  LLM sees flight + weather results, composes email
  LLM returns tool_calls:
    - SendEmailAgent(recipient="team@...", subject="NYC Trip", body="flight+weather info...")
  Execution -> Agent needs approval -> Interrupt, prompt user for confirmation

Turn 3:
  User: "yes"
  -> Pool finds SendEmailAgent -> Execute approval -> Complete
  -> Result returns to ReAct loop
  -> LLM thinks: Everything done
  -> Final answer: "Completed: flights/weather/email ..."
```

---

## 8. Memory System Integration

### 8.1 Design Principle

The memory system is a component that **users must explicitly configure**. The framework provides no default implementation and no degradation. Whichever provider the user chooses, they get that provider's full capabilities.

### 8.2 Recommended Memory Provider: Momex

Momex (typeagent-py) is this framework's recommended memory system, natively covering all memory capabilities needed for a personal agent:

**Long-term Memory (Structured RAG):**
- Structured knowledge extraction: entities (with facets/attributes), actions (subject-verb-object relations), topics
- Contradiction detection and auto-update (LLM-driven, new facts automatically replace old contradictions)
- Hybrid search: structured index + semantic vectors + full-text search
- LLM-driven Q&A (`memory.query("What languages does the user prefer?")`)

**Short-term Memory (Session History):**
- Session-based conversation history with multi-session concurrency support
- Database persistence + in-memory cache (configurable max_messages)
- Automatic session expiration cleanup

**Multi-tenant Isolation:**
- Hierarchical Collections (`user:xiaoyuzhang`, `team:eng:alice`)
- Prefix queries support cross-Collection retrieval

**Storage Backends:**
- SQLite (zero-config) / PostgreSQL (production-grade, supports pgvector)

### 8.3 Other Optional Providers

- **mem0**: Vector storage + simple retrieval, suitable for lightweight scenarios
- **Custom implementation**: Users implement the MemoryProvider interface

### 8.4 Integration Points

The memory system integrates through existing Extension Points without changing core logic:

**Before request — prepare_context():**
1. Call `memory.get_history(tenant_id)` to load conversation history (short-term memory)
2. Call `memory.search(query)` with the current user message to recall relevant long-term memories
3. Inject recalled results into the System Prompt (as user background/preference context)

**After request — post_process():**
1. Call `memory.save_history(tenant_id, messages)` to persist the current conversation
2. Call `memory.add(messages, infer=True)` for long-term knowledge extraction
3. Momex internally auto-completes entity extraction, contradiction detection, and index updates

**Conversation history is managed by the MemoryProvider, not by the framework or application layer separately.** This means:
- One system (Momex) handles both short-term history and long-term knowledge
- No need for a separate `chat_history` database table (KoiAI's current dual-system pattern is unified)
- If no MemoryProvider is configured, there is no conversation history (stateless). The application layer can still manage history themselves via `prepare_context()`/`post_process()` extension points if desired.

**Collaboration with Context Management:**
- History messages discarded by context trimming are not lost — important information was already stored in Momex long-term memory during `post_process()`
- On the next conversation, recalled via `prepare_context()`, achieving "discard history but retain knowledge"

### 8.5 Configuration

```python
orchestrator = Orchestrator(
    llm_client=llm,
    system_prompt="You are Koi. You're a person, not an AI...",
    memory_provider=MomexMemoryProvider(
        collection="user:xiaoyuzhang",
        config=MomexConfig.from_env(),
    ),
)
```

If `memory_provider` is not provided, there is no memory capability and no conversation history (stateless). The framework does not error — it simply operates without recall or storage. Applications can still manage history manually via the `prepare_context()`/`post_process()` extension points.

---

## 9. Credential Store

### 9.1 Design Principle

The framework has built-in per-user credential storage and retrieval — **no need for users to inject query logic**. Just as Django has built-in `auth.User`, FlowAgents has built-in `CredentialStore`.

The framework is only responsible for **storing and retrieving**. OAuth flows, token refresh, and Provider selection are business logic, belonging in specific Agent/Tool implementations.

### 9.2 CredentialStore

```python
class CredentialStore:
    """
    Per-user credential storage and retrieval. Built into the framework, works out of the box.
    Data is isolated by tenant_id, naturally supporting multi-tenancy.
    """

    async def save(
        self,
        tenant_id: str,
        service: str,
        credentials: dict,
        account_name: str = "primary"
    ):
        """
        Save credentials

        Args:
            tenant_id: User ID
            service: Service name ("google", "microsoft", "amadeus")
            account_name: Account name ("primary", "work", "personal")
            credentials: Credential data, format determined by service, e.g.:
                {
                    "access_token": "ya29...",
                    "refresh_token": "1//0g...",
                    "token_expiry": "2025-01-01T00:00:00",
                    "email": "user@gmail.com",
                    "scopes": ["gmail.send", "gmail.modify"]
                }
        """

    async def get(
        self,
        tenant_id: str,
        service: str,
        account_name: str = "primary"
    ) -> dict | None:
        """Retrieve credentials. Returns None if not found."""

    async def list(
        self,
        tenant_id: str,
        service: str | None = None
    ) -> list[dict]:
        """List all connected accounts for a user, optionally filtered by service."""

    async def delete(
        self,
        tenant_id: str,
        service: str,
        account_name: str = "primary"
    ):
        """Delete credentials."""
```

### 9.3 Storage Backend

Credentials must be persisted (OAuth tokens cannot be lost on restart). SQLite is the default:

```yaml
# flowagents.yaml
orchestrator:
  session_backend: "memory"       # Sessions can use memory
  credential_backend: "sqlite"    # Credentials must be persisted
```

Supported backends:
- **sqlite** (default): Zero-config, suitable for single-machine / personal assistant
- **redis**: Suitable for distributed deployments
- Extensible to other backends

### 9.4 Data Model

Internal storage structure (transparent to users):

```
Key: (tenant_id, service, account_name)
Value: {
    "credentials": { ... },      # The credential dict stored by the user
    "created_at": "...",
    "updated_at": "..."
}
```

Multi-tenant isolation relies on `tenant_id`, which is already pervasive throughout the framework.

### 9.5 Usage in Tools / Agents

Accessed via `ToolExecutionContext` — retrieve credentials in one line:

```python
@tool(name="send_email", description="Send an email")
async def send_email(
    to: str,
    subject: str,
    body: str,
    context: ToolExecutionContext
) -> str:
    creds = await context.credentials.get(context.tenant_id, "google")
    if not creds:
        return "Please connect your Google account first"

    # Business logic: send email using credentials
    provider = GmailProvider(creds)
    await provider.send_email(to, subject, body)
    return "Email sent"
```

Same in Agents:

```python
@flowagent(name="SendEmailAgent")
class SendEmailAgent(StandardAgent):
    async def on_running(self, msg):
        creds = await self.context.credentials.get(self.tenant_id, "google", "work")
        # ...
```

### 9.6 Application Layer Responsibilities

The framework handles storage and retrieval. The application layer handles:

1. **OAuth flow**: Application (Web/CLI) handles OAuth redirect -> obtains token -> calls `credential_store.save()`
2. **Token refresh**: Tool/Agent detects expired token -> refreshes -> calls `credential_store.save()` to update
3. **Multi-account management**: Application provides UI for users to connect multiple accounts (primary/work/personal)

### 9.7 KoiAI Migration

KoiAI's current credential flow:

```
Supabase oauth_accounts table -> AccountResolver -> ProviderFactory -> API call
```

After migration:

```
CredentialStore (SQLite/Redis) -> context.credentials.get() -> Provider -> API call
```

KoiAI's OAuth flow, AccountResolver, and ProviderFactory remain unchanged. Only the underlying storage changes from direct Supabase queries to the CredentialStore interface.

---

## 10. Streaming

### Event Stream Naturally Produced by the ReAct Loop

| Loop Phase | Event |
|------------|-------|
| LLM starts output | MESSAGE_START |
| LLM outputs text | MESSAGE_CHUNK |
| LLM output ends | MESSAGE_END |
| LLM decides to call tool | TOOL_CALL_START |
| Tool execution complete | TOOL_RESULT |
| Agent needs user input | STATE_CHANGE -> WAITING_FOR_INPUT |
| Agent needs approval | STATE_CHANGE -> WAITING_FOR_APPROVAL |
| Loop ends | EXECUTION_END |

The frontend can display real-time progress: "Checking weather..." -> "Weather done, searching flights..." -> "Please confirm sending email..."

Existing StreamEngine and emit_xxx methods are fully reused.

---

## 11. Concurrent Tool Execution

### 11.1 Why Concurrency

Personal agent tools are mostly remote API calls (email, calendar, weather, flights) — inherently independent and IO-intensive.
When the LLM returns multiple tool_calls in one turn, serial execution wastes time:

```
Serial:     get_weather(1s) -> search_flights(3s) -> check_calendar(1s) = 5s
Concurrent: get_weather(1s) + search_flights(3s) + check_calendar(1s) = 3s
```

### 11.2 Execution Strategy

```python
async def execute_tool_calls(tool_calls: list[ToolCall]) -> list[ToolResult]:
    # All tool_calls execute concurrently by default
    # Each call has independent timeout, no mutual interference
    tasks = [execute_with_timeout(tc, timeout=config.tool_execution_timeout) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Failed calls: error message as tool result, does not break the loop
    return [
        result if not isinstance(result, Exception)
        else ToolResult(tool_call_id=tc.id, content=f"Error: {result}", is_error=True)
        for tc, result in zip(tool_calls, results)
    ]
```

### 11.3 Special Handling for Agent-Tools

Agent-Tools may enter WAITING state (requiring user approval/input). If multiple Agent-Tools in one turn all need approval:

- **Execute concurrently** to each one's approval point
- **Collect all approval requests** and present to the user at once (rather than one by one)
- After user confirms each, resume execution in confirmation order

---

## 12. Lightweight Context Management

### 12.1 Design Principle

Personal agent conversations are typically short (3-10 ReAct turns) and don't need OpenClaw-style multi-stage LLM summarization.
Adopt a **truncation-first, simple discard** strategy, combined with the Momex memory system to ensure no information is lost.

### 12.2 ContextManager

```python
class ContextManager:
    """Lightweight context management with three lines of defense"""

    def truncate_tool_result(self, result: str) -> str:
        """Defense 1: Single tool result truncation (called immediately after each tool execution)"""
        max_tokens = int(config.context_token_limit * config.max_tool_result_share)
        max_chars = min(max_tokens * 4, config.max_tool_result_chars)  # ~4 chars/token

        if len(result) <= max_chars:
            return result

        truncated = result[:max_chars]
        # Try to truncate at newline boundary for readability
        last_newline = truncated.rfind('\n')
        if last_newline > len(truncated) // 2:
            truncated = truncated[:last_newline]
        return truncated + "\n[...truncated]"

    def trim_if_needed(self, messages: list) -> list:
        """Defense 2: History message trimming (called before each loop iteration)"""
        total = estimate_tokens(messages)
        if total <= config.context_token_limit * config.context_trim_threshold:
            return messages
        # Keep system prompt + most recent N turns, discard older ones
        system = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        kept = non_system[-config.max_history_messages:]
        return system + kept

    def force_trim(self, messages: list) -> list:
        """Defense 3: Force trim to safe range (triggered after context overflow)"""
        system = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        # Keep only the most recent 5 messages to ensure they fit
        return system + non_system[-5:]
```

### 12.3 Collaboration with Momex

Context trimming does not equal information loss:
- After each ReAct loop iteration ends, `post_process()` has already stored the complete conversation in Momex
- Momex auto-extracts entities, actions, preferences, and other structured knowledge
- On the next conversation, `prepare_context()` recalls relevant memories from Momex
- Effect: **discard short-term, retain long-term**

---

## 13. Structured Approval Flow

### 13.1 Why Structure is Needed

High-frequency personal agent operations involve irreversible actions (sending email, booking tickets, transfers). Approval is not an edge case but a core flow.
A simple WAITING_FOR_APPROVAL status marker is insufficient; structured approval requests are needed.

### 13.2 ApprovalRequest

```python
@dataclass
class ApprovalRequest:
    agent_name: str           # "SendEmailAgent"
    action_summary: str       # "Send email to team@company.com"
    details: dict             # {recipient, subject, body_preview}
    options: list[str]        # ["approve", "edit", "cancel"]
    timeout_minutes: int      # Auto-cancel on timeout
    allow_modification: bool  # Whether user can modify parameters before execution
```

### 13.3 Batch Approval

Multiple Agent-Tools in one loop iteration may all need approval:

```
Turn 1:
  LLM calls SendEmailAgent + BookFlightAgent simultaneously
  -> Both need approval
  -> Collected as [ApprovalRequest, ApprovalRequest]
  -> Presented to user at once

User response: "Approve the email, change the flight to a window seat"
  -> SendEmailAgent executes directly
  -> BookFlightAgent modifies parameters then executes
```

### 13.4 Approval Result Flow-back

After approval completes, the result is appended to messages as a tool result, and the ReAct loop continues:

```
tool_result = {
    "tool_call_id": "send_email_xxx",
    "content": "Email sent successfully to team@company.com"
}
```

If the user cancels an approval, it is similarly communicated to the LLM as a tool result:

```
tool_result = {
    "tool_call_id": "book_flight_xxx",
    "content": "User cancelled this action.",
    "is_error": true
}
```

The LLM decides subsequent behavior based on this (skip, provide alternatives, or summarize).

---

## 14. ReactLoopConfig

All ReAct loop configuration is centrally managed to avoid magic numbers scattered across files:

```python
@dataclass
class ReactLoopConfig:
    # Loop control
    max_turns: int = 10                          # Maximum loop iterations

    # Tool execution
    tool_execution_timeout: int = 30             # Single tool timeout (seconds)
    max_tool_result_share: float = 0.3           # Single tool result max 30% of context
    max_tool_result_chars: int = 400_000         # Single tool result hard limit (chars)

    # Context management
    context_token_limit: int = 128000            # Context window size
    context_trim_threshold: float = 0.8          # Trigger trimming at 80%
    max_history_messages: int = 40               # Max messages retained after trimming

    # LLM calls
    llm_max_retries: int = 2                     # Max LLM call retries
    llm_retry_base_delay: float = 1.0            # Retry base delay (seconds, exponential backoff)

    # Approval
    approval_timeout_minutes: int = 30           # Approval auto-cancel timeout
```

---

## 15. Proactive Trigger Engine (TriggerEngine)

### 15.1 Why the Framework Needs It Built-in

A personal agent cannot only passively respond to user messages. Proactive service is a core capability:

| Scenario | Trigger Type |
|----------|-------------|
| "Summarize unread emails every day at 8 AM" | Schedule trigger (Cron) |
| "Notify me when I receive an Amazon email" | Event trigger (Event Bus) |
| "Alert me when flight prices drop" | Condition trigger (polling + condition check) |
| "Remind me about the meeting at 3 PM" | One-time timer |

KoiAI has already implemented a complete TriggerEngine at the application layer. Moving it down to the framework allows all FlowAgents-based applications to use it out of the box.

### 15.2 Architecture: Migration from KoiAI

**KoiAI current architecture (application layer):**

```
TriggerEngine
  +-- ScheduleTrigger   -- cron / interval / one-time
  +-- EventTrigger      -- Redis Streams Event Bus event matching
  +-- ConditionTrigger  -- Periodic polling + condition check
  +-- ActionExecutor    -- Custom execution logic (NotifyExecutor, etc.)
        |
      Directly executes predefined actions (send notifications, etc.)
```

**After migration to framework:**

```
TriggerEngine
  +-- ScheduleTrigger   -- Unchanged
  +-- EventTrigger      -- Unchanged
  +-- ConditionTrigger  -- Unchanged
  +-- OrchestratorExecutor (new)
        |
      Generates TriggerMessage -> Orchestrator.handle_message()
        |
      ReAct loop makes autonomous decisions (LLM decides what to do)
```

### 15.3 Key Change: ActionExecutor -> Orchestrator

KoiAI's ActionExecutor is predefined ("send SMS on trigger", "call a specific Agent on trigger").
The framework layer changes to **trigger events entering the ReAct loop, with LLM autonomous decision-making**:

```python
class OrchestratorExecutor(ActionExecutor):
    """Converts trigger events to messages and delegates to the Orchestrator's ReAct loop"""

    async def execute(self, context: TriggerContext) -> ActionResult:
        # Convert trigger context to a virtual user message
        message = TriggerMessage(
            tenant_id=context.task.user_id,
            content=self._build_message(context),
            metadata={
                "source": "trigger",
                "trigger_type": context.trigger_type,
                "task_id": context.task.id,
            },
        )

        # Reuse existing ReAct loop
        response = await self.orchestrator.handle_message(message)

        # Push to user via NotificationChannel
        await self.notify(context.task.user_id, response, context.task.output)
        return ActionResult(success=True, output=response)

    def _build_message(self, context: TriggerContext) -> str:
        """Build context message based on trigger type"""
        if context.trigger_type == "schedule":
            return f"[Scheduled Task] {context.task.action.config.get('instruction', '')}"
        elif context.trigger_type == "event":
            return f"[Event Trigger] {context.event_type}: {json.dumps(context.event_data)}"
        elif context.trigger_type == "condition":
            return f"[Condition Trigger] Condition met: {context.condition_expression}"
```

### 15.4 Two Execution Paths for Triggered Tasks

Triggered tasks have two execution paths. The task creator chooses which executor to use:

#### Path 1: OrchestratorExecutor (ReAct Loop) — Default

The trigger event enters the ReAct loop. The LLM autonomously decides which tools to call and in what order. Suitable for tasks that need reasoning and flexible decision-making:

```
Trigger fires
  → OrchestratorExecutor generates TriggerMessage
  → "Check for new important emails, analyze importance, extract trips, track packages"
  → ReAct loop: LLM calls ReadEmailAgent → analyzes results → calls trip extraction → calls package tracking → decides to send SMS
  → NotificationChannel pushes result to user
```

The advantage: no hardcoded pipeline. The LLM figures out the steps based on available tools and intermediate results. New capabilities (new tools/agents) are automatically available to triggered tasks.

#### Path 2: Custom ActionExecutor — For High-Frequency / Deterministic Pipelines

For triggers that fire frequently (every few minutes) or have fixed deterministic logic, the full ReAct loop (LLM call per trigger) may be too expensive or too slow. Applications can implement custom `ActionExecutor` subclasses that execute predefined logic without LLM involvement:

```python
class EmailPipelineExecutor(ActionExecutor):
    """Deterministic email processing pipeline — no LLM per trigger"""

    async def execute(self, context: TriggerContext) -> ActionResult:
        emails = await self.fetch_new_emails(context.task.user_id)
        for email in emails:
            if self.is_important(email):  # Rule-based check
                await self.notify(context.task.user_id, email.summary)
        return ActionResult(success=True)
```

The two paths coexist in the same TriggerEngine. The task configuration specifies which executor to use:

```python
# Path 1: LLM-driven (flexible, more expensive)
await trigger_engine.create_task(
    user_id="user_123",
    trigger=ScheduleTrigger(cron="0 8 * * *"),
    executor="orchestrator",  # default
    action={"instruction": "Summarize my unread emails"},
)

# Path 2: Custom pipeline (deterministic, cheaper)
await trigger_engine.create_task(
    user_id="user_123",
    trigger=ScheduleTrigger(interval_minutes=5),
    executor="email_pipeline",  # registered custom executor
    action={"config": {"importance_threshold": 0.8}},
)
```

### 15.5 Components Reused from KoiAI

| Component | Source | Changes |
|-----------|--------|---------|
| TriggerEngine | `koiai/core/triggers/engine.py` | Moved to framework, remove KoiAI-specific dependencies |
| ScheduleTrigger | `koiai/core/triggers/trigger_types/schedule.py` | Unchanged (cron/interval/one-time) |
| EventTrigger | `koiai/core/triggers/trigger_types/event.py` | Unchanged (source+type+filter matching) |
| ConditionTrigger | `koiai/core/triggers/trigger_types/condition.py` | Complete implementation (placeholder in KoiAI) |
| Task model | `koiai/core/triggers/models.py` | Unchanged (ACTIVE/PAUSED/DISABLED/COMPLETED) |
| EventBus | `koiai/core/events/bus.py` | Extract as pluggable interface (Redis / In-Memory) |
| NotifyExecutor | KoiAI-specific | **Not migrated**, application layer implements as custom ActionExecutor |
| AgentExecutor | KoiAI-specific | **Replaced by** OrchestratorExecutor |
| EmailPipelineExecutor | KoiAI `email_handler.py` | Application layer implements as custom ActionExecutor (deterministic pipeline) |

### 15.6 NotificationChannel

Trigger results need to be pushed to users (user may be offline or not in a conversation). The framework defines the interface; the application layer implements:

```python
class NotificationChannel(ABC):
    """Push channel for trigger results"""

    @abstractmethod
    async def send(self, tenant_id: str, message: str, metadata: dict) -> bool:
        ...

# Application layer implementation examples
class SMSNotification(NotificationChannel): ...
class PushNotification(NotificationChannel): ...
class EmailNotification(NotificationChannel): ...
class WebSocketNotification(NotificationChannel): ...
```

### 15.7 EventBus Interface

Extracted from KoiAI's Redis Streams implementation as a pluggable interface:

```python
class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    async def subscribe(self, pattern: str, callback: Callable) -> None: ...

    @abstractmethod
    async def unsubscribe(self, pattern: str) -> None: ...

# Framework provides two implementations
class InMemoryEventBus(EventBus): ...      # For development/testing
class RedisStreamEventBus(EventBus): ...   # For production (migrated from KoiAI)
```

### 15.8 Configuration and Initialization

```python
orchestrator = Orchestrator(
    llm_client=llm,
    memory_provider=MomexMemoryProvider(config),
    trigger_engine=TriggerEngine(
        event_bus=RedisStreamEventBus(redis_url),
        executor=OrchestratorExecutor(orchestrator),
        notification_channels=[SMSNotification(), PushNotification()],
    ),
)

# Create a scheduled task
await orchestrator.trigger_engine.create_task(
    user_id="user_123",
    trigger=ScheduleTrigger(cron="0 8 * * *", timezone="Asia/Shanghai"),
    action={"instruction": "Summarize my unread emails, send SMS for important ones"},
    output={"channel": "sms"},
)
```

### 15.9 Relationship with ReAct Loop

```
                    +-----------------------------+
                    |        TriggerEngine         |
                    |  Schedule / Event / Condition |
                    +-------------+---------------+
                                  | TriggerMessage
                                  v
User Message --> Orchestrator.handle_message()
                    |
                    +-- prepare_context()    <- Momex recalls memories
                    +-- should_process()
                    +-- check_pending_agents()
                    +-- react_loop()         <- LLM autonomous decision
                    +-- post_process()       <- Momex stores knowledge
                                  |
                                  v
                    +-----------------------------+
                    |     NotificationChannel      |
                    |   SMS / Push / Email / WS    |
                    +-----------------------------+
```

The Orchestrator does not need to know whether the message comes from a user or a trigger. The ReAct loop treats both equally.

### 15.10 Approval Handling for Triggered Tasks

When a triggered task enters the ReAct loop, an Agent-Tool may need user approval, but the user may be offline. Handling strategy: **queue for user, with TTL**.

**Flow:**

```
Scheduled task triggers -> ReAct loop -> SendEmailAgent needs approval
  -> Approval request pushed to user via NotificationChannel ("An email awaits your confirmation")
  -> Agent stored in Pool (with TTL, default approval_timeout_minutes)
  -> ReAct loop breaks, triggered task marked as PENDING_APPROVAL
  -> When user comes online:
      - Actively query pending approvals list (orchestrator.list_pending_approvals(tenant_id))
      - Or reply directly via NotificationChannel link
  -> User confirms -> Agent resumes execution -> Result pushed via NotificationChannel
  -> TTL expires without confirmation -> Agent removed from Pool, task marked as EXPIRED
```

**Query interface:**

```python
# User can query all tasks awaiting approval
pending = await orchestrator.list_pending_approvals(tenant_id="user_123")
# Returns: [
#   {agent_name: "SendEmailAgent", action_summary: "Send email to team@...",
#    created_at: "...", expires_at: "...", source: "trigger", task_id: "..."},
# ]
```

---

## 16. Extension Points

All existing extension points are preserved. Developers customize behavior by subclassing Orchestrator:

| Extension Point | Purpose | Change |
|-----------------|---------|--------|
| `prepare_context()` | Load memories, user profile, permissions, conversation history | Unchanged |
| `should_process()` | Safety checks, rate limiting, tier control | Unchanged |
| `post_process()` | Store memories, send notifications, persona wrapping, record usage | Unchanged |
| `reject_message()` | Custom response for rejected messages | Unchanged |
| `@callback_handler` | Agent access to external services (send SMS, read cache) | Unchanged |
| `create_agent()` | Custom Agent instantiation | Preserved, called within ReAct loop |
| `execute_agent()` | Custom Agent execution logic | Preserved, called within ReAct loop |
| `route_message()` | Custom routing logic | **Removed** |

---

## 17. Extensibility

### Adding New Capabilities

Write a `@tool` function or a `@flowagent` Agent, register it, and the Orchestrator can use it automatically.

No need to:
- Configure trigger keywords
- Modify routing logic
- Modify Orchestrator code

The LLM autonomously understands when to use it through the tool's description.

### Controlling ReAct Behavior

Orchestrator-level configuration:
- `system_prompt` — **Persona injection point.** The ReAct loop's LLM uses this as its personality. All responses are generated in this voice natively, eliminating the need for a separate persona wrapping layer. If not provided, the LLM uses its default behavior.
- `max_turns` — Maximum loop iterations
- `max_tokens_budget` — Total token budget (optional)
- `tool_names` — Regular Tools exposed to the LLM
- `agent_tool_filter` — Agent-Tools exposed to the LLM

---

## 18. Impact on KoiAI

### No Changes Needed

- All 25+ Agent implementation code
- Agent InputField / OutputField definitions
- Agent approval flow (`needs_approval()` / `parse_approval_async()` pattern unchanged)
- Agent tool usage
- FastAPI routes and endpoints
- Redis / Supabase storage
- OAuth and authentication (OAuthClient unchanged, underlying storage migrates to CredentialStore interface)

### Changes Needed

- `KoiOrchestrator`'s `handle_message` internal logic (routing -> ReAct loop)
- Remove `MessageRouter`, LLM routing related code, and `routing_llm_provider` (routing is eliminated; the ReAct LLM handles both intent understanding and response generation in one call)
- `agent_registry.yaml` triggers field becomes optional (LLM understands intent via description)
- **KoiAgent persona wrapping eliminated**: Koi's personality prompt (`PERSONALITY_PROMPT`) becomes the Orchestrator's `system_prompt`. The ReAct LLM natively speaks as Koi. KoiAgent's 6 response wrapping methods (`_wrap_completed_result`, `_generate_input_request`, `_generate_approval_request`, `_generate_tier_upgrade_message`, `_generate_error_response`, `_generate_clarification`) are all removed — the ReAct LLM handles all response formatting in Koi's voice. KoiAgent remains only as a registered `@flowagent` for its `on_running` chat capability (google_search, important_dates tools).
- **Chat history unified into Momex**: Remove separate `chat_history` Supabase table. Conversation history is managed by MomexMemoryProvider's short-term memory. `prepare_context()` loads history via `memory.get_history()`, `post_process()` saves via `memory.save_history()`.
- **Auto profile detection stays in post_process()**: KoiAgent's `_detect_and_update_profile()` moves to `KoiOrchestrator.post_process()` as a background task.
- TriggerEngine replaced with framework built-in version (interface-compatible, ActionExecutor -> OrchestratorExecutor). KoiAI's `email_handler.py` pipeline can either use OrchestratorExecutor (LLM-driven) or be reimplemented as a custom ActionExecutor (deterministic, for high-frequency triggers).
- Implement KoiAI-specific NotificationChannels (SMS / Push)

### Benefits Gained

- Multi-Agent collaboration (one message triggers multiple Agents)
- Free conversation ability (LLM responds directly when no Agent matches)
- More accurate intent understanding (LLM native tool calling beats keyword matching)
- More flexible field extraction (LLM extracts parameters directly when calling Agent-Tools)
- Triggered tasks can leverage the ReAct loop (LLM autonomous decisions instead of predefined actions)

---

## 19. Implementation Plan

### Phase 1a: Minimum Viable ReAct Loop

Get end-to-end working first to validate the core flow:

1. Implement `ReactLoopConfig` centralized configuration
2. Implement `react_loop()` basic logic in Orchestrator (serial tool_calls execution)
3. Implement Agent-Tool schema auto-generation (read InputFields from AgentRegistry -> tool schema)
4. Implement Agent-Tool execution logic (create Agent, inject parameters, handle incomplete state)
5. Modify `handle_message()` flow (Pool check -> ReAct loop)
6. Implement `stream_message()` ReAct version
7. Return `ReactLoopResult` (response, turns, token_usage, duration_ms)

### Phase 1b: Engineering Robustness

Build robustness on top of 1a:

1. Implement `ContextManager` (tool result truncation + history trimming + force trim — three lines of defense)
2. Concurrent tool execution (`asyncio.gather` + independent timeouts)
3. Tool execution error isolation (failures returned as tool results to LLM)
4. LLM call retry + exponential backoff + error classification handling
5. Implement structured approval flow (ApprovalRequest, batch approval collection, approval result flow-back)
6. `list_pending_approvals()` query interface

### Phase 2: CredentialStore

1. Implement `CredentialStore` interface (save / get / list / delete)
2. Implement SQLite backend (default)
3. Implement Redis backend (optional)
4. Inject `CredentialStore` into `ToolExecutionContext` (`context.credentials`)
5. Create CredentialStore instance during Orchestrator initialization

### Phase 3: Memory Provider (Momex Integration)

1. Define MemoryProvider interface
2. Implement MomexMemoryProvider (based on typeagent-py's Memory + ShortTermMemory API)
3. Integrate memory recall in `prepare_context()` (`memory.search()`)
4. Integrate knowledge storage in `post_process()` (`memory.add(messages, infer=True)`)
5. Accept memory_provider parameter during Orchestrator initialization

### Phase 4: TriggerEngine (Proactive Triggers)

1. Migrate TriggerEngine core from KoiAI (remove KoiAI-specific dependencies)
2. Migrate ScheduleTrigger (cron / interval / one-time)
3. Migrate EventTrigger (event matching + filter)
4. Extract EventBus interface + InMemoryEventBus (development) + RedisStreamEventBus (production)
5. Implement OrchestratorExecutor (trigger event -> TriggerMessage -> Orchestrator.handle_message())
6. Define NotificationChannel interface
7. Complete ConditionTrigger implementation (placeholder in KoiAI)

### Phase 5: KoiAI Adaptation

1. Migrate KoiOrchestrator to ReAct mode
2. Remove routing-related code
3. Replace KoiAI's TriggerEngine with framework built-in version
4. KoiAI implements its own NotificationChannels (SMS / Push)
5. Test all existing Agents for compatibility in ReAct mode

---

## 20. OpenClaw Reference Analysis

> Compare with OpenClaw's ReAct Agent implementation to extract lessons learned and issues to avoid.

### 20.1 Designs Worth Borrowing

#### 1) Onion-style Degradation Chain

OpenClaw has multi-layer auto-recovery strategies on LLM call failure, each layer independent and decoupled:

```
LLM call fails
  -> Auth Profile rotation (switch API Key)
    -> Thinking Level degradation (high thinking -> low thinking)
      -> Model Fallback (switch model)
        -> Final error

Context overflow
  -> Auto-Compaction (LLM multi-stage summarization, max 3 attempts)
    -> Tool Result Truncation (truncate oversized tool results)
      -> Final error
```

**Insight**: `react_loop()` needs a similar degradation chain. At minimum:
- Retry + exponential backoff on LLM call failure
- Auto compression/truncation on context overflow
- Tool execution timeout/exception isolation (one Agent crash should not break the entire loop)
- Correction when LLM returns non-existent tool names

#### 2) Tool Result Truncation Mechanism

Tool return values are the biggest culprit for context bloat. OpenClaw's strategy:
- Single tool result cannot exceed 30% of context
- Hard limit of 400K characters
- Truncation at newline boundaries for readability, keeping at least 2000 characters at the beginning
- Appended truncation warning marker

**Insight**: `react_loop()` should immediately check length after receiving tool results, truncating before appending to messages if over limit.

#### 3) Context Auto-Compaction

The ReAct loop naturally causes rapid context growth (tool_call + result accumulation per turn). OpenClaw's approach:
- Triggered by token threshold (default ~88% of context)
- Chunks historical messages, uses LLM to generate summary for each chunk
- Merges summaries, replaces history, keeps most recent N raw messages
- Auto-repairs tool_use / tool_result pairings during compaction (removes "orphan" results)
- Session branching mechanism preserves original data, supports rollback

**Insight**: Add ContextManager to Orchestrator, check token usage before each loop iteration, summarize history when threshold exceeded. Simplified approach: keep most recent N messages + generate a summary of older messages injected into system prompt.

#### 4) Loop Runtime Transcript

OpenClaw uses JSONL format to completely record each loop iteration's input/output, tool calls, token usage, compaction count, etc.

**Insight**: `react_loop()` should maintain a structured execution record internally for debugging and observability. Including each turn's LLM input/output, tool call duration, and token consumption attribution.

### 20.2 Issues to Avoid

#### 1) Logic Fragmentation

OpenClaw's compaction-related logic is scattered across 7+ files (compact.ts, compaction.ts, tool-result-truncation.ts, agent-runner-memory.ts, memory-flush.ts, run.ts, session-updates.ts). Understanding "what happens when context is full" requires jumping across 7 files to piece together the puzzle — extremely high onboarding cost for newcomers.

**Avoidance**: Aggregate context management into a single `ContextManager` class, exposing `should_compact()`, `compact()`, `truncate_tool_result()` methods with cohesive responsibilities.

#### 2) Deeply Nested Error Handling

Error handling in the main loop is a massive if-else chain (context overflow -> prompt error -> assistant error, each with 5-6 subtypes underneath), approximately 400 lines. Untestable, unverifiable, and adding new error types easily affects existing paths.

**Avoidance**: Use strategy pattern for error handling. Register a handler for each error type; the main loop only dispatches:

```python
error_handlers = {
    ContextOverflowError: handle_context_overflow,
    AuthError: handle_auth_rotation,
    RateLimitError: handle_rate_limit,
    ToolExecutionError: handle_tool_error,
}
```

#### 3) Over-engineered Multi-stage Summarization

OpenClaw's compaction requires 3 LLM calls (chunk summary x2 + merge x1). If the summary request itself overflows, there are further degradation strategies (summarize only small messages, only record counts). Solving complexity with complexity.

**Avoidance**: Adopt a simpler compression strategy. Prioritize truncating tool_results (the biggest token consumers). Conversation text is usually small; simple "keep most recent N messages + discard older ones" is sufficient in most scenarios. Only upgrade to LLM summarization when there's a clear need.

#### 4) Memory Flush and Compaction Responsibility Overlap

OpenClaw runs a Memory Flush before compaction (letting the agent save important information to disk), but the flush itself is an LLM call that consumes tokens, generates new messages, and may make context even fuller. The two mechanisms have overlapping goals.

**Avoidance**: If a memory system exists, extract key information and store in memory during compaction — no need for a separate flush phase. One step accomplishes both things.

#### 5) Magic Numbers Without Centralized Management

OpenClaw has numerous constants scattered across files (reserve tokens 20000, soft threshold 4000, max tool result share 0.3, chunk ratio 0.4, etc.), with implicit dependencies between them but no explicit constraints.

**Avoidance**: Centralize all ReAct loop configuration into a single config dataclass:

```python
@dataclass
class ReactLoopConfig:
    max_turns: int = 10
    max_tool_result_share: float = 0.3
    context_trim_threshold: float = 0.8  # Trigger trimming at 80% of context
    max_overflow_recovery_attempts: int = 3
    tool_execution_timeout: int = 30     # seconds
```

### 20.3 Summary: Borrowing Strategy

| Item | Priority | Notes |
|------|----------|-------|
| Tool Result Truncation | **P0** | Prevent single tool result from blowing up context; simple to implement |
| LLM Call Error Classification and Retry | **P0** | Production requirement; use strategy pattern |
| Context Auto-Compression | **P0** | Start with simple approach (keep most recent N), upgrade later |
| Agent-Tool Parallel Execution | **P1** | Independent tool_calls from one LLM turn can run in parallel |
| Loop Transcript Recording | **P1** | Debugging and observability |
| ReactLoopConfig Centralized Config | **P1** | Avoid magic numbers scattered across files |
| ContextManager Aggregate Class | **P2** | Code organization to prevent logic fragmentation |

---

## 21. Change Summary

| Component | Change | Scope |
|-----------|--------|-------|
| Orchestrator.handle_message | route+execute -> ReAct loop | **Medium** |
| Orchestrator.system_prompt | New persona injection point (replaces KoiAgent wrapping) | **New parameter** |
| Orchestrator.react_loop | New (with concurrent execution, error recovery) | **New method** |
| ReactLoopConfig | New centralized config | **New file** |
| ContextManager | New (truncation + trimming + force trim) | **New file** |
| ApprovalRequest | New structured approval | **New file** |
| Agent-Tool schema generation | New | **Small** |
| MessageRouter | **Removed** | Deleted |
| Agent Pool | Entry path changed, interface unchanged | **None** |
| StandardAgent | Unchanged | **None** |
| Tool system | Unchanged | **None** |
| @flowagent decorator | Optional new expose_as_tool parameter | **Minimal** |
| Streaming | Unchanged | **None** |
| Extension Points | route_message removed, others unchanged | **Minimal** |
| CredentialStore | New (SQLite/Redis backends) | **New file** |
| ToolExecutionContext | New credentials field | **Small** |
| MemoryProvider | New interface + MomexMemoryProvider implementation | **New file** |
| TriggerEngine | Migrated from KoiAI to framework | **Migration** |
| EventBus | New interface + InMemory/Redis implementations | **New file** |
| OrchestratorExecutor | New (trigger event -> ReAct loop) | **New file** |
| NotificationChannel | New interface (application layer implements) | **New file** |
