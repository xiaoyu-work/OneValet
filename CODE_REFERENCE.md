# OneValet ReAct Loop - Exact Code References

## Quick Reference by Requirement

### 1. Tool Result Processing Flow

**Entry Point**: react_loop.py:391 (in _react_loop_events)

```python
# Tool execution
timed_results = [TimedResult(r, d) for r, d in zip(results, durations)]

# Result processing loop
for tc, result in zip(tool_calls, results):
    is_agent = self._is_agent_tool(tc.name)
    
    # [Lines 428-433] Check if agent waiting for input
    if isinstance(result, AgentToolResult) and not result.completed:
        # Handle incomplete agent-tool...
    else:
        # [Lines 472-489] Handle completed result
        if isinstance(result, AgentToolResult):
            result_text = result.result_text
            r_meta = result.metadata if isinstance(result.metadata, dict) else {}
            tool_trace = r_meta.get("tool_trace") or []
        else:
            result_text = str(result) if result is not None else ""
            tool_trace = []
        
        result_chars_original = len(result_text)
        original_len = len(result_text)
        
        # [Line 482] First truncation: hard cap
        result_text = self._cap_tool_result(result_text)
        
        # [Line 483] Second truncation: context-aware
        result_text = self._context_manager.truncate_tool_result(result_text)
        
        # Additional agent-tool truncation
        if is_agent and len(result_text) > 2000:
            result_text = result_text[:1500] + f"\n...[truncated from {original_len} to 1500 chars]"
        elif len(result_text) < original_len:
            result_text += f"\n...[truncated from {original_len} to {len(result_text)} chars]"
        
        # [Line 489] Build message
        messages.append(self._build_tool_result_message(tc.id, result_text))
```

---

### 2. Tool Execution Call Chain

**Location**: tool_manager.py:226-267

```python
async def _execute_single(
    self,
    tool_call: Any,  # Has .name, .arguments, .id
    tenant_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    request_tools: Optional[List] = None,
    request_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Dispatch to agent-tool or regular tool execution."""
    
    # [Line 237] Parse arguments
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else json.loads(tool_call.arguments)

    if self._is_agent_tool(tool_call.name):
        # Agent-tool path (line 247)
        task_instruction = args.pop("task_instruction", "")
        return await execute_agent_tool(
            self,
            agent_type=tool_call.name,
            tenant_id=tenant_id,
            tool_call_args=args,
            task_instruction=task_instruction,
            request_context=request_context,
        )
    else:
        # Builtin tool path (lines 249-260)
        tools = request_tools if request_tools is not None else getattr(self, 'builtin_tools', [])
        tool = next((t for t in tools if t.name == tool_call.name), None)
        
        if not tool:
            return f"Error: Tool '{tool_call.name}' not found"

        # Create execution context
        context = AgentToolContext(
            tenant_id=tenant_id,
            credentials=self.credential_store,
            metadata=self._build_tool_metadata(metadata),
        )
        
        # [Line 260] CALL EXECUTOR
        return await tool.executor(args, context)
```

---

### 3. AgentTool Registration Pattern

**Location**: tool_manager.py:80-182

```python
def _build_builtin_tools(self) -> List[AgentTool]:
    """Build orchestrator's builtin tools as AgentTool instances."""
    
    # Import executor and schema
    from ..builtin_agents.tools.google_search import (
        google_search_executor, GOOGLE_SEARCH_SCHEMA,
    )
    
    tools: List[AgentTool] = []
    
    # Create tool instance
    tools.append(AgentTool(
        name="google_search",
        description="Search the web using Google...",
        parameters=GOOGLE_SEARCH_SCHEMA,  # JSON Schema
        executor=google_search_executor,   # Async callable
        category="web",
    ))
    
    return tools
```

**Schema Location**: builtin_agents/tools/google_search.py

```python
GOOGLE_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results (1-10)",
            "default": 5
        }
    },
    "required": ["query"]
}
```

---

### 4. Media Handling in LLM Client

**Location**: litellm_client.py:138-181 (in _call_api)

```python
async def _call_api(
    self,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> LLMResponse:
    """Make a non-streaming call via litellm.acompletion."""
    import litellm

    # [Line 148-150] Extract and apply media
    media = kwargs.pop("media", None)
    if media and messages:
        messages = self._add_media_to_messages_openai(messages, media)

    model = kwargs.get("model") or self._litellm_model
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,  # Now has media embedded
        **self._model_params(self.config.model, **kwargs),
        **self._base_kwargs,
    }

    if tools:
        params["tools"] = tools
        params["tool_choice"] = kwargs.get("tool_choice", "auto")

    # [Line 181] Make API call
    response = await litellm.acompletion(**params)
    
    # ... parse response and return LLMResponse
```

**Base Class Method**: base.py:246-294

```python
def _add_media_to_messages_openai(
    self,
    messages: List[Dict[str, Any]],
    media: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Add media (images) to the last user message."""
    
    if not media:
        return messages

    # Copy messages to avoid mutation
    messages = [msg.copy() for msg in messages]
    
    # Find last user message (backwards scan)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text_content = messages[i].get("content", "")
            content_parts = []

            # Add text if present
            if text_content:
                content_parts.append({"type": "text", "text": text_content})

            # Add images
            for item in media:
                if item.get("type") == "image":
                    data = item.get("data", "")
                    media_type = item.get("media_type", "image/jpeg")

                    if data.startswith(("http://", "https://")):
                        # URL reference
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data}
                        })
                    else:
                        # Base64 embedded
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })

            # Replace content with multimodal parts
            messages[i]["content"] = content_parts
            break

    return messages
```

---

### 5. Tool Result Message Building

**Location**: react_loop.py:703-716

```python
def _build_tool_result_message(
    self,
    tool_call_id: str,
    content: str,
    is_error: bool = False,
) -> Dict[str, Any]:
    """Build a tool result message for the LLM messages list."""
    
    if is_error:
        content = f"[ERROR] {content}"
    
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
```

**Called from**: react_loop.py:489

```python
messages.append(self._build_tool_result_message(tc.id, result_text))
```

---

### 6. Truncation Methods

**Hard Cap** (tool_manager.py:275-286):

```python
def _cap_tool_result(self, result_text: str) -> str:
    """Hard cap on tool result size to prevent context window overflow."""
    
    if len(result_text) <= TOOL_RESULT_HARD_CAP_CHARS:  # 400_000
        return result_text
    
    cut = TOOL_RESULT_HARD_CAP_CHARS
    
    # Try to cut at newline boundary
    newline_pos = result_text.rfind("\n", int(cut * 0.8), cut)
    if newline_pos > 0:
        cut = newline_pos
    
    logger.warning(
        f"[ReAct] Tool result truncated: {len(result_text)} -> {cut} chars"
    )
    
    return result_text[:cut] + "\n\n[truncated - result exceeded size limit]"
```

**Context-Aware Truncation** (context_manager.py:92-115):

```python
def truncate_tool_result(self, result: str) -> str:
    """Truncate a single tool result to stay within budget.
    
    Budget = min(
        context_token_limit * max_tool_result_share * 4,  # chars
        max_tool_result_chars
    )
    """
    
    max_chars = int(
        min(
            self.config.context_token_limit * self.config.max_tool_result_share * 4,
            self.config.max_tool_result_chars,
        )
    )
    
    if len(result) <= max_chars:
        return result

    # Try to cut at newline
    cut = result[:max_chars]
    newline_pos = cut.rfind("\n")
    if newline_pos > max_chars // 2:
        cut = cut[: newline_pos + 1]

    return cut + "\n[...truncated]"
```

---

### 7. Final Response Event

**Location**: react_loop.py:611-626

```python
yield AgentEvent(
    type=EventType.EXECUTION_END,
    data={
        "duration_ms": duration_ms,
        "turns": turn,
        "tool_calls_count": len(all_tool_records),
        "final_response": final_response,  # ← Main response text
        "result_status": result_status,  # COMPLETED, WAITING_FOR_INPUT, etc.
        "pending_approvals": pending_approvals,  # Approval requests if any
        "token_usage": {
            "input_tokens": total_usage.input_tokens,
            "output_tokens": total_usage.output_tokens,
        },
        "tool_calls": [dataclasses.asdict(r) for r in all_tool_records],
    },
)
```

---

### 8. Token Estimation

**Location**: context_manager.py:36-73

```python
def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
    """Estimate token count from messages."""
    
    total = 0
    for msg in messages:
        total += TOKENS_PER_MESSAGE_OVERHEAD  # 4
        content = msg.get("content")
        
        if content is None:
            continue
        
        if isinstance(content, str):
            total += self._estimate_string_tokens(content)
        elif isinstance(content, list):
            # Multimodal content (text + images)
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type in ("image_url", "image"):
                        total += IMAGE_TOKEN_ESTIMATE  # 170 tokens/image
                        continue
                    text = part.get("text") or part.get("content", "")
                    if isinstance(text, str):
                        total += self._estimate_string_tokens(text)
        
        # Tool calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total += TOOL_CALL_STRUCTURE_OVERHEAD_TOKENS  # 20
                args = tc.get("arguments") or tc.get("function", {}).get("arguments", "")
                if isinstance(args, str):
                    total += len(args) // JSON_CHARS_PER_TOKEN  # 3 chars/token
    
    return total
```

**Constants** (constants.py):
- `IMAGE_TOKEN_ESTIMATE = 170`
- `TOOL_RESULT_HARD_CAP_CHARS = 400_000`
- `TOKENS_PER_MESSAGE_OVERHEAD = 4`
- `TOOL_CALL_STRUCTURE_OVERHEAD_TOKENS = 20`
- `TEXT_CHARS_PER_TOKEN = 4`
- `JSON_CHARS_PER_TOKEN = 3`

---

### 9. AgentToolContext Structure

**Location**: models.py:49-63

```python
@dataclass
class AgentToolContext:
    """Context passed to tool executors."""
    
    llm_client: Optional["BaseLLMClient"] = None
    tenant_id: str = ""
    user_profile: Optional[Dict[str, Any]] = None
    context_hints: Optional[Dict[str, Any]] = None
    credentials: Optional["CredentialStore"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Created at**: tool_manager.py:255-259

```python
context = AgentToolContext(
    tenant_id=tenant_id,
    credentials=self.credential_store,
    metadata=self._build_tool_metadata(metadata),
)
```

---

### 10. AgentToolResult Structure

**Location**: agent_tool.py:31-39

```python
@dataclass
class AgentToolResult:
    """Result from executing an agent as a tool in the ReAct loop."""
    
    completed: bool  # Terminal status
    result_text: str = ""  # Main response text
    agent: Optional[Any] = None  # Agent instance (for incomplete states)
    approval_request: Optional[Any] = None  # For approval waits
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra data
```

**Returned from**: agent_tool.py:79-243 (execute_agent_tool)

Example returns:
- Completed: `AgentToolResult(completed=True, result_text="Success", metadata={...})`
- Waiting for input: `AgentToolResult(completed=False, result_text="Need more info", agent=agent, metadata={...})`
- Error: `AgentToolResult(completed=True, result_text="Error: ...", metadata={...})`

---

### 11. Event Types

**Location**: streaming/models.py:31-71

```python
class EventType(str, Enum):
    """Types of events that can be streamed"""
    
    # Message events (relevant to responses)
    MESSAGE_START = "message_start"
    MESSAGE_CHUNK = "message_chunk"
    MESSAGE_END = "message_end"
    
    # Tool events
    TOOL_RESULT = "tool_result"
    
    # Execution lifecycle
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"  # ← Final event with complete data
```

**Message events sequence** (react_loop.py:527-529):

```python
yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
yield AgentEvent(
    type=EventType.MESSAGE_CHUNK, 
    data={"chunk": final_response}  # ← User-visible text
)
yield AgentEvent(type=EventType.MESSAGE_END, data={})
```

**Followed by** (react_loop.py:611-626):

```python
yield AgentEvent(
    type=EventType.EXECUTION_END,
    data={...}  # ← Complete metadata
)
```

