# OneValet ReAct Loop and Tool System - Deep Dive Analysis

## 1. ReAct Loop Tool Result Handling

### Location
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/orchestrator/react_loop.py`
- **Class**: `ReactLoopMixin`

### Lines 460-500: Tool Result Processing

**Lines 450-490** - Tool result handling after tool execution:

```python
450.                             type=EventType.TOOL_RESULT,
451.                             data={
452.                                 "tool_name": tc_name, "call_id": tc.id,
453.                                 "kind": "agent", "success": True,
454.                                 "waiting": True, "status": waiting_status,
455.                                 "result_preview": waiting_text[:240],
456.                                 "tool_trace": tool_trace,
457.                             },
458.                         )
459.                         yield AgentEvent(
460.                             type=EventType.STATE_CHANGE,
461.                             data={"agent_type": tc_name, "status": waiting_status},
462.                         )
463.                         self._audit.log_tool_execution(...)
468.                         loop_broken = True
469.                         loop_broken_text = waiting_text
470. 
471.                     else:
472.                         if isinstance(result, AgentToolResult):
473.                             result_text = result.result_text
474.                             r_meta = result.metadata if isinstance(result.metadata, dict) else {}
475.                             tool_trace = r_meta.get("tool_trace") or []
476.                         else:
477.                             result_text = str(result) if result is not None else ""
478.                             tool_trace = []
479.                         result_chars_original = len(result_text)
480.                         original_len = len(result_text)
481.                         # Hard cap on tool result size
482.                         result_text = self._cap_tool_result(result_text)
483.                         result_text = self._context_manager.truncate_tool_result(result_text)
484.                         if is_agent and len(result_text) > 2000:
485.                             result_text = result_text[:1500] + f"\n...[truncated from {original_len} to 1500 chars]"
486.                         elif len(result_text) < original_len:
487.                             result_text += f"\n...[truncated from {original_len} to {len(result_text)} chars]"
488.                         logger.info(f"[ReAct]   {kind}={tc_name} OK ({len(result_text)} chars)")
489.                         messages.append(self._build_tool_result_message(tc.id, result_text))
490.                         all_tool_records.append(ToolCallRecord(...))
```

**Key observations**:
- Line 472-478: Tool results can be either `AgentToolResult` objects (structured) or raw strings
- Line 472: Checks `isinstance(result, AgentToolResult)` to extract text and metadata
- Line 474: Extracts `result.metadata` and gets `tool_trace` from it
- Line 482-483: **Two-stage truncation**:
  1. `_cap_tool_result()` - Hard character limit
  2. `_context_manager.truncate_tool_result()` - Context-aware truncation

### `_build_tool_result_message` Method (Lines 703-716)

```python
703.    def _build_tool_result_message(
704.        self,
705.        tool_call_id: str,
706.        content: str,
707.        is_error: bool = False,
708.    ) -> Dict[str, Any]:
709.        """Build a tool result message for the LLM messages list."""
710.        if is_error:
711.            content = f"[ERROR] {content}"
712.        return {
713.            "role": "tool",
714.            "tool_call_id": tool_call_id,
715.            "content": content,
716.        }
```

**Current limitations**:
- Only accepts `content: str` - **no multimodal support**
- Returns a dict with `role="tool"`, `tool_call_id`, and `content` (string only)
- Cannot carry image data or other media types

### `_cap_tool_result` Method (Lines 275-286 in tool_manager.py)

```python
275.    def _cap_tool_result(self, result_text: str) -> str:
276.        """Hard cap on tool result size to prevent context window overflow."""
277.        if len(result_text) <= TOOL_RESULT_HARD_CAP_CHARS:
278.            return result_text
279.        cut = TOOL_RESULT_HARD_CAP_CHARS
280.        newline_pos = result_text.rfind("\n", int(cut * 0.8), cut)
281.        if newline_pos > 0:
282.            cut = newline_pos
283.        logger.warning(
284.            f"[ReAct] Tool result truncated: {len(result_text)} -> {cut} chars"
285.        )
286.        return result_text[:cut] + "\n\n[truncated - result exceeded size limit]"
```

**Constant**: `TOOL_RESULT_HARD_CAP_CHARS = 400_000` (from `/onevalet/orchestrator/constants.py` line 37)

### `_context_manager.truncate_tool_result` Method (Lines 92-115 in context_manager.py)

```python
92.    def truncate_tool_result(self, result: str) -> str:
93.        """Truncate a single tool result to stay within budget.
94.
95.        The budget is the smaller of:
96.          - context_token_limit * max_tool_result_share * 4  (chars)
97.          - max_tool_result_chars
98.        Truncation prefers a newline boundary when possible.
99.        """
100.        max_chars = int(
101.            min(
102.                self.config.context_token_limit * self.config.max_tool_result_share * 4,
103.                self.config.max_tool_result_chars,
104.            )
105.        )
106.        if len(result) <= max_chars:
107.            return result
108.
109.        # Try to cut at the last newline within the budget
110.        cut = result[:max_chars]
111.        newline_pos = cut.rfind("\n")
112.        if newline_pos > max_chars // 2:
113.            cut = cut[: newline_pos + 1]
114.
115.        return cut + "\n[...truncated]"
```

**Configuration parameters** (from `ReactLoopConfig`):
- `context_token_limit`: Maximum context window size
- `max_tool_result_share`: Maximum fraction of context a single tool result can occupy
- `max_tool_result_chars`: Absolute character limit per tool result

---

## 2. Tool Registration System

### Location
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/orchestrator/tool_manager.py`
- **Class**: `ToolManagerMixin`

### Google Search Registration (Lines 107-114)

```python
107.        # Google search
108.        tools.append(AgentTool(
109.            name="google_search",
110.            description="Search the web using Google. Returns titles, URLs, and snippets of top results.",
111.            parameters=GOOGLE_SEARCH_SCHEMA,
112.            executor=google_search_executor,
113.            category="web",
114.        ))
```

**Imports**:
```python
86.        from ..builtin_agents.tools.google_search import (
87.            google_search_executor, GOOGLE_SEARCH_SCHEMA,
88.        )
```

### Tool Registration Overview (Lines 80-182)

The `_build_builtin_tools()` method creates a list of `AgentTool` instances. All builtin tools follow the same pattern:
- Web tools: `google_search`, `web_fetch`
- Important dates: 6 tool instances
- User tools: `get_user_accounts`, `get_user_profile`
- Location tools: `get_user_location`, `set_location_reminder`
- Weather tool (reused from trip_planner)
- Action history tool: `recall_recent_actions`

### `AgentTool` Data Structure (Lines 67-98 in models.py)

```python
@dataclass
class AgentTool:
    """A tool available inside a StandardAgent's mini ReAct loop.

    Attributes:
        name: Tool function name (used in LLM tool_calls).
        description: What this tool does (shown to the LLM).
        parameters: JSON Schema for tool arguments.
        executor: Async function(args: dict, context: AgentToolContext) -> str.
        needs_approval: If True, pause execution for user confirmation before running.
        risk_level: One of "read", "write", "destructive".
        get_preview: Async function to generate human-readable preview for approval.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    executor: Callable
    needs_approval: bool = False
    risk_level: str = "read"  # "read", "write", "destructive"
    category: str = "utility"
    get_preview: Optional[Callable] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

### Tool Execution (Lines 226-267 in tool_manager.py)

```python
226.    async def _execute_single(
227.        self,
228.        tool_call: Any,
229.        tenant_id: str,
230.        metadata: Optional[Dict[str, Any]] = None,
231.        request_tools: Optional[List] = None,
232.        request_context: Optional[Dict[str, Any]] = None,
233.    ) -> Any:
234.        """Dispatch to agent-tool or regular tool execution."""
235.        tool_name = tool_call.name
236.        try:
237.            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else json.loads(tool_call.arguments)
238.        except (json.JSONDecodeError, TypeError) as e:
239.            return (
240.                f"Error: Failed to parse arguments for tool '{tool_name}': {e}. "
241.                "Please retry with valid JSON arguments."
242.            )
243.
244.        if self._is_agent_tool(tool_name):
245.            # Agent-Tool execution
246.            task_instruction = args.pop("task_instruction", "")
247.            return await execute_agent_tool(...)
248.        else:
249.            # Builtin tool execution — use request_tools (local copy) if available
250.            tools = request_tools if request_tools is not None else getattr(self, 'builtin_tools', [])
251.            tool = next((t for t in tools if t.name == tool_name), None)
252.            if not tool:
253.                return f"Error: Tool '{tool_name}' not found"
254.
255.            context = AgentToolContext(
256.                tenant_id=tenant_id,
257.                credentials=self.credential_store,
258.                metadata=self._build_tool_metadata(metadata),
259.            )
260.            return await tool.executor(args, context)
261.        ```

**Key execution flow**:
- Line 260: **Executor is called with `(args: dict, context: AgentToolContext)`**
- Expected return type: Currently `str`, but can also return `AgentToolResult`
- Executor function signature: `async def executor(args: dict, context: AgentToolContext) -> str | AgentToolResult`

### Tool Executor Signature

From `/onevalet/builtin_agents/tools/google_search.py`:

```python
async def google_search_executor(args: dict, context: AgentToolContext = None) -> str:
    """Search the web using Google Custom Search API."""
    query = args.get("query", "")
    num_results = args.get("num_results", 5)
    # ... returns str
```

---

## 3. LLM Client Media Handling

### Location
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/llm/base.py`
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/llm/litellm_client.py`

### `_add_media_to_messages_openai` Method (Lines 246-294 in base.py)

```python
246.    def _add_media_to_messages_openai(
247.        self,
248.        messages: List[Dict[str, Any]],
249.        media: List[Dict[str, Any]]
250.    ) -> List[Dict[str, Any]]:
251.        """
252.        Add media (images) to the last user message in OpenAI vision format.
253.
254.        Used by: OpenAI, Azure OpenAI, and other OpenAI-compatible providers.
255.
256.        Args:
257.            messages: List of message dicts
258.            media: List of media dicts with 'type', 'data', and 'media_type'
259.
260.        Returns:
261.            Updated messages list with images embedded
262.        """
263.        if not media:
264.            return messages
265.
266.        messages = [msg.copy() for msg in messages]
267.        for i in range(len(messages) - 1, -1, -1):
268.            if messages[i].get("role") == "user":
269.                text_content = messages[i].get("content", "")
270.                content_parts = []
271.
272.                if text_content:
273.                    content_parts.append({"type": "text", "text": text_content})
274.
275.                for item in media:
276.                    if item.get("type") == "image":
277.                        data = item.get("data", "")
277.                        media_type = item.get("media_type", "image/jpeg")
278.
279.                        if data.startswith(("http://", "https://")):
280.                            content_parts.append({
281.                                "type": "image_url",
282.                                "image_url": {"url": data}
283.                            })
284.                        else:
285.                            content_parts.append({
286.                                "type": "image_url",
287.                                "image_url": {"url": f"data:{media_type};base64,{data}"}
288.                            })
289.
290.                messages[i]["content"] = content_parts
291.                break
292.
293.        return messages
```

**Media format**:
```python
[
    {
        "type": "image",  # Media type
        "data": "base64_string_or_url",  # Base64-encoded data or URL
        "media_type": "image/jpeg"  # MIME type
    }
]
```

**Output format** (OpenAI vision):
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Your message"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
}
```

### `_call_api` Integration (Lines 138-231 in litellm_client.py)

```python
138.    async def _call_api(
139.        self,
140.        messages: List[Dict[str, Any]],
141.        tools: Optional[List[Dict[str, Any]]] = None,
142.        **kwargs,
143.    ) -> LLMResponse:
144.        """Make a non-streaming call via litellm.acompletion."""
145.        import litellm
146.
147.        # Handle media (images) - use base class helper
148.        media = kwargs.pop("media", None)
149.        if media and messages:
150.            messages = self._add_media_to_messages_openai(messages, media)
151.
152.        model = kwargs.get("model") or self._litellm_model
153.        params: Dict[str, Any] = {
154.            "model": model,
155.            "messages": messages,
156.            **self._model_params(self.config.model, **kwargs),
157.            **self._base_kwargs,
158.        }
159.
160.        if tools:
161.            params["tools"] = tools
162.            params["tool_choice"] = kwargs.get("tool_choice", "auto")
163.
164.        # ... rest of API call
181.        response = await litellm.acompletion(**params)
```

**Media flow**:
1. Media passed in `kwargs["media"]`
2. Line 148-150: Extracted and added to messages via `_add_media_to_messages_openai()`
3. Messages with embedded images passed to `litellm.acompletion()`

---

## 4. Models and Data Structures

### Location
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/models.py`
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/orchestrator/agent_tool.py`

### `AgentToolContext` (Lines 49-63 in models.py)

```python
@dataclass
class AgentToolContext:
    """Context passed to tool executors.

    Provides access to shared resources that tool functions need.
    Used by both agent-level tools (tools) and orchestrator-level
    builtin tools.
    """

    llm_client: Optional["BaseLLMClient"] = None
    tenant_id: str = ""
    user_profile: Optional[Dict[str, Any]] = None
    context_hints: Optional[Dict[str, Any]] = None
    credentials: Optional["CredentialStore"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `AgentToolResult` (Lines 31-39 in agent_tool.py)

```python
@dataclass
class AgentToolResult:
    """Result from executing an agent as a tool in the ReAct loop."""

    completed: bool
    result_text: str = ""
    agent: Optional[Any] = None
    approval_request: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Used for**:
- Agent-tool execution results (when agents are called as tools)
- Can return structured metadata alongside result text
- `metadata` dict can carry `tool_trace`, agent status, and other info
- Example usage in `react_loop.py` line 472-475:
  ```python
  if isinstance(result, AgentToolResult):
      result_text = result.result_text
      r_meta = result.metadata if isinstance(result.metadata, dict) else {}
      tool_trace = r_meta.get("tool_trace") or []
  ```

### Tool Executor Return Type

**Current supported returns**:
1. **String**: `async def executor(...) -> str` — converted to simple text tool result
2. **AgentToolResult**: `async def executor(...) -> AgentToolResult` — structured result with metadata

**From tool_manager.py lines 260** and **react_loop.py lines 472-478**:
```python
# Executor call (line 260 in tool_manager.py)
return await tool.executor(args, context)

# Result handling (lines 472-478 in react_loop.py)
if isinstance(result, AgentToolResult):
    result_text = result.result_text
    r_meta = result.metadata if isinstance(result.metadata, dict) else {}
    tool_trace = r_meta.get("tool_trace") or []
else:
    result_text = str(result) if result is not None else ""
    tool_trace = []
```

---

## 5. How User-Facing Responses Are Sent Back

### Location
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/orchestrator/react_loop.py`
- **File**: `/Users/vince/workspace/withkoi/OneValet/onevalet/streaming/models.py`

### Response Flow Through AgentEvents

**Main method**: `_react_loop_events()` (Line 51-627 in react_loop.py)

The ReAct loop yields `AgentEvent` objects at key points:

#### 1. **Final Response (Completed)**

**Lines 527-529**:
```python
final_response = complete_task_result.result
self._audit.log_react_turn(...)
yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
yield AgentEvent(type=EventType.MESSAGE_END, data={})
```

#### 2. **Final Execution End Event (Lines 611-626)**

```python
611.        yield AgentEvent(
612.            type=EventType.EXECUTION_END,
613.            data={
614.                "duration_ms": duration_ms,
615.                "turns": turn,
616.                "tool_calls_count": len(all_tool_records),
617.                "final_response": final_response,
618.                "result_status": result_status,
619.                "pending_approvals": pending_approvals,
620.                "token_usage": {
621.                    "input_tokens": total_usage.input_tokens,
622.                    "output_tokens": total_usage.output_tokens,
623.                },
624.                "tool_calls": [dataclasses.asdict(r) for r in all_tool_records],
625.            },
626.        )
```

**This is the final event carrying**:
- `final_response`: The complete response text to the user
- `result_status`: COMPLETED, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL, or ERROR
- `pending_approvals`: Any pending user confirmations
- `token_usage`: Token consumption tracking
- `tool_calls`: All tool execution records

### AgentEvent Structure (Lines 74-100 in streaming/models.py)

```python
@dataclass
class AgentEvent:
    """
    Base event structure for streaming.

    All events have:
    - type: The type of event
    - data: Event-specific data
    - timestamp: When the event occurred
    - agent_id: Which agent generated the event
    - sequence: Sequence number for ordering
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "sequence": self.sequence,
        }
```

### EventType Enum (Lines 31-71 in streaming/models.py)

```python
class EventType(str, Enum):
    """Types of events that can be streamed"""
    # State events
    STATE_CHANGE = "state_change"
    FIELD_COLLECTED = "field_collected"
    FIELD_VALIDATED = "field_validated"

    # Message events
    MESSAGE_START = "message_start"
    MESSAGE_CHUNK = "message_chunk"
    MESSAGE_END = "message_end"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"

    # Progress events
    PROGRESS_UPDATE = "progress_update"

    # Execution events
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"

    # Error events
    ERROR = "error"
    WARNING = "warning"
    
    # ... more events
```

### Response Message Events (Lines 291-293, 344-345, 528-530, 577-579, 605-607)

All completed responses follow the pattern:
```python
yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
yield AgentEvent(type=EventType.MESSAGE_END, data={})
```

**Note**: Currently `MESSAGE_CHUNK` carries `chunk` as **text only** - no multimodal support for response media.

---

## Key Findings for Multimodal Tool Results

### Current Limitations

1. **Tool Result Message Format**: Only accepts `content: str` (line 706 in react_loop.py)
   - `_build_tool_result_message()` creates `{"role": "tool", "tool_call_id": id, "content": content_str}`
   - Cannot embed images or other media in tool results sent back to LLM

2. **Tool Result Truncation**: Only handles text (lines 482-483)
   - `_cap_tool_result()`: String-only truncation
   - `truncate_tool_result()`: String-only truncation
   - No awareness of media types or multi-part content

3. **User Response Events**: Only carry text (line 292, 528, 540)
   - `MESSAGE_CHUNK` events contain `{"chunk": text}`
   - No multimodal response data structure

4. **Tool Executor Return Type**: Currently limited
   - Builtin tools: Expected to return `str`
   - Agent tools: Can return `AgentToolResult` with metadata
   - **No structured support for media in tool results**

### Extension Points for Multimodal Support

1. **Tool Executor Signature Enhancement**:
   - Could return `AgentToolResult` with enriched `metadata` containing media references
   - Metadata could include: `{"images": [...], "media": [...]}`

2. **Tool Result Message Modification**:
   - Extend `_build_tool_result_message()` to handle content as list (like OpenAI vision format)
   - Accept `content: Union[str, List[Dict[str, Any]]]`

3. **Message Content Structure**:
   - Current: `{"role": "tool", "tool_call_id": id, "content": "text"}`
   - Future: `{"role": "tool", "tool_call_id": id, "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}`

4. **Response Event Enhancement**:
   - Extend `MESSAGE_CHUNK` to carry multimodal data
   - New event type like `MESSAGE_MEDIA` for non-text content

5. **Context Manager Awareness**:
   - Modify truncation logic to handle media and compressed content
   - Add token estimation for different media types (already has `IMAGE_TOKEN_ESTIMATE`)

---

## Summary of File Structure

```
/Users/vince/workspace/withkoi/OneValet/
├── onevalet/
│   ├── models.py                           # AgentTool, AgentToolContext, RequiredField
│   ├── orchestrator/
│   │   ├── orchestrator.py                 # Main Orchestrator class
│   │   ├── react_loop.py                   # ReactLoopMixin, main ReAct loop (753 lines)
│   │   ├── tool_manager.py                 # ToolManagerMixin, tool execution
│   │   ├── agent_tool.py                   # AgentToolResult, execute_agent_tool()
│   │   ├── context_manager.py              # ContextManager, truncation logic
│   │   ├── react_config.py                 # ReactLoopConfig, COMPLETE_TASK_SCHEMA
│   │   └── constants.py                    # TOOL_RESULT_HARD_CAP_CHARS, etc.
│   ├── llm/
│   │   ├── base.py                         # BaseLLMClient, _add_media_to_messages_openai()
│   │   └── litellm_client.py               # LiteLLMClient implementation
│   ├── streaming/
│   │   └── models.py                       # AgentEvent, EventType
│   └── builtin_agents/tools/
│       ├── google_search.py                # Example tool executor
│       └── ...
```

