# Multimodal Tool Results Extension Plan

## Architecture Overview

### Current Tool Result Flow

```
Tool Executor (returns str or AgentToolResult)
    ↓
_execute_single() (line 260 in tool_manager.py)
    ↓
React Loop collects result
    ↓
isinstance(result, AgentToolResult)? → Extract result_text, metadata
    ↓
_cap_tool_result() [400KB hard cap]
    ↓
_context_manager.truncate_tool_result() [context-aware cap]
    ↓
_build_tool_result_message(tool_call_id, result_text)
    ↓
Message: {"role": "tool", "tool_call_id": id, "content": text_str}
    ↓
Appended to messages list for next LLM call
    ↓
LLM processes tool result
```

### Proposed Multimodal Flow

```
Tool Executor (returns str or AgentToolResult with media)
    ↓
_execute_single() [unchanged]
    ↓
React Loop collects result
    ↓
isinstance(result, AgentToolResult)? → Extract result_text, metadata, media
    ↓
_cap_tool_result() [handle text portion]
    ↓
_context_manager.truncate_tool_result() [handle text + media budget]
    ↓
_build_tool_result_message(tool_call_id, result_text, media) ← NEW
    ↓
Message: {"role": "tool", "tool_call_id": id, "content": [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "..."}},
    ...
]}
    ↓
Appended to messages list for next LLM call
    ↓
LLM processes multimodal tool result
```

---

## 1. Enhanced AgentToolResult

**File**: `onevalet/orchestrator/agent_tool.py`

Current:
```python
@dataclass
class AgentToolResult:
    completed: bool
    result_text: str = ""
    agent: Optional[Any] = None
    approval_request: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Proposed:
```python
@dataclass
class AgentToolResult:
    completed: bool
    result_text: str = ""
    agent: Optional[Any] = None
    approval_request: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    media: Optional[List[Dict[str, Any]]] = None  # NEW: [{type, data, media_type}, ...]
    media_references: Dict[str, Any] = field(default_factory=dict)  # NEW: urls, paths, etc
```

---

## 2. Enhanced Tool Executor Capability

**File**: `onevalet/models.py`

Add new return type option:

```python
# Current - tool can return:
# 1. str → converted to plain text result
# 2. AgentToolResult → structured result

# New unified signature:
async def tool_executor(
    args: dict, 
    context: AgentToolContext
) -> Union[str, AgentToolResult]

# Examples:
# Example 1: Simple text (unchanged)
async def search_executor(args, context) -> str:
    return "Search results..."

# Example 2: Structured with images (NEW)
async def image_search_executor(args, context) -> AgentToolResult:
    images = fetch_images(args['query'])
    return AgentToolResult(
        completed=True,
        result_text="Found 3 images of cats",
        media=[
            {
                "type": "image",
                "data": "base64_encoded_image_1",
                "media_type": "image/jpeg",
            },
            {
                "type": "image", 
                "data": "https://url.to/image2.jpg",
                "media_type": "image/jpeg",
            },
            # ... more images
        ]
    )

# Example 3: Video results (NEW)
async def video_search_executor(args, context) -> AgentToolResult:
    return AgentToolResult(
        completed=True,
        result_text="Video 1: Cat videos compilation",
        media=[
            {
                "type": "video",
                "data": "base64_encoded_or_url",
                "media_type": "video/mp4",
                "thumbnail": "base64_thumbnail",
            }
        ]
    )
```

---

## 3. Enhanced Truncation Logic

**File**: `onevalet/orchestrator/context_manager.py`

Current `truncate_tool_result()` (line 92):
- Only handles text string truncation

Proposed `truncate_tool_result_multimodal()`:
```python
def truncate_tool_result_multimodal(
    self, 
    result: str, 
    media: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Truncate tool result and media to fit budget.
    
    Returns:
        (truncated_text, truncated_media)
    """
    max_text_chars = int(
        min(
            self.config.context_token_limit * self.config.max_tool_result_share * 4,
            self.config.max_tool_result_chars,
        )
    )
    
    # Budget allocation: 80% text, 20% media
    text_budget = int(max_text_chars * 0.8)
    media_budget = int(max_text_chars * 0.2)
    
    # Truncate text
    truncated_text = result[:text_budget] if len(result) > text_budget else result
    
    # Truncate media (keep only highest priority, stay within budget)
    truncated_media = None
    if media:
        truncated_media = []
        current_size = 0
        for item in media:
            item_size = len(item.get('data', '')) // 4  # Rough estimate
            if current_size + item_size <= media_budget:
                truncated_media.append(item)
                current_size += item_size
            else:
                break
    
    return truncated_text, truncated_media
```

New token estimation for media (already exists, line 50-56):
```python
if isinstance(content, list):
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type", "")
            if part_type in ("image_url", "image"):
                total += IMAGE_TOKEN_ESTIMATE  # 170 tokens per image
                continue
```

---

## 4. Enhanced Tool Result Message Building

**File**: `onevalet/orchestrator/react_loop.py`

Current `_build_tool_result_message()` (line 703):
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

Proposed multimodal version:
```python
def _build_tool_result_message(
    self,
    tool_call_id: str,
    content: Union[str, List[Dict[str, Any]]],
    is_error: bool = False,
    media: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a tool result message for the LLM messages list.
    
    Supports both text-only and multimodal content.
    """
    if isinstance(content, str):
        # Text-only path (unchanged)
        if is_error:
            content = f"[ERROR] {content}"
        final_content = content
        
        # But check if media should be added
        if media:
            final_content = [{"type": "text", "text": content}]
            for item in media:
                if item.get("type") == "image":
                    data = item.get("data", "")
                    media_type = item.get("media_type", "image/jpeg")
                    if data.startswith(("http://", "https://")):
                        final_content.append({
                            "type": "image_url",
                            "image_url": {"url": data}
                        })
                    else:
                        final_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })
    else:
        # Already structured content (list of parts)
        final_content = content
    
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": final_content,
    }
```

---

## 5. Updated React Loop Tool Result Handling

**File**: `onevalet/orchestrator/react_loop.py` (lines 472-489)

Current:
```python
if isinstance(result, AgentToolResult):
    result_text = result.result_text
    r_meta = result.metadata if isinstance(result.metadata, dict) else {}
    tool_trace = r_meta.get("tool_trace") or []
else:
    result_text = str(result) if result is not None else ""
    tool_trace = []
    
result_chars_original = len(result_text)
original_len = len(result_text)
# Hard cap on tool result size
result_text = self._cap_tool_result(result_text)
result_text = self._context_manager.truncate_tool_result(result_text)
...
messages.append(self._build_tool_result_message(tc.id, result_text))
```

Proposed (with media support):
```python
media = None
if isinstance(result, AgentToolResult):
    result_text = result.result_text
    media = result.media  # NEW
    r_meta = result.metadata if isinstance(result.metadata, dict) else {}
    tool_trace = r_meta.get("tool_trace") or []
else:
    result_text = str(result) if result is not None else ""
    media = None
    tool_trace = []
    
result_chars_original = len(result_text)
original_len = len(result_text)
# Hard cap on tool result size (handles text)
result_text = self._cap_tool_result(result_text)

# NEW: Multimodal-aware truncation
if media:
    result_text, media = self._context_manager.truncate_tool_result_multimodal(
        result_text, media
    )
else:
    result_text = self._context_manager.truncate_tool_result(result_text)

...
# NEW: Pass media to message builder
messages.append(
    self._build_tool_result_message(tc.id, result_text, media=media)
)
```

---

## 6. User Response Multimodal Support

**File**: `onevalet/streaming/models.py`

Current MESSAGE_CHUNK:
```python
yield AgentEvent(
    type=EventType.MESSAGE_CHUNK,
    data={"chunk": text_string}
)
```

Proposed new event type for multimodal content:
```python
class EventType(str, Enum):
    # ... existing events ...
    MESSAGE_MEDIA = "message_media"  # NEW
    RESPONSE_MEDIA = "response_media"  # NEW

# Usage:
yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": text_part})
yield AgentEvent(type=EventType.MESSAGE_MEDIA, data={
    "type": "image",
    "media_type": "image/jpeg",
    "data": "base64_or_url",
    "alt_text": "Description for accessibility"
})
```

Or enhanced MESSAGE_CHUNK:
```python
yield AgentEvent(
    type=EventType.MESSAGE_CHUNK,
    data={
        "chunk": text_content,
        "media": [  # NEW: optional media in same event
            {"type": "image", "data": "..."}
        ]
    }
)
```

---

## 7. Implementation Checklist

- [ ] Extend `AgentToolResult` with `media` field
- [ ] Enhance tool executor documentation to support media returns
- [ ] Add `truncate_tool_result_multimodal()` to `ContextManager`
- [ ] Update `_build_tool_result_message()` to handle multimodal content
- [ ] Update react loop tool result handling (lines 472-489)
- [ ] Add new `EventType` enum values for media
- [ ] Update MESSAGE_CHUNK/EXECUTION_END event data structures
- [ ] Add integration test for image in tool result
- [ ] Add integration test for media truncation
- [ ] Update tool executor examples in docstrings
- [ ] Document media format in README/docs

---

## Media Format Specification

```python
# Standard media item format (used in AgentToolResult.media)
{
    "type": "image" | "video" | "audio" | "document",
    "data": "base64_string | url_string",
    "media_type": "image/jpeg | image/png | video/mp4 | ...",
    "size_bytes": int,  # optional
    "thumbnail": "base64_or_url",  # optional, for video/document
    "metadata": {  # optional
        "width": int,
        "height": int,
        "duration_seconds": float,  # for video/audio
        "title": str,
        "description": str,
    }
}
```

---

## Backward Compatibility

All changes are **backward compatible**:

1. Tool executors returning `str` work unchanged
2. Tool executors returning `AgentToolResult` without `media` work unchanged
3. LLM will handle both text-only and multimodal messages
4. Event consumers can ignore new media events

