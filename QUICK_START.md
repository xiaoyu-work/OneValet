# OneValet Multimodal Tool Results - Quick Start Guide

## TL;DR - Start Here

You have 3 comprehensive documents totaling 2000+ lines:

| Document | Size | Purpose |
|----------|------|---------|
| **DEEP_DIVE_ANALYSIS.md** | 24 KB | Complete code architecture breakdown |
| **CODE_REFERENCE.md** | 14 KB | Exact code snippets with line numbers |
| **MULTIMODAL_EXTENSION_PLAN.md** | 11 KB | Implementation strategy & roadmap |
| **ANALYSIS_INDEX.md** | 8 KB | Navigation guide (you're reading the summary) |

---

## 📍 Where Are Things?

### ReAct Loop Entry Point
```python
# File: onevalet/orchestrator/react_loop.py (753 lines)
# Method: _react_loop_events() - Line 51
# Tool result processing: Lines 472-489
# Build result message: Lines 703-716
```

### Tool Registration & Execution
```python
# File: onevalet/orchestrator/tool_manager.py
# Build builtin tools: Lines 80-182
# Google search registration: Lines 107-114
# Execute single tool: Lines 226-267 (line 260 = executor call)
```

### Tool Result Truncation
```python
# Hard cap (400KB):
# File: onevalet/orchestrator/tool_manager.py:275-286

# Context-aware truncation:
# File: onevalet/orchestrator/context_manager.py:92-115
```

### LLM Media Handling
```python
# Add media to messages:
# File: onevalet/llm/base.py:246-294

# LLM API call with media:
# File: onevalet/llm/litellm_client.py:138-181 (lines 148-150)
```

---

## 🔴 Current Limitation

Tool results can only be **text**, not multimodal:

```python
# Current (react_loop.py:703-716)
def _build_tool_result_message(self, tool_call_id, content: str, is_error=False):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content  # ← STRING ONLY ❌
    }

# Needed for multimodal
{
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": [  # ← LIST (like OpenAI Vision) ✅
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "..."}},
    ]
}
```

---

## ✅ Solution Overview

### 1. Extend Tool Result Return Type
```python
# agents/orchestrator/agent_tool.py - Add to AgentToolResult:
@dataclass
class AgentToolResult:
    completed: bool
    result_text: str
    media: Optional[List[Dict[str, Any]]] = None  # ← NEW
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. Update Tool Result Message Building
```python
# orchestrator/react_loop.py:703 - Support multimodal content:
def _build_tool_result_message(
    self, 
    tool_call_id: str,
    content: Union[str, List[Dict[str, Any]]],  # ← NEW: list support
    media: Optional[List[Dict[str, Any]]] = None  # ← NEW: media param
):
    # Build multimodal content if media provided
    if media:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": [
                {"type": "text", "text": content_str},
                *[format_media_item(item) for item in media]
            ]
        }
    # ... existing text-only path
```

### 3. Update React Loop to Collect Media
```python
# orchestrator/react_loop.py:472-489
if isinstance(result, AgentToolResult):
    result_text = result.result_text
    media = result.media  # ← NEW: extract media
    # ...
    messages.append(
        self._build_tool_result_message(tc.id, result_text, media=media)  # ← NEW
    )
```

### 4. Update Truncation for Media
```python
# orchestrator/context_manager.py - Add method:
def truncate_tool_result_multimodal(self, text: str, media: List):
    # Budget allocation: 80% text, 20% media
    # Truncate both components proportionally
    # Return (truncated_text, truncated_media)
```

---

## 📝 Example Tool Returning Images

```python
# builtin_agents/tools/image_search.py
from onevalet.orchestrator.agent_tool import AgentToolResult

async def image_search_executor(args: dict, context: AgentToolContext) -> AgentToolResult:
    """Search for images and return them as multimodal result."""
    query = args.get("query", "")
    
    # Fetch images...
    images = fetch_from_google_images(query)
    
    # Return multimodal result
    return AgentToolResult(
        completed=True,
        result_text=f"Found {len(images)} images for '{query}'",
        media=[
            {
                "type": "image",
                "data": base64_image,
                "media_type": "image/jpeg"
            }
            for base64_image in images
        ]
    )
```

---

## 🚀 Implementation Steps

### Phase 1: Core Support (1-2 days)
- [ ] Add `media` field to `AgentToolResult` 
- [ ] Update `_build_tool_result_message()` to handle media
- [ ] Update react loop to extract and pass media
- [ ] Update token estimation to include media

### Phase 2: Content Management (1-2 days)
- [ ] Add `truncate_tool_result_multimodal()` method
- [ ] Implement budget allocation (80% text, 20% media)
- [ ] Update tool result processing to use multimodal truncation

### Phase 3: User Responses (1 day)
- [ ] Add `MESSAGE_MEDIA` event type
- [ ] Update response event generation
- [ ] Test streaming with media

### Phase 4: Testing & Examples (1-2 days)
- [ ] Write integration tests
- [ ] Create example tools (image search, web screenshot)
- [ ] Document media format in README

---

## 🔍 Critical Locations to Modify

```
1. onevalet/orchestrator/agent_tool.py        Line 32-39   (AgentToolResult)
2. onevalet/orchestrator/react_loop.py        Line 472-489 (Extract media)
3. onevalet/orchestrator/react_loop.py        Line 703-716 (Build message)
4. onevalet/orchestrator/context_manager.py   Line 92-115  (New truncation method)
5. onevalet/streaming/models.py               Line 31-71   (Add EVENT_TYPE)
```

---

## 📚 Deep Dive by Topic

### "I need to understand tool result flow"
→ Read: **DEEP_DIVE_ANALYSIS.md § 1** (ReAct loop tool result handling)

### "I need exact code locations"  
→ Read: **CODE_REFERENCE.md § 1, 2, 5, 6** (Processing, execution, messages, truncation)

### "I need implementation strategy"
→ Read: **MULTIMODAL_EXTENSION_PLAN.md § 2-7** (Data structures, logic updates, checklist)

### "I need to understand media in LLM"
→ Read: **DEEP_DIVE_ANALYSIS.md § 3** + **CODE_REFERENCE.md § 4**

### "I need to implement this now"
→ Follow: **MULTIMODAL_EXTENSION_PLAN.md § 7** (Implementation checklist)

---

## ⚡ Key Numbers to Remember

| Constant | Value | Location |
|----------|-------|----------|
| Hard cap on tool result | 400,000 chars | constants.py:37 |
| Agent tool result truncation | 2000 chars threshold | constants.py:42 |
| Image token estimate | 170 tokens | constants.py:29 |
| Message overhead | 4 tokens | constants.py:10 |
| Tool call overhead | 20 tokens | constants.py:13 |

---

## 🔗 Architecture Map

```
User Message
    ↓
[ReAct Loop Iteration]
    ↓
LLM generates Tool Calls
    ↓
_execute_single() dispatches tools
    ↓
Tool Executor runs (returns str or AgentToolResult)
    ↓
If AgentToolResult:
  - Extract result_text ✓
  - Extract metadata ✓
  - Extract media ← NEW
    ↓
Truncation:
  1. _cap_tool_result() - hard cap text
  2. truncate_tool_result() - context-aware
  3. truncate_tool_result_multimodal() ← NEW (allocate budget)
    ↓
_build_tool_result_message() builds:
  - role: "tool"
  - tool_call_id: string
  - content: string | [text, image, ...] ← NEW format
    ↓
Append to messages list
    ↓
Next LLM call gets multimodal tool result ✓
    ↓
LLM can see images/media in tool result ✓
```

---

## ✨ Backward Compatibility

All changes are **100% backward compatible**:

```python
# Old tool (still works)
async def old_tool(args, context) -> str:
    return "Just text"

# New tool (with images)
async def new_tool(args, context) -> AgentToolResult:
    return AgentToolResult(
        completed=True,
        result_text="Text with images",
        media=[...images...]
    )

# Both work in same system ✓
```

---

## 📞 Questions?

- **"Why 80/20 budget split?"** → See MULTIMODAL_EXTENSION_PLAN.md § 4
- **"How does token estimation work?"** → See CODE_REFERENCE.md § 8
- **"What media types are supported?"** → See MULTIMODAL_EXTENSION_PLAN.md § 8
- **"How do I test this?"** → See MULTIMODAL_EXTENSION_PLAN.md § 7

---

**Start by reading: ANALYSIS_INDEX.md (quick navigation) or DEEP_DIVE_ANALYSIS.md (full context)**
