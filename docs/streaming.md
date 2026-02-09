# Streaming

FlowAgents supports real-time streaming for better user experience.

## Quick Start

```python
from flowagents import StreamEngine, StreamMode, EventType

engine = StreamEngine(mode=StreamMode.INCREMENTAL)

async for event in engine.stream(agent, message):
    if event.type == EventType.MESSAGE_CHUNK:
        print(event.data["chunk"], end="", flush=True)
    elif event.type == EventType.STATE_CHANGE:
        print(f"\n[State: {event.data['new_status']}]")
```

## Stream Modes

| Mode | Description |
|------|-------------|
| `StreamMode.VALUES` | Complete state after each update |
| `StreamMode.UPDATES` | Only incremental changes |
| `StreamMode.MESSAGES` | LLM messages (token-by-token) |
| `StreamMode.EVENTS` | All events (state changes, tool calls) |

## Event Types

| Event Type | Description |
|------------|-------------|
| `MESSAGE_CHUNK` | Partial message content |
| `MESSAGE_COMPLETE` | Full message finished |
| `STATE_CHANGE` | Agent state transition |
| `TOOL_CALL_START` | Tool execution started |
| `TOOL_CALL_END` | Tool execution completed |
| `ERROR` | An error occurred |
| `DONE` | Stream finished |

## Handling Events

```python
from flowagents import EventType

async for event in engine.stream(agent, message):
    match event.type:
        case EventType.MESSAGE_CHUNK:
            ui.append_text(event.data["chunk"])

        case EventType.STATE_CHANGE:
            ui.update_status(f"Status: {event.data['new_status']}")

        case EventType.TOOL_CALL_START:
            ui.show_spinner(f"Running {event.data['tool_name']}...")

        case EventType.TOOL_CALL_END:
            ui.hide_spinner()

        case EventType.ERROR:
            ui.show_error(event.data["error"])

        case EventType.DONE:
            ui.complete()
```

## Streaming with Orchestrator

```python
orchestrator = Orchestrator(llm_client=client)

async for event in orchestrator.process_stream(
    message="Book a flight to Paris",
    tenant_id="user_123"
):
    handle_event(event)
```

## Streaming with LLM Clients

All built-in LLM clients support streaming:

```python
from flowagents import OpenAIClient

client = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")

async for chunk in client.stream_completion(messages):
    print(chunk.content, end="", flush=True)
```

## WebSocket Integration

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    while True:
        message = await websocket.receive_text()

        async for event in orchestrator.process_stream(message):
            await websocket.send_json({
                "type": event.type.value,
                "data": event.data,
            })
```

## Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat/stream")
async def chat_stream(message: str):
    async def event_generator():
        async for event in orchestrator.process_stream(message):
            yield f"data: {json.dumps({'type': event.type.value, 'data': event.data})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

## Best Practices

1. **Use incremental mode** - Better UX for long responses
2. **Handle all event types** - Don't ignore errors
3. **Show progress** - Use state changes to show what's happening
4. **Graceful degradation** - Fall back to non-streaming if needed
