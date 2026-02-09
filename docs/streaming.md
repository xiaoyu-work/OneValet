# Streaming

OneValet streams responses in real time via Server-Sent Events (SSE).

## Endpoint

```
POST /stream
```

### Request Body

```json
{
  "message": "Book a flight to Paris",
  "tenant_id": "user_123",
  "metadata": {}
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `message` | Yes | The user message to process |
| `tenant_id` | Yes | Identifies the tenant / conversation |
| `metadata` | No | Arbitrary key-value pairs passed to the agent |

### Response

The response uses the `text/event-stream` content type. Each event is a line prefixed with `data: ` followed by a JSON object and a blank line:

```
data: {"type": "message_chunk", "data": "Hello"}\n\n
data: {"type": "message_chunk", "data": ", how"}\n\n
data: {"type": "message_end", "data": "Hello, how can I help?"}\n\n
data: [DONE]\n\n
```

The final event is always `data: [DONE]\n\n`, signaling the stream is finished.

## Event Types

| Type | Description |
|------|-------------|
| `message_start` | Message stream started |
| `message_chunk` | Partial message content (token-by-token) |
| `message_end` | Message stream finished |
| `state_change` | Agent state transition |
| `field_collected` | Input field collected from user |
| `field_validated` | Input field validated |
| `tool_call_start` | Tool execution started |
| `tool_call_end` | Tool execution completed |
| `tool_result` | Tool returned a result |

## Examples

### curl

```bash
curl -N -X POST https://your-host/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Book a flight to Paris",
    "tenant_id": "user_123",
    "metadata": {}
  }'
```

The `-N` flag disables output buffering so events appear as they arrive.

### JavaScript (EventSource / fetch)

The `EventSource` API only supports GET requests, so use `fetch` with a readable stream instead:

```javascript
const response = await fetch("https://your-host/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Book a flight to Paris",
    tenant_id: "user_123",
    metadata: {}
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value, { stream: true });

  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const payload = line.slice(6);

    if (payload === "[DONE]") {
      console.log("Stream finished");
      break;
    }

    const event = JSON.parse(payload);
    if (event.type === "message_chunk") {
      process.stdout.write(event.data);
    }
  }
}
```

## Best Practices

1. **Always handle the `[DONE]` event** -- use it to finalize your UI or close the connection.
2. **Handle errors gracefully** -- check for `error` events and display them to the user.
3. **Show progress** -- use `state_change` and `tool_call_start` / `tool_call_end` events to indicate what the agent is doing.
