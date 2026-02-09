# OneValet

A zero-code AI personal assistant with built-in agents, ReAct orchestration, and multi-tenant support.

## Quick Start

```bash
git clone https://github.com/xiaoyu-work/onevalet.git
cd onevalet
uv sync --extra openai        # or: --extra anthropic, --all-extras
cp .env.example .env          # configure API keys
cp config.yaml.example config.yaml
```

```bash
# Start
python -m onevalet

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?"}'

# Stream (SSE)
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?"}'

# Multi-tenant
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "user_123", "message": "What is on my calendar today?"}'
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, get response |
| `/stream` | POST | Send message, stream response (SSE) |
| `/health` | GET | Health check |

**Request body:**

```json
{
  "message": "...",
  "tenant_id": "...",   // optional, default "default"
  "metadata": {}        // optional
}
```

## Config

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | `openai` / `anthropic` / `azure` / `dashscope` / `gemini` / `ollama` |
| `model` | Yes | Model name, e.g. `gpt-4o`, `claude-sonnet-4-5-20250929` |
| `database` | Yes | PostgreSQL DSN, supports `${ENV_VAR}` |
| `api_key` | No | If omitted, reads from provider's default env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) |
| `base_url` | No | Custom endpoint (required for Azure OpenAI) |
| `system_prompt` | No | System prompt / personality |

See [docs/configuration.md](docs/configuration.md) for full configuration reference.

## Custom Agents

```python
from onevalet import valet, StandardAgent, InputField, AgentStatus

@valet(triggers=["send email"])
class SendEmailAgent(StandardAgent):
    """Send emails to users"""

    recipient = InputField("Who should I send to?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Email sent to {self.recipient}!",
        )
```

## License

[MIT](LICENSE)
