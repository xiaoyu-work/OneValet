# OneValet

A self-hosted AI personal assistant. Manage emails, calendar, and more through a chat interface.

## Quick Start

### 1. Install

```bash
git clone https://github.com/xiaoyu-work/onevalet.git
cd onevalet
uv sync --extra openai        # or: --extra anthropic, --all-extras
```

### 2. Start

```bash
python -m onevalet --ui
```

Open **http://localhost:8000** in your browser.

### 3. Configure

Go to **http://localhost:8000/settings** and set up:

1. **LLM Provider** - Choose your AI provider (OpenAI, Azure, Anthropic, etc.), enter API key, model name, and database URL
2. **OAuth Apps** *(optional)* - Add Google/Microsoft OAuth app credentials to enable one-click account connection
3. **Connect Accounts** *(optional)* - Connect Gmail, Outlook, Google Calendar, or Outlook Calendar

That's it. Go back to the chat and start talking.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, get response |
| `/stream` | POST | Send message, stream response (SSE) |
| `/health` | GET | Health check |

```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Do I have any unread emails?"}'

# Stream (SSE)
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Do I have any unread emails?"}'

# Multi-tenant
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "user_123", "message": "What is on my calendar today?"}'
```

## Config

`config.yaml` is created automatically via the settings page. You can also create it manually:

```yaml
provider: openai          # openai / anthropic / azure / dashscope / gemini / ollama
model: gpt-4o
api_key: sk-...           # or omit to use provider's default env var
database: postgresql://user:pass@host:5432/dbname
```

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | LLM provider |
| `model` | Yes | Model name |
| `database` | Yes | PostgreSQL connection URL |
| `api_key` | No | API key (defaults to provider env var) |
| `base_url` | No | Custom endpoint (required for Azure) |
| `system_prompt` | No | System prompt / personality |

## License

[MIT](LICENSE)
