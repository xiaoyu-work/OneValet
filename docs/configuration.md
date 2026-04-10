# Configuration

## config.yaml

```yaml
database: ${DATABASE_URL}

llm:
  provider: openai
  model: gpt-4o

# Optional: embedding for long-term memory
# embedding:
#   provider: openai
#   model: text-embedding-3-small
```

`${VAR}` will be replaced with the corresponding environment variable value.

| Field | Required | Description |
|-------|----------|-------------|
| `database` | Yes | PostgreSQL DSN |
| `llm.provider` | Yes | `openai` / `anthropic` / `azure` / `dashscope` / `gemini` / `ollama` |
| `llm.model` | Yes | e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`, `qwen-max` |

### Optional Fields

| Field | Description |
|-------|-------------|
| `llm.api_key` | LLM API key. If omitted, reads from provider's default env var (see below) |
| `llm.base_url` | Custom LLM endpoint. Required for Azure OpenAI |
| `embedding.provider` | Embedding provider for memory (`openai` / `azure`) |
| `embedding.model` | e.g. `text-embedding-3-small` |
| `system_prompt` | Customize assistant personality, e.g. `"You are a travel assistant"` |

## Starting the Server

```bash
# Default: reads config.yaml, listens on 127.0.0.1:8000
uv run koa serve

# Custom host/port
uv run koa serve --host 0.0.0.0 --port 9000

# Or via env vars
KOA_HOST=0.0.0.0 KOA_PORT=9000 uv run koa serve

# Custom config path
KOA_CONFIG=my_config.yaml uv run koa serve
```

### CLI Chat

```bash
# Interactive chat (connects to running server)
uv run koa chat
```

## Environment Variables

### LLM API Key (based on provider)

| Provider | Env Var |
|----------|---------|
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| azure | `AZURE_OPENAI_API_KEY` |
| dashscope | `DASHSCOPE_API_KEY` |
| gemini | `GOOGLE_API_KEY` |
| ollama | Not needed |

### Embedding (Memory)

Koa uses OpenAI embeddings for long-term memory. If your LLM provider is OpenAI or Azure, the same API key is reused automatically. For other providers (Anthropic, DashScope, Gemini, Ollama), set `OPENAI_API_KEY` for embedding support:

| Provider | Embedding Key |
|----------|--------------|
| openai | Reused automatically |
| azure | Reused automatically |
| anthropic / dashscope / gemini / ollama | Requires `OPENAI_API_KEY` |

### Agent Services

| Env Var | Agent | Get it from |
|---------|-------|-------------|
| `WEATHER_API_KEY` | Weather | [weatherapi.com](https://www.weatherapi.com/) |
| `GOOGLE_MAPS_API_KEY` | Maps, Directions, Air Quality, Hotel Search | [Google Cloud Console](https://console.cloud.google.com/) |
| `GOOGLE_SEARCH_API_KEY` | Google Search | [Google Cloud Console](https://console.cloud.google.com/) |
| `GOOGLE_SEARCH_ENGINE_ID` | Google Search | [Programmable Search Engine](https://programmablesearchengine.google.com/) |
| `JINA_API_KEY` | Flight Search, Hotel Search (via Jina Reader) | [jina.ai](https://jina.ai/) |
| `TRACK17_API_KEY` | Shipment Tracking | [17TRACK API](https://api.17track.net/) |

All service keys are optional. Agents without their key configured will return empty results.

### OAuth Credentials

Email and Calendar agents require per-user Google OAuth tokens, stored in the database via `CredentialStore`.

See `.env.example` for a complete template.
