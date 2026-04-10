# Getting Started

Deploy Koa and start chatting in 5 minutes.

## Prerequisites

- **Python 3.12+**
- **PostgreSQL 16+** — or use Docker: `docker compose up -d db`
- **An LLM API key** — OpenAI, Anthropic, Azure, Gemini, DashScope, or Ollama (local)

## 1. Clone and install

```bash
git clone https://github.com/xiaoyu-work/koa.git
cd koa

# Pick your LLM provider
uv sync --extra openai
# or
uv sync --extra anthropic
# or install everything
uv sync --all-extras
```

## 2. Configure

```bash
cp .env.example .env
cp config.yaml.example config.yaml
```

Edit `.env` with your database and API keys:

```
DATABASE_URL=postgresql://user:pass@localhost:5432/koa
OPENAI_API_KEY=sk-...
```

Edit `config.yaml`:

```yaml
database: ${DATABASE_URL}

llm:
  provider: openai
  model: gpt-4o

embedding:
  provider: openai
  model: text-embedding-3-small
```

See [Configuration](configuration.md) for the full reference.

## 3. Start the server

```bash
uv run koa serve
```

The server starts on `http://localhost:8000` by default.

## 4. Chat

Open a second terminal:

```bash
uv run koa chat
```

```
Connected to Koa v0.1.1 at http://localhost:8000
Type your message and press Enter. Ctrl+C to quit.

You: Hello!
Koa: Hi! I'm Koa, your AI assistant. How can I help you today?
```

## 5. Or use the API directly

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Streaming (SSE)
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

See [Streaming](streaming.md) for SSE event details.

## Docker

If you prefer Docker:

```bash
cp .env.example .env           # edit with your API keys
docker compose up
```

This starts PostgreSQL and Koa together. Chat with:

```bash
docker compose exec app koa chat
```

## Next steps

- [Configuration](configuration.md) — Full config reference
- [Agents](agents.md) — Create custom agents with `@valet` and `InputField`
- [Tools](tools.md) — Add tools for LLM function calling
- [LLM Providers](llm-providers.md) — Switch between OpenAI, Anthropic, Azure, and more
