# Getting Started

Deploy OneValet and send your first request in 5 minutes.

## 1. Clone the repo

```bash
git clone https://github.com/xiaoyu-work/onevalet.git
cd onevalet
```

## 2. Install dependencies

```bash
# Pick your LLM provider
uv sync --extra openai
# or
uv sync --extra anthropic
# or install everything
uv sync --all-extras
```

## 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the keys you need:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/onevalet
```

See [Configuration](configuration.md) for the full list of environment variables.

## 4. Configure the server

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` to set your provider, model, and database:

```yaml
provider: openai
model: gpt-4o
database: ${DATABASE_URL}
```

## 5. Start the server

```bash
python -m onevalet
```

The server starts on `http://0.0.0.0:8000` by default.

## 6. Send requests

### Health check

```bash
curl http://localhost:8000/health
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "user_1", "message": "Hello!"}'
```

### Streaming

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "user_1", "message": "Hello!"}'
```

The `/stream` endpoint returns server-sent events. See [Streaming](streaming.md) for details.

## Custom agents

OneValet ships with built-in agents, but you can create your own using `@valet`, `StandardAgent`, and `InputField`. See the [Agents](agents.md) guide.

## Next steps

- [Configuration](configuration.md) - Full config reference
- [Agents](agents.md) - Create custom agents
- [Tools](tools.md) - Add tools for LLM function calling
- [LLM Providers](llm-providers.md) - Switch between OpenAI, Anthropic, Azure, and more
