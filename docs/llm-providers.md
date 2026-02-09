# LLM Providers

OneValet supports multiple LLM providers. Configure your provider in `config.yaml`.

## Supported Providers

| Provider | Install | Env Var |
|----------|---------|---------|
| OpenAI | `uv sync --extra openai` | `OPENAI_API_KEY` |
| Anthropic | `uv sync --extra anthropic` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `uv sync --extra openai` | `AZURE_OPENAI_API_KEY` |
| DashScope | `uv sync --extra dashscope` | `DASHSCOPE_API_KEY` |
| Google Gemini | `uv sync --extra gemini` | `GOOGLE_API_KEY` |
| Ollama | *(included)* | *(none -- runs locally)* |

## Configuration

### OpenAI

```yaml
provider: openai
model: gpt-4o
database: ${DATABASE_URL}
```

### Anthropic

```yaml
provider: anthropic
model: claude-sonnet-4-5-20250929
database: ${DATABASE_URL}
```

### Azure OpenAI

Azure requires `base_url` pointing to your Azure OpenAI resource.

```yaml
provider: azure
model: gpt-4o
base_url: ${AZURE_OPENAI_ENDPOINT}
database: ${DATABASE_URL}
```

### DashScope (Alibaba Cloud)

```yaml
provider: dashscope
model: qwen-max
database: ${DATABASE_URL}
```

### Google Gemini

```yaml
provider: gemini
model: gemini-pro
database: ${DATABASE_URL}
```

### Ollama (Local)

No API key needed. Set `base_url` if Ollama is not on the default address.

```yaml
provider: ollama
model: llama2
base_url: http://localhost:11434
database: ${DATABASE_URL}
```
