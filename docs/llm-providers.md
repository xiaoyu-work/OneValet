# LLM Providers

OneValet supports multiple LLM providers out of the box.

## Supported Providers

| Provider | Install | Models |
|----------|---------|--------|
| OpenAI | `pip install onevalet[openai]` | GPT-4, GPT-3.5 |
| Anthropic | `pip install onevalet[anthropic]` | Claude 3 Opus/Sonnet/Haiku |
| Azure OpenAI | `pip install onevalet[openai]` | GPT-4, GPT-3.5 |
| DashScope | `pip install onevalet[dashscope]` | Qwen |
| Google Gemini | `pip install onevalet[gemini]` | Gemini Pro |
| Ollama | `pip install onevalet[ollama]` | Llama, Mistral, etc. |

## OpenAI

```python
from onevalet import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",           # Or set OPENAI_API_KEY env var
    model="gpt-4",              # Default model
    temperature=0.7,
    max_tokens=4096,
)

agent = MyAgent(llm_client=client)
```

## Anthropic (Claude)

```python
from onevalet import AnthropicClient

client = AnthropicClient(
    api_key="sk-ant-...",       # Or set ANTHROPIC_API_KEY env var
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
)

agent = MyAgent(llm_client=client)
```

## Azure OpenAI

```python
from onevalet import AzureOpenAIClient

client = AzureOpenAIClient(
    api_key="...",
    base_url="https://your-resource.openai.azure.com",
    model="gpt-4",         # Your deployment name
    api_version="2024-12-01-preview",
)

agent = MyAgent(llm_client=client)
```

## DashScope (Alibaba Cloud)

```python
from onevalet import DashScopeClient

client = DashScopeClient(
    api_key="sk-...",           # Or set DASHSCOPE_API_KEY env var
    model="qwen-max",
)

agent = MyAgent(llm_client=client)
```

## Google Gemini

```python
from onevalet import GeminiClient

client = GeminiClient(
    api_key="...",              # Or set GOOGLE_API_KEY env var
    model="gemini-pro",
)

agent = MyAgent(llm_client=client)
```

## Ollama (Local)

```python
from onevalet import OllamaClient

client = OllamaClient(
    base_url="http://localhost:11434",  # Default Ollama URL
    model="llama2",
)

agent = MyAgent(llm_client=client)
```

## Streaming

All clients support streaming:

```python
async for chunk in client.stream_completion(messages):
    print(chunk.content, end="", flush=True)
```

## Custom LLM Client

Implement the `LLMClientProtocol` for custom providers:

```python
from onevalet import LLMClientProtocol, LLMResponse
from typing import List, Dict, Optional, AsyncIterator

class MyCustomClient(LLMClientProtocol):

    async def chat_completion(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ) -> LLMResponse:
        # Your implementation
        response = await my_api_call(messages, tools)
        return LLMResponse(
            content=response.text,
            tool_calls=response.tool_calls,
            usage={"prompt_tokens": 100, "completion_tokens": 50}
        )

    async def stream_completion(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ) -> AsyncIterator:
        # Your streaming implementation
        async for chunk in my_streaming_api(messages):
            yield StreamChunk(content=chunk.text)
```

## LLM Registry

Register and retrieve clients by name:

```python
from onevalet import LLMRegistry

registry = LLMRegistry.get_instance()

# Register a client
registry.register("my-gpt4", OpenAIClient(model="gpt-4"))
registry.register("my-claude", AnthropicClient(model="claude-3-sonnet"))

# Get a client
client = registry.get("my-gpt4")
```

## Configuration via YAML

Create `config/onevalet.yaml`:

```yaml
llm:
  default: main       # Required: LLM for agents
  routing: quick      # Optional: LLM for Orchestrator routing (falls back to default)

  providers:
    main:
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}    # Use env var
      temperature: 0.7

    quick:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

### Provider Configuration Options

Each provider in `providers:` supports:

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | Provider type: `openai`, `anthropic`, `azure`, `dashscope`, `gemini`, `ollama` |
| `model` | Yes | Model name (e.g., `gpt-4o`, `claude-3-sonnet-20240229`) |
| `api_key` | Yes* | API key (use `${ENV_VAR}` for environment variables) |
| `api_key_env` | Yes* | Environment variable name for API key |
| `base_url` | No | Custom API endpoint (for Azure, vLLM, proxies) |
| `temperature` | No | Temperature (default: 0.7) |
| `max_tokens` | No | Max output tokens |
| `timeout` | No | Request timeout in seconds (default: 60.0) |

*Either `api_key` or `api_key_env` is required.

### Azure OpenAI Example

```yaml
llm:
  default: azure-gpt4

  providers:
    azure-gpt4:
      provider: azure
      model: gpt-4o                    # Your deployment name
      api_key: ${AZURE_OPENAI_API_KEY}
      base_url: https://your-resource.openai.azure.com
```

### Loading Configuration

```python
from onevalet import Orchestrator

# Load from config directory
orchestrator = Orchestrator(config_dir="./config")
await orchestrator.initialize()
```

### Agent-specific LLM

```python
@valet(llm="claude")  # Uses claude provider
class MyAgent(StandardAgent):
    ...

@valet  # Uses default
class OtherAgent(StandardAgent):
    ...
```

## Best Practices

1. **Use environment variables** - Never hardcode API keys
2. **Set appropriate timeouts** - LLM calls can be slow
3. **Handle rate limits** - Implement retry logic
4. **Monitor costs** - Track token usage
5. **Use streaming for UX** - Better user experience for long responses
