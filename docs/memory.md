# Memory System

OneValet provides long-term memory capabilities powered by [mem0](https://mem0.ai/).

## Overview

Memory enables agents to remember user preferences and information across conversations. When enabled, the system automatically:

1. **Recalls** relevant memories before agent execution
2. **Stores** collected information after successful completion

## Configuration

### YAML Configuration

```yaml
# config.yaml
memory:
  enabled: true
  use_platform: true
  api_key: ${MEM0_API_KEY}
```

### Self-Hosted mem0

```yaml
memory:
  enabled: true
  use_platform: false
  vector_store_provider: qdrant
  vector_store_config:
    host: localhost
    port: 6333
  llm_provider: openai
  llm_model: gpt-4o-mini
  embedder_provider: openai
  embedder_model: text-embedding-3-small
```

## Per-Agent Configuration

Enable memory for specific agents in workflow:

```yaml
# workflow.yaml
agents:
  FlightAgent:
    enable_memory: true
    memory_config:
      recall_limit: 10
      store_on_complete: true

  HotelAgent:
    enable_memory: true
```

## Field Filtering

Control which fields are remembered:

```yaml
memory:
  enabled: true
  api_key: ${MEM0_API_KEY}
  remember_fields:
    - email
    - phone
    - preferences
  exclude_fields:
    - password
    - credit_card
```

## Auto Behaviors

```yaml
memory:
  enabled: true
  api_key: ${MEM0_API_KEY}
  auto_recall: true   # Recall memories before agent runs
  auto_store: true    # Store collected fields after completion
```

## Best Practices

1. **Use mem0 platform for production** - Managed infrastructure, no setup
2. **Filter sensitive fields** - Use `exclude_fields` for passwords, tokens
3. **Set field allowlist** - Use `remember_fields` to limit what's stored
4. **Test with self-hosted first** - Use local Qdrant for development
