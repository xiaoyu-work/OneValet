# Integration Tests

End-to-end tests that verify orchestrator routing, agent tool selection, and response quality using a **real LLM**.

## How it works

- A real LLM client makes routing and tool selection decisions
- Tool executors are replaced with mocks that return canned data (no external services needed)
- A `ToolCallRecorder` captures which agents and tools were called
- An LLM-as-judge validates response quality

## Setup

Set the following environment variables:

```bash
# Required
export INTEGRATION_TEST_API_KEY=your-api-key

# Optional (defaults shown)
export INTEGRATION_TEST_PROVIDER=openai        # openai | anthropic | azure | gemini
export INTEGRATION_TEST_MODEL=gpt-4o-mini      # model name
export INTEGRATION_TEST_BASE_URL=              # custom base URL (Azure, Ollama, etc.)
```

### Provider Examples

**OpenAI:**
```bash
export INTEGRATION_TEST_PROVIDER=openai
export INTEGRATION_TEST_MODEL=gpt-4o-mini
export INTEGRATION_TEST_API_KEY=sk-...
```

**Azure OpenAI:**
```bash
export INTEGRATION_TEST_PROVIDER=azure
export INTEGRATION_TEST_MODEL=azure/gpt-4o-mini
export INTEGRATION_TEST_API_KEY=your-azure-key
export INTEGRATION_TEST_BASE_URL=https://your-resource.openai.azure.com/
```

**Anthropic:**
```bash
export INTEGRATION_TEST_PROVIDER=anthropic
export INTEGRATION_TEST_MODEL=claude-haiku-4-5-20251001
export INTEGRATION_TEST_API_KEY=sk-ant-...
```

## Running Tests

```bash
# All integration tests
pytest tests/integration/ -v --timeout=120

# Routing tests only
pytest tests/integration/test_routing.py -v --timeout=120

# Single agent tests
pytest tests/integration/agents/test_expense_agent.py -v --timeout=120

# With token usage logs
pytest tests/integration/ -v --timeout=120 -s
```

## Cost Estimates (per full run)

| Provider | Model | Estimated Cost |
|----------|-------|---------------|
| OpenAI | gpt-4o-mini | ~$1-2 |
| OpenAI | gpt-4o | ~$10-20 |
| Anthropic | claude-haiku-4-5 | ~$1-3 |
| Azure OpenAI | gpt-4o-mini | ~$1-2 |

## Test Structure

```
tests/integration/
  conftest.py              # Core fixtures (LLM client, orchestrator, recorder)
  test_routing.py          # Orchestrator routes message → correct agent
  test_edge_cases.py       # Ambiguous inputs, multi-step, unknown intent
  agents/
    conftest.py            # Shared agent test helpers
    test_<agent>.py        # Per-agent: tool selection + arg extraction + response quality
```
