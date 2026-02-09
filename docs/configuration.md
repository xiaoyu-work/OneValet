# Configuration

## config.yaml

```yaml
provider: openai
model: gpt-4o
database: ${DATABASE_URL}
```

`${VAR}` will be replaced with the corresponding environment variable value.

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | `openai` / `anthropic` / `azure` / `dashscope` / `gemini` / `ollama` |
| `model` | Yes | e.g. `gpt-4o`, `claude-sonnet-4-5-20250929`, `qwen-max` |
| `database` | Yes | PostgreSQL DSN |

### Optional Fields

| Field | Description |
|-------|-------------|
| `api_key` | LLM API key. If omitted, reads from provider's default env var (see below) |
| `base_url` | Custom LLM endpoint. Required for Azure OpenAI |
| `system_prompt` | Customize assistant personality, e.g. `"You are a travel assistant"` |

## Starting the Server

```bash
# Default: reads config.yaml, listens on 0.0.0.0:8000
python -m onevalet

# Custom config path
ONEVALET_CONFIG=my_config.yaml python -m onevalet

# Custom host/port
ONEVALET_HOST=127.0.0.1 ONEVALET_PORT=9000 python -m onevalet
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

### Agent Services

| Env Var | Agent | Get it from |
|---------|-------|-------------|
| `WEATHER_API_KEY` | Weather | [weatherapi.com](https://www.weatherapi.com/) |
| `GOOGLE_MAPS_API_KEY` | Maps, Directions, Air Quality, Hotel Search | [Google Cloud Console](https://console.cloud.google.com/) |
| `GOOGLE_SEARCH_API_KEY` | Google Search | [Google Cloud Console](https://console.cloud.google.com/) |
| `GOOGLE_SEARCH_ENGINE_ID` | Google Search | [Programmable Search Engine](https://programmablesearchengine.google.com/) |
| `AMADEUS_API_KEY` | Flight Search, Hotel Search | [Amadeus for Developers](https://developers.amadeus.com/) |
| `AMADEUS_API_SECRET` | Flight Search, Hotel Search | [Amadeus for Developers](https://developers.amadeus.com/) |
| `TRACK17_API_KEY` | Shipment Tracking | [17TRACK API](https://api.17track.net/) |

All service keys are optional. Agents without their key configured will return empty results.

### OAuth Credentials

Email and Calendar agents require per-user Google OAuth tokens, stored in the database via `CredentialStore`.

See `.env.example` for a complete template.
