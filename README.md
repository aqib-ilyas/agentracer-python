# Agentracer Python SDK

Lightweight AI cost observability. Know exactly what your AI features cost.

[![PyPI version](https://badge.fury.io/py/agentracer.svg)](https://pypi.org/project/agentracer/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install agentracer
```

With provider-specific extras:

```bash
# OpenAI support
pip install agentracer[openai]

# Anthropic support
pip install agentracer[anthropic]

# Google Gemini support
pip install agentracer[gemini]

# All providers
pip install agentracer[all]
```

## Quick Start

```python
import agentracer

# Initialize once at app startup
agentracer.init(
    tracker_api_key="at_your_api_key",
    project_id="your_project_id",
    environment="production"
)
```

## Usage

### OpenAI

```python
from agentracer.openai import openai

# Use exactly like the normal OpenAI client
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    feature_tag="chatbot"  # optional: tag this call
)
```

### Anthropic

```python
from agentracer.anthropic import anthropic

response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
    feature_tag="summarizer"  # optional: tag this call
)
```

### Google Gemini

```python
from agentracer.gemini import gemini

model = gemini.GenerativeModel("gemini-1.5-pro")
response = model.generate_content(
    "Hello!",
    feature_tag="content-gen"  # optional: tag this call
)
```

## Streaming

All providers support streaming. Token usage is automatically tracked after the stream completes.

### OpenAI Streaming

```python
from agentracer.openai import openai

stream = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
    feature_tag="poet"
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
# Telemetry is sent automatically after the stream ends
```

### Anthropic Streaming

```python
from agentracer.anthropic import anthropic

stream = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
    feature_tag="poet"
)

for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text, end="")
```

### Gemini Streaming

```python
from agentracer.gemini import gemini

model = gemini.GenerativeModel("gemini-1.5-pro")

stream = model.generate_content(
    "Write a poem",
    stream=True,
    feature_tag="poet"
)

for chunk in stream:
    print(chunk.text, end="")
```

> Streaming works transparently -- usage is captured from the final chunk (OpenAI), SSE events (Anthropic), or chunk metadata (Gemini), then sent as a single telemetry event after the stream finishes.

## Feature Tags

Feature tags let you group and track costs by feature across your application. There are three ways to apply them:

### 1. Inline (per-call)

Pass `feature_tag` directly on any tracked call:

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this"}],
    feature_tag="summarizer"
)
```

### 2. Decorator

Use the `@observe` decorator to tag all LLM calls inside a function:

```python
import agentracer

@agentracer.observe(feature_tag="chatbot")
def handle_chat(message: str):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    return response

# Async functions work too
@agentracer.observe(feature_tag="async-chatbot")
async def handle_chat_async(message: str):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    return response
```

### 3. Context Manager

Use `feature_context` for block-level tagging:

```python
from agentracer import feature_context

with feature_context("report-generator"):
    # All LLM calls in this block get tagged "report-generator"
    summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize the data"}]
    )
    analysis = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": "Analyze trends"}]
    )
```

## Manual Tracking

For custom integrations or providers not yet supported, use `track()` directly:

```python
import agentracer

agentracer.track(
    model="gpt-4o",
    input_tokens=150,
    output_tokens=300,
    latency_ms=420.5,
    feature_tag="custom-pipeline",
    provider="openai"
)
```

## FastAPI Example

```python
from fastapi import FastAPI
import agentracer
from agentracer.openai import openai

app = FastAPI()

agentracer.init(
    tracker_api_key="at_your_api_key",
    project_id="your_project_id",
    environment="production"
)

@app.post("/chat")
@agentracer.observe(feature_tag="chatbot")
async def chat(message: str):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    return {"reply": response.choices[0].message.content}

@app.post("/summarize")
async def summarize(text: str):
    with agentracer.feature_context("summarizer"):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize: {text}"}]
        )
    return {"summary": response.choices[0].message.content}
```

## Django Example

```python
# settings.py
import agentracer

agentracer.init(
    tracker_api_key="at_your_api_key",
    project_id="your_project_id",
    environment="production"
)

# views.py
from agentracer.openai import openai
from agentracer import feature_context

def chat_view(request):
    message = request.POST.get("message")

    with feature_context("django-chatbot"):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}]
        )

    return JsonResponse({"reply": response.choices[0].message.content})
```

## Configuration

```python
agentracer.init(
    tracker_api_key="at_your_api_key",   # Required: your API key from the dashboard
    project_id="your_project_id",         # Required: your project ID
    environment="production",             # Optional: environment name (default: "production")
    host="https://api.agentracer.dev",    # Optional: API host (default: production)
    debug=False,                          # Optional: enable debug logging (default: False)
    enabled=True                          # Optional: enable/disable tracking (default: True)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracker_api_key` | `str` | *required* | Your API key from the Agentracer dashboard |
| `project_id` | `str` | *required* | Your project ID |
| `environment` | `str` | `"production"` | Environment name (e.g., `"staging"`, `"development"`) |
| `host` | `str` | `"https://api.agentracer.dev"` | API endpoint |
| `debug` | `bool` | `False` | Print telemetry payloads to console |
| `enabled` | `bool` | `True` | Set to `False` to disable all tracking |

## What We Track

Every LLM call automatically captures:

| Field | Description |
|-------|-------------|
| `project_id` | Your project identifier |
| `provider` | LLM provider (`openai`, `anthropic`, `gemini`, `custom`) |
| `model` | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |
| `feature_tag` | Feature label for cost grouping |
| `input_tokens` | Number of input/prompt tokens |
| `output_tokens` | Number of output/completion tokens |
| `latency_ms` | Request latency in milliseconds |
| `environment` | Deployment environment |

We **never** capture prompts, responses, or any PII. Only metadata and usage metrics.

## Troubleshooting

### Telemetry not appearing in the dashboard

1. **Check your API key and project ID** -- ensure they match what's in your [dashboard](https://agentracer.dev/dashboard).
2. **Enable debug mode** to see what's being sent:
   ```python
   agentracer.init(
       tracker_api_key="at_your_api_key",
       project_id="your_project_id",
       debug=True
   )
   ```
3. **Check that tracking is enabled** -- make sure you haven't set `enabled=False`.
4. **Verify network connectivity** -- the SDK sends data to `https://api.agentracer.dev/api/ingest`.

### Import errors for provider modules

If you see `ImportError: openai package not installed`, install the provider extra:

```bash
pip install agentracer[openai]
# or
pip install agentracer[anthropic]
# or
pip install agentracer[gemini]
```

### Feature tags showing as "unknown"

Make sure you're setting feature tags using one of the three methods:

- Inline: `feature_tag="my-feature"` on the call
- Decorator: `@agentracer.observe(feature_tag="my-feature")`
- Context manager: `with agentracer.feature_context("my-feature"):`

### SDK not throwing errors

This is by design. The SDK uses a fire-and-forget pattern and silently catches all exceptions to ensure it never impacts your application's performance or reliability.

## License

MIT
