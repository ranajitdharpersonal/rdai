<div align="center">
  <pre>
██████╗ ██████╗  █████╗ ██╗
██╔══██╗██╔══██╗██╔══██╗██║
██████╔╝██║  ██║███████║██║
██╔══██╗██║  ██║██╔══██║██║
██║  ██║██████╔╝██║  ██║██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝
  </pre>

  <h1>rdai — Multi-Brain AI Orchestrator</h1>
  <p><b>One Python SDK. Any AI Provider. Resilient Auto-Failover.</b></p>
  <p>Route requests across Gemini, OpenAI, Groq, Claude and more — with live model discovery, automatic failover, streaming, health checks, and an interactive CLI.</p>

  <p>
    <a href="https://pypi.org/project/rdai/"><img src="https://img.shields.io/pypi/v/rdai?color=blue" alt="PyPI" /></a>
    <a href="https://pypi.org/project/rdai/"><img src="https://img.shields.io/pypi/pyversions/rdai" alt="Python" /></a>
    <a href="https://github.com/ranajitdharpersonal/rdai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT" /></a>
    <a href="https://pypi.org/project/rdai/"><img src="https://img.shields.io/pypi/dm/rdai" alt="Downloads" /></a>
    <a href="https://github.com/ranajitdharpersonal/rdai/stargazers"><img src="https://img.shields.io/github/stars/ranajitdharpersonal/rdai?style=social" alt="Stars" /></a>
  </p>

  <b>👑 Created by:</b> Ranajit Dhar &nbsp;|&nbsp; 🌐 <a href="https://ranajitdhar.in">ranajitdhar.in</a> &nbsp;|&nbsp; 📦 <a href="https://pypi.org/project/rdai/">PyPI</a> &nbsp;|&nbsp; <b>Version:</b> v1.1.0
</div>

---

## ⚡ What rdai solves

```text
Your selected provider or model fails
                ↓
   rdai classifies the failure
                ↓
   Refresh discovered models when needed
                ↓
 Retry safely or fail over to another provider
                ↓
        Your application keeps running
```

`rdai` is a provider-agnostic Python SDK and CLI for resilient AI generation. It gives your application one interface across multiple AI providers, automatically discovers available generation-capable models when you do not explicitly pin one, and handles provider/model failures through retry and failover policies.

---

## 🩺 See it in action — `rdai doctor`

The built-in doctor checks configured providers live and reports real connectivity/authentication results.

![rdai doctor output](./assets/doctor-screenshot.png)

**Real provider diagnostics — live authentication, reachability, and latency in one command.**

One command tells you which providers are configured and reachable before you ship.

---

## 📦 Installation

```bash
pip install rdai
```

## 🚦 Quick Start

```bash
rdai init
# choose your providers and routing strategy
# then add your API keys to the generated .env file

rdai doctor
# verify configured providers
```

Example `.env`:

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
# add other provider keys as selected...
```

Your API keys stay local to your environment and are used only when the corresponding provider is called.

---

## 💻 Python SDK Usage

No need to learn ten different SDKs — `rdai` gives you one unified interface.

```python
from rdai import AI

ai = AI()

response = ai.generate(
    "Write a multi-agent orchestration script in Python."
)

print(response)
```

`rdai` routes the request through your configured strategy and automatically handles eligible provider failures.

---

## 🌊 Streaming

Stream generated output through the same provider-agnostic interface:

```python
from rdai import AI

ai = AI()

for chunk in ai.stream("Explain dependency inversion in simple terms."):
    print(chunk, end="")
```

For the CLI:

```bash
rdai generate "Explain dependency inversion in simple terms." --stream
```

Streaming failover is safety-aware: failures that happen before any output can trigger recovery or failover, while a stream that has already emitted partial output is not restarted on another provider.

---

## 🔎 Live Model Discovery

When you do not explicitly provide a model, `rdai` discovers available models from the provider at runtime.

The discovery layer:

- retrieves the provider's current model catalog when supported
- filters out models that are not suitable for text generation
- caches the discovered candidates
- excludes a discovered model after a model-availability/access failure
- refreshes discovery and selects another candidate when possible

This means model selection does not depend on a hardcoded fallback model list.

---

## 🎯 Choosing Models

You can explicitly pin a model for a provider:

```python
from rdai import AI

ai = AI(
    models={
        "gemini": "your-model-id",
        "groq": "your-model-id",
    }
)

response = ai.generate("Hello world!")
```

### Explicit Model Authority

When you provide a model explicitly, `rdai` treats that model as authoritative.

It will not silently replace your requested model with another model during automatic model recovery. This keeps explicit configuration predictable and prevents unexpected model substitution.

---

## 🛠️ Bring Your Own Model (BYOM)

Plug a custom or private provider into the same failover and routing system:

```python
from rdai import AI
from rdai.providers.base import BaseProvider


class CustomNexusProvider(BaseProvider):
    traits = ["reasoning", "proprietary"]

    def __init__(self, api_key: str):
        super().__init__(api_key, "nexus-v1")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        # Call your private/custom API here.
        return f"[NexusAI] Response for: {prompt}"


ai = AI(
    providers=[
        CustomNexusProvider(api_key="your_custom_key"),
    ]
)

print(
    ai.generate("How should I structure a Next.js application?")
)
```

Custom providers remain compatible with the core routing/failover engine without requiring changes to the built-in provider registry.

---

## ✨ Why rdai?

| | |
|---|---|
| ✔ **Interactive CLI** | Guided setup wizard with strategy and provider selection |
| ✔ **Live Model Discovery** | Uses current provider catalogs when no model is pinned |
| ✔ **Automatic Failover** | Recovers from eligible provider and model failures |
| ✔ **Streaming** | Unified `AI.stream()` API plus CLI streaming |
| ✔ **Smart Routing** | Selects providers according to routing strategy and request traits |
| ✔ **Provider Agnostic** | 11 built-in providers plus BYOM |
| ✔ **No Hardcoded Model Fallbacks** | Discovered models are the source of truth |
| ✔ **Live Doctor** | Real provider diagnostics and health checks |

---

## ⚙️ Routing Strategies & Configuration

`rdai` reads its setup from an optional `rdai.yaml` in your working directory. Environment variables take precedence over `.env` values.

Example:

```yaml
strategy: smart
providers:
  - gemini
  - openai
  - groq
```

### `smart`

Selects a ready provider that best matches the request and configured provider capabilities.

### `manual`

Follows the configured provider order.

```yaml
strategy: manual
provider_order:
  - gemini
  - groq
  - openai
```

The failover engine can still use other ready providers for eligible transient failures.

---

## 🔐 Environment Variables

Common provider configuration:

```env
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
CLAUDE_API_KEY=
DEEPSEEK_API_KEY=
QWEN_API_KEY=
LLAMA_API_KEY=
MISTRAL_API_KEY=
HUGGINGFACE_API_KEY=
AWS_BEDROCK_API_KEY=
VERTEXAI_PROJECT_ID=
```

`rdai` gives process environment variables precedence over values loaded from `.env`.

---

## 🎛️ Supported AI Providers

| Provider | Environment Variable | Backend Logic |
| :--- | :--- | :--- |
| **Gemini** | `GEMINI_API_KEY` | Modern `google.genai` SDK |
| **OpenAI** | `OPENAI_API_KEY` | Official `openai` SDK |
| **Groq** | `GROQ_API_KEY` | Official `groq` SDK |
| **VertexAI** | `VERTEXAI_PROJECT_ID` | Google GenAI Vertex AI client |
| **Claude** | `CLAUDE_API_KEY` | Direct Anthropic REST API |
| **AWS Bedrock** | `AWS_BEDROCK_API_KEY` | AWS `boto3` SDK |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Direct DeepSeek REST API |
| **Qwen** | `QWEN_API_KEY` | Alibaba DashScope REST API |
| **Llama** | `LLAMA_API_KEY` | OpenAI-compatible API |
| **Mistral** | `MISTRAL_API_KEY` | Direct Mistral REST API |
| **HuggingFace** | `HUGGINGFACE_API_KEY` | Hugging Face Inference API |

---

## 🖥️ CLI Command Reference

| Command | Description |
| :--- | :--- |
| `rdai init` | Setup workspace, strategy, and providers |
| `rdai generate "<prompt>"` | Generate a response |
| `rdai generate "<prompt>" --stream` | Stream a response |
| `rdai doctor` | Live provider diagnostics |
| `rdai config` | View active routing strategy and provider order |
| `rdai benchmark` | Run a latency test across active providers |
| `rdai health` | Check overall system health |
| `rdai about` | Learn about rdai and its architecture |

---

## 🧠 Failure Recovery

`rdai` distinguishes between different classes of failure so recovery remains predictable.

For transient provider failures such as timeouts or rate limits, the failover engine can move to another ready provider.

For discovered-model failures, `rdai` can refresh provider discovery, exclude the failed discovered model, and retry with another discovered candidate when available.

For an explicitly requested model, the requested model remains authoritative and is not silently replaced.

For streaming, recovery is attempted only when it is safe to do so before partial output has been emitted.

---

## 📊 Health & Diagnostics

Use the CLI to inspect readiness before production traffic:

```bash
rdai doctor
rdai health
rdai benchmark
```

These commands provide operational visibility without exposing API secrets.

---

## 📂 Examples

Ready-to-run examples are available in [`/examples`](https://github.com/ranajitdharpersonal/rdai/tree/main/examples).

```bash
git clone https://github.com/ranajitdharpersonal/rdai.git
cd rdai/examples

python basic_chat.py
python smart_failover.py
python custom_brain.py
```

---

## 🗺️ Roadmap

- ✅ **v1.0.0** — Multi-provider orchestrator core, automatic failover, setup wizard, and live diagnostics
- ✅ **v1.0.1** — CLI/dashboard improvements, expanded package metadata, and corrected repository URLs
- ✅ **v1.0.2** — Unified core architecture, dependency fixes, REST timeouts, and improved doctor diagnostics
- ✅ **v1.1.0** — Live model discovery, model filtering, explicit model authority, model recovery, provider-agnostic streaming, streaming-aware failover, expanded provider support, and release packaging updates

Future releases will build on the orchestration core with additional routing intelligence, observability, and provider integrations.

Full version history: [CHANGELOG.md](https://github.com/ranajitdharpersonal/rdai/blob/main/CHANGELOG.md)

---

## 🤝 Contributing

Issues and pull requests are welcome.

Please check [open issues](https://github.com/ranajitdharpersonal/rdai/issues) or open a new issue to discuss larger changes before submitting a PR.

If `rdai` saved you from a 3am provider outage, a ⭐ on the repo helps other developers discover it.

---

<div align="center">
  <i>Built with ❤️ for the next generation of resilient AI applications.</i>

  <br /><br />

  <a href="https://github.com/ranajitdharpersonal/rdai">⭐ Star this repo</a> · <a href="https://github.com/ranajitdharpersonal/rdai/issues">Report a bug</a> · <a href="https://pypi.org/project/rdai/">View on PyPI</a>
</div>
