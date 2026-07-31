<div align="center">
  <pre>
██████╗ ██████╗  █████╗ ██╗
██╔══██╗██╔══██╗██╔══██╗██║
██████╔╝██║  ██║███████║██║
██╔══██╗██║  ██║██╔══██║██║
██║  ██║██████╔╝██║  ██║██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝
  </pre>
</div>

# 🚀 rdai (Ranajit Dhar AI) - Multi-Brain AI Orchestrator

> **One Interface. Any AI. Unbreakable Auto-Failover.**

**Version:** v1.0.0  
**👑 Created by:** Ranajit Dhar  
**🌐 Website:** [https://ranajitdhar.in](https://ranajitdhar.in)

`rdai` is a Quantum-Ready, Self-Healing AI Operating System and Python SDK. It allows you to call multiple AI models (Gemini, OpenAI, Claude, DeepSeek, and more) through one single, unified interface. It automatically discovers your API keys, selects the best provider based on your strategy, and features a limitless **Circuit Breaker** that silently auto-failovers to a backup provider when rate limits, crashes, or timeouts occur. 

Zero downtime. 100% Reliability.

---

## ✨ Key Features

*   🧠 **11+ Built-in AI Brains:** Native support for Gemini, OpenAI, Groq, Claude, DeepSeek, Qwen, Llama, Mistral, VertexAI, AWS Bedrock, and HuggingFace.
*   ⚡ **Unbreakable Auto-Failover:** If your primary model crashes or hits a rate limit, the system instantly and silently routes the prompt to the next available model.
*   🛠️ **Bring-Your-Own-Model (BYOM):** Inject any custom/private AI API into the failover chain without changing the core framework.
*   🩺 **Live Ping Diagnostics:** The built-in CLI doctor actually pings the AI servers to report real-time ALIVE/ERROR status and latency.
*   🪄 **Smart Auto-Setup:** Automatically installs missing required SDKs in the background during initialization. Prevents accidental configuration overwrites.

---

## 📦 Installation

```bash
pip install rdai
# Or if using local source: pip install -r requirements.txt
```
---
## 🚦 Quick Start (The CLI Wizard)

`rdai` comes with a powerful CLI dashboard. To configure your multi-brain environment, simply run:

```bash
rdai init
```

This interactive wizard will let you select your AI providers, define your failover strategy, auto-install missing packages, and generate your `.env` and `rdai.yaml` files.  

Next, add your API keys to the generated `.env` file:Code snippet

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
# Add others as selected...
```
**(Note: Your API keys are kept strictly local and secure. They are never hardcoded.)**

Finally, verify your setup with the live diagnostic tool:

```bash
rdai doctor
```
The doctor will scan your `.env` file, ping the AI networks, and show you the real-time latency and health of your failover chain.

---

## 💻 Python SDK Usage (Writing Code)

Using the SDK is incredibly simple. You don't need to learn 10 different SDKs; rdai standardizes everything.

```python
from rdai import AI

# Initialize the AI Engine. It automatically loads your rdai.yaml strategy.
ai = AI()

# Send a prompt. If the first model fails, it silently falls back to the next!
response = ai.generate("Write a multi-agent orchestration script in Python.")
print(response)
```

# Bring Your Own Model (BYOM)

Want to use a custom API? Inject it directly into the engine:

```python
from rdai.providers.base import BaseProvider
from rdai import AI
import requests

class CustomNexusProvider(BaseProvider):
    def __init__(self, api_key):
        super().__init__(api_key, "nexus-v1")
        
    def generate(self, prompt: str, **kwargs) -> str:
        res = requests.post("[https://api.nexus.com/v1](https://api.nexus.com/v1)", headers={"Key": self.api_key}, json={"text": prompt})
        return res.json()["reply"]

# Inject your custom engine into the failover chain!
ai = AI(providers=[CustomNexusProvider(api_key="your_custom_key")])
print(ai.generate("Hello Custom Engine!"))
```

---

## ⚙️ Routing Strategies & Configuration

`rdai` reads your setup from an optional `rdai.yaml` file in the current directory. Environment variables take precedence over values in `.env`

```yaml
strategy: smart
provider_order:
  - gemini
  - openai
  - groq
```

---

### 🧠 Routing Strategies

*   **`smart`**: Selects a ready provider whose traits best fit the specific request.
*   **`manual`**: Strictly follows the exact `provider_order` defined in your `rdai.yaml`.

> **Note:** Both strategies retain the remaining ready providers as automatic fallbacks for transient rate-limit and timeout failures.

---

## 🎛️ Supported AI Engines & Environment Keys

| Provider Engine | Environment Variable | Backend Logic |
| :--- | :--- | :--- |
| **Gemini** | `GEMINI_API_KEY` | Modern `google.genai` SDK |
| **OpenAI** | `OPENAI_API_KEY` | Official `openai` SDK |
| **Groq** | `GROQ_API_KEY` | Official `groq` SDK |
| **VertexAI** | `VERTEXAI_API_KEY` | GCP Project ID via `google.genai` |
| **Claude** | `CLAUDE_API_KEY` | Direct Anthropic REST API |
| **AWS Bedrock** | `AWS_BEDROCK_API_KEY` | AWS `boto3` SDK |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Direct DeepSeek REST API |
| **Qwen** | `QWEN_API_KEY` | Alibaba DashScope REST API |
| **Llama** | `LLAMA_API_KEY` | Universal OpenAI-Compatible API |
| **Mistral** | `MISTRAL_API_KEY` | Direct Mistral REST API |
| **HuggingFace** | `HUGGINGFACE_API_KEY` | HF Serverless Inference API |


---

## 🛠️ Complete CLI Command Reference

*   `rdai init` - Setup your workspace, strategies, and AI providers.
*   `rdai doctor` - Run a live `.env` network scan and API health check.
*   `rdai config` - View your active routing strategy and failover chain.
*   `rdai benchmark` - Run a latency test across your active models.
*   `rdai health` - Check overall internal system health and readiness.
*   `rdai about` - Learn about the master orchestration architecture.

---
<div align="center">
  <i>Built with ❤️ for the next generation of unbreakable AI applications.</i>
</div>