"""API-key discovery utilities for rdai.

The SDK deliberately keeps provider credentials out of its configuration file.
Instead, credentials are discovered from the process environment first and from
an optional ``.env`` file second.  This makes a checked-in ``rdai.yaml`` safe
while still allowing a frictionless local-development setup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

DEFAULT_ENV_FILENAME: Final = ".env"

# Keep this mapping in one place so the CLI, configuration layer, and SDK all
# agree on the canonical provider names and their expected credentials.
PROVIDER_ENV_KEYS: Final[dict[str, str]] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    # 🎯 FIX: Changed to PROJECT_ID for VertexAI
    "vertexai": "VERTEXAI_PROJECT_ID",
    "claude": "CLAUDE_API_KEY",
    "aws_bedrock": "AWS_BEDROCK_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "llama": "LLAMA_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}

PROVIDER_DISPLAY_NAMES: Final[dict[str, str]] = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openai": "OpenAI",
    "vertexai": "VertexAI",
    "claude": "Claude",
    "aws_bedrock": "AWS Bedrock",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Llama",
    "mistral": "Mistral",
    "huggingface": "HuggingFace",
}

# These are the values emitted by ``rdai init``.  They are deliberately not
# treated as credentials by discovery or ``rdai doctor``.
API_KEY_PLACEHOLDERS: Final[dict[str, str]] = {
    "gemini": "your-gemini-api-key",
    "groq": "your-groq-api-key",
    "openai": "your-openai-api-key",
    # 🎯 FIX: Changed placeholder
    "vertexai": "your-gcp-project-id",
    "claude": "your-claude-api-key",
    "aws_bedrock": "your-aws_bedrock-api-key",
    "deepseek": "your-deepseek-api-key",
    "qwen": "your-qwen-api-key",
    "llama": "your-llama-api-key",
    "mistral": "your-mistral-api-key",
    "huggingface": "your-huggingface-api-key",
}


def normalize_provider_name(provider: str) -> str:
    """Return the canonical, registry-friendly spelling of a provider name.

    Registry implementations may support providers beyond the built-ins, so an
    unknown but non-empty name remains valid.  It is simply normalised for
    consistent lookup and configuration storage.
    """

    normalized = provider.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("Provider names cannot be empty.")
    return normalized


def resolve_env_path(env_file: str | Path | None = None) -> Path:
    """Resolve an explicit env path or the local project's ``.env`` path."""

    candidate = Path(env_file) if env_file is not None else Path.cwd() / DEFAULT_ENV_FILENAME
    return candidate.expanduser().resolve()


def read_dotenv(env_file: str | Path | None = None) -> dict[str, str]:
    """Read non-empty values from an env file without mutating ``os.environ``.

    A missing file is normal for zero-configuration usage and therefore returns
    an empty mapping rather than raising an error.
    """

    path = resolve_env_path(env_file)
    if not path.is_file():
        return {}

    values = dotenv_values(path)
    return {
        key: value.strip()
        for key, value in values.items()
        if isinstance(value, str) and value.strip()
    }


def _usable_api_key(value: object) -> str | None:
    """Return a stripped credential unless it is blank or an rdai placeholder."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.casefold() in {
        placeholder.casefold() for placeholder in API_KEY_PLACEHOLDERS.values()
    }:
        return None
    return stripped


def discover_api_keys(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Discover configured built-in provider keys.

    Process environment values take precedence over values in ``env_file``. The
    returned mapping is keyed by canonical provider name and intentionally does
    not include providers that have no usable key.

    Args:
        env_file: Optional path to a dotenv file. Defaults to ``./.env``.
        environ: Optional environment mapping, mainly useful for tests.
    """

    file_values = read_dotenv(env_file)
    active_environment = os.environ if environ is None else environ
    discovered: dict[str, str] = {}

    for provider, variable_name in PROVIDER_ENV_KEYS.items():
        value = _usable_api_key(active_environment.get(variable_name))
        if value is None:
            value = _usable_api_key(file_values.get(variable_name))
        if value is not None:
            discovered[provider] = value

    return discovered


def get_api_key(
    provider: str,
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return one provider's discovered API key, or ``None`` when unavailable."""

    return discover_api_keys(env_file, environ=environ).get(normalize_provider_name(provider))


def provider_key_statuses(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Return readiness booleans for every built-in provider without exposing keys."""

    discovered = discover_api_keys(env_file, environ=environ)
    return {provider: provider in discovered for provider in PROVIDER_ENV_KEYS}
