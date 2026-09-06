"""Public SDK interface for rdai.

The :class:`AI` facade keeps application code independent from a specific
model vendor. It discovers local credentials, constructs provider adapters at
the package boundary, and delegates routing, retry, failover, and streaming
policy to the provider-agnostic core.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .config.discovery import discover_api_keys, normalize_provider_name
from .config.loader import RdaiConfig, load_config
from .core.engine import Failover, ProviderRegistry, Router
from .providers.aws_bedrock import AwsBedrockProvider
from .providers.base import BaseProvider
from .providers.claude import ClaudeProvider
from .providers.deepseek import DeepseekProvider
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .providers.huggingface import HuggingfaceProvider
from .providers.llama import LlamaProvider
from .providers.mistral import MistralProvider
from .providers.openai import OpenAIProvider
from .providers.qwen import QwenProvider
from .providers.vertexai import VertexaiProvider

__version__ = "1.1.0"

_BUILTIN_PROVIDER_CLASSES: Final[
    dict[str, type[BaseProvider]]
] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "vertexai": VertexaiProvider,
    "claude": ClaudeProvider,
    "aws_bedrock": AwsBedrockProvider,
    "deepseek": DeepseekProvider,
    "qwen": QwenProvider,
    "llama": LlamaProvider,
    "mistral": MistralProvider,
    "huggingface": HuggingfaceProvider,
}

_DEFAULT_PROVIDER_ORDER: Final[
    tuple[str, ...]
] = tuple(_BUILTIN_PROVIDER_CLASSES)


class AI:
    """A resilient, provider-agnostic AI generation client."""

    def __init__(
        self,
        *,
        strategy: str | None = None,
        config_path: str | Path | None = None,
        env_file: str | Path | None = None,
        providers: Iterable[BaseProvider] | None = None,
        api_keys: Mapping[str, str] | None = None,
        models: Mapping[str, str] | None = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        config_file = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else (Path.cwd() / "rdai.yaml").resolve()
        )

        effective_env_file = (
            Path(env_file).expanduser().resolve()
            if env_file is not None
            else config_file.parent / ".env"
        )

        self.config: RdaiConfig = load_config(
            config_file,
            env_file=effective_env_file,
        )

        self.strategy = (
            strategy
            if strategy is not None
            else self.config.strategy
        )

        self.registry = ProviderRegistry()

        if providers is None:
            self._register_builtin_providers(
                api_keys=self._resolved_api_keys(
                    effective_env_file,
                    api_keys,
                ),
                models=models,
            )
        else:
            for provider in providers:
                self.registry.register(provider)

        configured_order = (
            self.config.provider_order
            or _DEFAULT_PROVIDER_ORDER
        )

        self.router = Router(
            self.registry,
            strategy=self.strategy,
            priority=configured_order,
        )

        self.failover = Failover(
            self.router,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    @staticmethod
    def _resolved_api_keys(
        env_file: Path,
        explicit_api_keys: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Merge discovered and explicitly supplied API keys."""

        discovered = discover_api_keys(
            env_file
        )

        if explicit_api_keys is None:
            return discovered

        for provider, key in explicit_api_keys.items():
            if not isinstance(
                key,
                str,
            ):
                continue

            if not key.strip():
                continue

            discovered[
                normalize_provider_name(provider)
            ] = key.strip()

        return discovered

    def _register_builtin_providers(
        self,
        *,
        api_keys: Mapping[str, str],
        models: Mapping[str, str] | None,
    ) -> None:
        """Register built-in providers at the package boundary."""

        selected = (
            self.config.provider_order
            or _DEFAULT_PROVIDER_ORDER
        )

        normalized_models = {
            normalize_provider_name(provider): model
            for provider, model in (models or {}).items()
            if isinstance(model, str)
            and model.strip()
        }

        for provider_name in selected:
            provider_class = _BUILTIN_PROVIDER_CLASSES.get(
                provider_name
            )

            if provider_class is None:
                continue

            model_val = normalized_models.get(
                provider_name
            )

            if model_val:
                provider = provider_class(
                    api_key=api_keys.get(
                        provider_name
                    ),
                    model=model_val,
                )
            else:
                provider = provider_class(
                    api_key=api_keys.get(
                        provider_name
                    ),
                )

            self.registry.register(
                provider
            )

    @staticmethod
    def _validate_prompt(
        prompt: str,
    ) -> None:
        """Validate a public generation prompt."""

        if not isinstance(
            prompt,
            str,
        ) or not prompt.strip():
            raise ValueError(
                "prompt must be a non-empty string."
            )

    def generate(
        self,
        prompt: str,
        *,
        traits: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text through the selected provider with automatic failover."""

        self._validate_prompt(
            prompt
        )

        return self.failover.generate(
            prompt,
            traits=traits,
            **kwargs,
        )

    def stream(
        self,
        prompt: str,
        *,
        traits: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text through the selected provider with automatic failover."""

        self._validate_prompt(
            prompt
        )

        return self.failover.stream(
            prompt,
            traits=traits,
            **kwargs,
        )


__all__ = [
    "AI",
    "__version__",
]