from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from rdai.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Discovered candidates are restricted to models suitable for the
    standard chat-completions path used by this adapter.
    """

    traits = (
        "ultra-fast",
        "groq-lpu",
        "low-latency",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        """Discover active models exposed by Groq."""

        if not self.is_available:
            return ()

        try:
            from groq import Groq
        except ImportError:
            return ()

        try:
            client = Groq(api_key=self.api_key)
            response = client.models.list()

            models: list[str] = []

            for model in response.data:
                model_id = getattr(model, "id", None)
                is_active = getattr(model, "active", True)

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                    and is_active
                ):
                    continue

                models.append(model_id.strip())

            return tuple(models)

        except Exception:
            # Discovery is best-effort.
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Keep models appropriate for standard text chat generation.

        Groq exposes several categories through the same model catalog,
        including Compound systems and specialized preview models. This
        adapter uses the normal chat-completions path, so known non-chat
        families are excluded by category rather than by individual model ID.
        """

        filtered: list[str] = []

        excluded_prefixes = (
            "canopylabs/orpheus",
            "meta-llama/llama-prompt-guard",
            "groq/compound",
        )

        for model in models:
            normalized = model.strip()

            if not normalized:
                continue

            lowered = normalized.lower()

            if any(
                lowered.startswith(prefix)
                for prefix in excluded_prefixes
            ):
                continue

            # Audio/speech models are not valid for this text-chat adapter.
            if any(
                marker in lowered
                for marker in (
                    "whisper",
                    "speech",
                )
            ):
                continue

            filtered.append(normalized)

        return filtered

    def rank_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Prefer production language models while preserving discovery order."""

        candidates = list(models)

        preferred_markers = (
            "llama",
            "gpt-oss",
            "qwen",
            "kimi",
        )

        def score(model: str) -> tuple[int, int]:
            lowered = model.lower()

            preferred = any(
                marker in lowered
                for marker in preferred_markers
            )

            # Production/general language models should be preferred over
            # other less obvious catalog entries.
            return (
                0 if preferred else 1,
                candidates.index(model),
            )

        return sorted(
            candidates,
            key=score,
        )

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Groq chat completions."""

        if not self.is_available:
            raise ValueError(
                "Groq API key is missing."
            )

        model = self.ensure_model()

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "Please install the Groq SDK using: pip install groq"
            ) from exc

        client = Groq(
            api_key=self.api_key,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )

        try:
            content = response.choices[0].message.content
        except (
            AttributeError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Groq returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return str(content)