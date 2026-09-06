from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from rdai.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Discovered models are filtered to identifiers that are suitable
    for text/chat generation through this adapter.
    """

    traits = (
        "coding",
        "reasoning",
        "industry-standard",
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        """Discover models currently exposed by the OpenAI account."""

        if not self.is_available:
            return ()

        try:
            from openai import OpenAI
        except ImportError:
            return ()

        try:
            client = OpenAI(
                api_key=self.api_key,
            )

            response = client.models.list()

            models: list[str] = []

            for model in response.data:
                model_id = getattr(
                    model,
                    "id",
                    None,
                )

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                ):
                    continue

                models.append(
                    model_id.strip()
                )

            return tuple(models)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Keep models suitable for standard text/chat generation.

        OpenAI exposes multiple specialized model families through the same
        catalog. This adapter uses text chat completions, so clearly
        incompatible categories are excluded by capability/category markers.
        """

        filtered: list[str] = []

        excluded_prefixes = (
            "whisper",
            "gpt-image",
            "chatgpt-image",
            "dall-e",
        )

        excluded_markers = (
            "embedding",
            "moderation",
            "transcribe",
            "tts",
            "text-to-speech",
            "realtime",
            "audio",
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

            if any(
                marker in lowered
                for marker in excluded_markers
            ):
                continue

            filtered.append(normalized)

        return filtered

    def rank_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Preserve the provider's discovery order.

        RDAI does not hardcode a preferred OpenAI model as a fallback.
        """

        return models

    def _client(self) -> Any:
        """Create an authenticated OpenAI client."""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Please install the OpenAI SDK using: pip install openai"
            ) from exc

        if not self.is_available:
            raise ValueError(
                "OpenAI API key is missing."
            )

        return OpenAI(
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response using OpenAI chat completions."""

        model = self.ensure_model()

        client = self._client()

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
                "OpenAI returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "OpenAI returned an empty response."
            )

        return str(content)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks from OpenAI."""

        model = self.ensure_model()

        client = self._client()

        # Streaming is controlled by this provider method rather than by
        # exposing provider-specific flags to the public API.
        kwargs = dict(kwargs)
        kwargs.pop(
            "stream",
            None,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        )

        for chunk in response:
            try:
                content = chunk.choices[0].delta.content
            except (
                AttributeError,
                IndexError,
                TypeError,
            ):
                continue

            if content:
                yield str(content)