from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional

from rdai.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Discovered candidates are restricted to models suitable for the
    standard chat-completions API used by this adapter.
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
            client = Groq(
                api_key=self.api_key,
            )

            response = client.models.list()

            models: list[str] = []

            for item in getattr(response, "data", ()):
                model_id = getattr(
                    item,
                    "id",
                    None,
                )

                is_active = getattr(
                    item,
                    "active",
                    True,
                )

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                    and is_active
                ):
                    continue

                models.append(
                    model_id.strip()
                )

            return tuple(models)

        except Exception:
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Keep models suitable for standard text chat."""

        filtered: list[str] = []

        excluded_prefixes = (
            "canopylabs/orpheus",
            "meta-llama/llama-prompt-guard",
            "meta-llama/prompt-guard",
            "groq/compound",
        )

        excluded_markers = (
            "whisper",
            "speech",
            "tts",
            "text-to-speech",
            "transcribe",
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
        """Preserve provider discovery order."""

        return models

    def _client(self) -> Any:
        """Create an authenticated Groq client."""

        if not self.is_available:
            raise ValueError(
                "Groq API key is missing."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "Please install the Groq SDK using: pip install groq"
            ) from exc

        return Groq(
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response using Groq chat completions."""

        model = self.ensure_model()

        client = self._client()

        request_kwargs = dict(kwargs)
        request_kwargs.pop(
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
            **request_kwargs,
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

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks from Groq."""

        model = self.ensure_model()

        client = self._client()

        request_kwargs = dict(kwargs)
        request_kwargs.pop(
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
            **request_kwargs,
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