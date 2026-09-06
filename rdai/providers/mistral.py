from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from rdai.providers.base import BaseProvider


class MistralProvider(BaseProvider):
    """Mistral provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Streaming uses the Mistral chat-completions SSE interface.
    """

    traits = (
        "european",
        "efficient",
        "open-weights",
    )

    models_endpoint = "https://api.mistral.ai/v1/models"
    chat_endpoint = "https://api.mistral.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def _headers(self) -> dict[str, str]:
        """Build authenticated Mistral headers."""

        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

    def available_models(self) -> tuple[str, ...]:
        """Discover models that support chat completion."""

        if not self.is_available:
            return ()

        try:
            response = requests.get(
                self.models_endpoint,
                headers=self._headers(),
                timeout=15.0,
            )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                return ()

            data = payload.get(
                "data",
                [],
            )

            if not isinstance(data, list):
                return ()

            models: list[str] = []

            for item in data:
                if not isinstance(item, dict):
                    continue

                model_id = item.get("id")

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                ):
                    continue

                if item.get("archived") is True:
                    continue

                capabilities = item.get(
                    "capabilities",
                    {},
                )

                if isinstance(
                    capabilities,
                    dict,
                ):
                    chat_capability = capabilities.get(
                        "completion_chat"
                    )

                    if chat_capability is False:
                        continue

                models.append(
                    model_id.strip()
                )

            return tuple(models)

        except Exception:
            return ()

    def filter_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Keep discovered models suitable for text chat."""

        filtered: list[str] = []

        excluded_markers = (
            "embed",
            "moderation",
            "ocr",
            "transcription",
            "tts",
            "text-to-speech",
        )

        for model in models:
            normalized = model.strip()

            if not normalized:
                continue

            lowered = normalized.lower()

            if any(
                marker in lowered
                for marker in excluded_markers
            ):
                continue

            filtered.append(normalized)

        return tuple(filtered)

    def rank_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Preserve Mistral's discovery order."""

        return tuple(models)

    @staticmethod
    def _timeout(
        kwargs: dict[str, Any],
    ) -> float:
        """Extract and validate the HTTP timeout."""

        timeout = kwargs.pop(
            "timeout",
            15.0,
        )

        if not isinstance(timeout, (int, float)):
            raise TypeError(
                "timeout must be an int or float."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        return float(timeout)

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response through Mistral."""

        if not self.is_available:
            raise ValueError(
                "Mistral API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        timeout = self._timeout(
            request_kwargs,
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        payload.update(request_kwargs)

        response = requests.post(
            self.chat_endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Mistral returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Mistral returned an empty response."
            )

        return str(content)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text deltas from Mistral."""

        if not self.is_available:
            raise ValueError(
                "Mistral API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        timeout = self._timeout(
            request_kwargs,
        )

        request_kwargs.pop(
            "stream",
            None,
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": True,
        }

        payload.update(request_kwargs)

        response = requests.post(
            self.chat_endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
            stream=True,
        )

        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True,
        ):
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode(
                    "utf-8",
                    errors="replace",
                )

            if not line.startswith("data:"):
                continue

            raw_data = line[5:].strip()

            if not raw_data:
                continue

            if raw_data == "[DONE]":
                break

            try:
                chunk = requests.models.complexjson.loads(
                    raw_data
                )
            except (
                ValueError,
                TypeError,
            ):
                continue

            if not isinstance(chunk, dict):
                continue

            choices = chunk.get(
                "choices",
                [],
            )

            if not isinstance(choices, list):
                continue

            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                delta = choice.get(
                    "delta",
                    {},
                )

                if not isinstance(delta, dict):
                    continue

                content = delta.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ) and content:
                    yield content