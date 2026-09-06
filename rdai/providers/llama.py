from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class LlamaProvider(BaseProvider):
    """Llama models through the Together AI API.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Only discovered Llama-family chat-capable models are used.
    """

    traits = (
        "open-source",
        "meta",
        "fast",
    )

    models_endpoint = "https://api.together.ai/v1/models"
    chat_endpoint = "https://api.together.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def _headers(self) -> dict[str, str]:
        """Build authenticated Together AI headers."""

        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

    def available_models(self) -> tuple[str, ...]:
        """Discover chat-capable Llama-family models from Together."""

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

            if not isinstance(payload, list):
                return ()

            models: list[str] = []

            for item in payload:
                if not isinstance(item, dict):
                    continue

                model_id = item.get(
                    "id"
                )

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                ):
                    continue

                model_type = item.get(
                    "type"
                )

                if model_type not in (
                    None,
                    "chat",
                    "language",
                ):
                    continue

                lowered = model_id.lower()

                if "llama" not in lowered:
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
        """Keep Llama models suitable for text chat."""

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "moderation",
            "guard",
            "whisper",
            "audio",
            "tts",
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
        """Preserve Together's model discovery order."""

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
        """Generate a complete response through Together AI."""

        if not self.is_available:
            raise ValueError(
                "Llama API key is missing."
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
                "Llama returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Llama returned an empty response."
            )

        return str(content)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text deltas from Together AI."""

        if not self.is_available:
            raise ValueError(
                "Llama API key is missing."
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