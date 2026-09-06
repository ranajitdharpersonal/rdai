from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from rdai.providers.base import BaseProvider


class DeepseekProvider(BaseProvider):
    """DeepSeek provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Discovered models are filtered for compatibility with the
    standard chat-completions path used by this adapter.
    """

    traits = (
        "coding",
        "logic",
        "fast",
    )

    models_endpoint = "https://api.deepseek.com/models"
    chat_endpoint = "https://api.deepseek.com/chat/completions"

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
        """Discover currently available DeepSeek models."""

        if not self.is_available:
            return ()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(
                self.models_endpoint,
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                return ()

            data = payload.get("data", [])

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
        """Keep models compatible with text chat generation.

        Vision-capable models may still accept text, so they are not excluded
        solely because their names contain "vision".
        """

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "moderation",
            "reranker",
            "whisper",
            "tts",
            "text-to-speech",
            "audio",
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
        """Preserve DeepSeek's discovery order.

        RDAI does not hardcode a preferred DeepSeek model.
        """

        return tuple(models)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        """Build authenticated DeepSeek API headers."""

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request_timeout(
        self,
        kwargs: dict[str, Any],
    ) -> float:
        """Extract an optional request timeout without leaking it upstream."""

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
        """Generate a complete response using DeepSeek chat completions."""

        if not self.is_available:
            raise ValueError(
                "DeepSeek API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)
        timeout = self._request_timeout(
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
            headers=self._headers(self.api_key),
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
                "DeepSeek returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "DeepSeek returned an empty response."
            )

        return str(content)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text chunks from DeepSeek."""

        if not self.is_available:
            raise ValueError(
                "DeepSeek API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)
        timeout = self._request_timeout(
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
            headers=self._headers(self.api_key),
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

            data = line[5:].strip()

            if data == "[DONE]":
                break

            try:
                chunk = requests.models.complexjson.loads(data)
            except (
                ValueError,
                TypeError,
            ):
                continue

            try:
                content = chunk["choices"][0]["delta"].get(
                    "content"
                )
            except (
                KeyError,
                IndexError,
                TypeError,
            ):
                continue

            if content:
                yield str(content)