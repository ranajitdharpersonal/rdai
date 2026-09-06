from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from rdai.providers.base import BaseProvider


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider adapter.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Streaming uses the Anthropic Messages SSE interface.
    """

    traits = (
        "creative",
        "analytical",
        "reasoning",
    )

    api_version = "2023-06-01"
    models_endpoint = "https://api.anthropic.com/v1/models"
    messages_endpoint = "https://api.anthropic.com/v1/messages"

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
        """Build authenticated Anthropic headers."""

        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def available_models(self) -> tuple[str, ...]:
        """Discover models currently available to the Anthropic account."""

        if not self.is_available:
            return ()

        models: list[str] = []
        after_id: str | None = None

        try:
            while True:
                params: dict[str, Any] = {
                    "limit": 1000,
                }

                if after_id:
                    params["after_id"] = after_id

                response = requests.get(
                    self.models_endpoint,
                    headers=self._headers(),
                    params=params,
                    timeout=15.0,
                )

                response.raise_for_status()

                payload = response.json()

                if not isinstance(payload, dict):
                    return tuple(models)

                data = payload.get("data", [])

                if not isinstance(data, list):
                    return tuple(models)

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    model_id = item.get("id")

                    if (
                        isinstance(model_id, str)
                        and model_id.strip()
                    ):
                        models.append(
                            model_id.strip()
                        )

                if not payload.get("has_more"):
                    break

                next_after_id = payload.get("last_id")

                if not isinstance(next_after_id, str):
                    break

                next_after_id = next_after_id.strip()

                if not next_after_id:
                    break

                if next_after_id == after_id:
                    break

                after_id = next_after_id

            return tuple(models)

        except Exception:
            return ()

    def filter_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Keep models compatible with the Anthropic Messages API."""

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "moderation",
            "whisper",
            "transcribe",
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
        """Preserve Anthropic discovery order."""

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
        """Generate a complete response through the Anthropic Messages API."""

        if not self.is_available:
            raise ValueError(
                "Claude API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        max_tokens = request_kwargs.pop(
            "max_tokens",
            1024,
        )

        timeout = self._timeout(
            request_kwargs,
        )

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        payload.update(request_kwargs)

        response = requests.post(
            self.messages_endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        try:
            data = response.json()
            blocks = data["content"]

            if not blocks:
                raise RuntimeError(
                    "Claude returned an empty response."
                )

            text_parts = [
                block.get("text", "")
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") == "text"
            ]

            content = "".join(text_parts)

        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ) as exc:
            raise RuntimeError(
                "Claude returned an unexpected response format."
            ) from exc

        if not content:
            raise RuntimeError(
                "Claude returned an empty response."
            )

        return content

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text deltas from the Anthropic Messages API."""

        if not self.is_available:
            raise ValueError(
                "Claude API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        max_tokens = request_kwargs.pop(
            "max_tokens",
            1024,
        )

        timeout = self._timeout(
            request_kwargs,
        )

        request_kwargs.pop(
            "stream",
            None,
        )

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        payload.update(request_kwargs)

        response = requests.post(
            self.messages_endpoint,
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

            try:
                event = requests.models.complexjson.loads(
                    raw_data
                )
            except (
                ValueError,
                TypeError,
            ):
                continue

            if not isinstance(event, dict):
                continue

            if event.get("type") != "content_block_delta":
                continue

            delta = event.get("delta")

            if not isinstance(delta, dict):
                continue

            text = delta.get("text")

            if text:
                yield str(text)