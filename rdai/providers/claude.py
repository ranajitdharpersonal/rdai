from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider adapter."""

    traits = (
        "creative",
        "analytical",
        "reasoning",
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

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Claude fallback model."""
        return ("claude-3-haiku-20240307",)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the Anthropic Messages API."""

        if not self.is_available:
            raise ValueError("Claude API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Claude model is configured or available."
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        max_tokens = kwargs.pop("max_tokens", 1024)
        timeout = kwargs.pop("timeout", 15.0)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        try:
            data = response.json()
            content = data["content"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Claude returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("Claude returned an empty response.")

        return str(content)