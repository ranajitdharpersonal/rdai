from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class LlamaProvider(BaseProvider):
    """Llama provider through the Together AI OpenAI-compatible endpoint."""

    traits = (
        "open-source",
        "meta",
        "fast",
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

        self.endpoint = (
            "https://api.together.xyz/v1/chat/completions"
        )

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Together AI Llama model."""
        return ("meta-llama/Llama-3-70b-chat-hf",)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through Together AI."""

        if not self.is_available:
            raise ValueError("Llama API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Llama model is configured or available."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        timeout = kwargs.pop("timeout", 15.0)

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Llama returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("Llama returned an empty response.")

        return str(content)