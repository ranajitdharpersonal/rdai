from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class MistralProvider(BaseProvider):
    """Mistral provider adapter."""

    traits = (
        "european",
        "efficient",
        "open-weights",
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
        """Return the default Mistral fallback model."""
        return ("mistral-large-latest",)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the Mistral API."""

        if not self.is_available:
            raise ValueError("Mistral API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Mistral model is configured or available."
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
            "https://api.mistral.ai/v1/chat/completions",
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
                "Mistral returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("Mistral returned an empty response.")

        return str(content)