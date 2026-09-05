from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider adapter.

    Model selection is discovery-based unless the user explicitly
    supplies a model.
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
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        """Discover models currently available to the Anthropic account."""

        if not self.is_available:
            return ()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }

        models: list[str] = []
        after_id: Optional[str] = None

        try:
            while True:
                params: dict[str, Any] = {
                    "limit": 1000,
                }

                if after_id:
                    params["after_id"] = after_id

                response = requests.get(
                    self.models_endpoint,
                    headers=headers,
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
                        models.append(model_id.strip())

                if not payload.get("has_more"):
                    break

                next_after_id = payload.get("last_id")

                if not isinstance(next_after_id, str):
                    break

                if not next_after_id.strip():
                    break

                after_id = next_after_id.strip()

            return tuple(models)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the Anthropic Messages API."""

        if not self.is_available:
            raise ValueError("Claude API key is missing.")

        model = self.ensure_model()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

        max_tokens = kwargs.pop("max_tokens", 1024)
        timeout = kwargs.pop("timeout", 15.0)

        payload = {
            "model": model,
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
            self.messages_endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        try:
            data = response.json()
            content = data["content"][0]["text"]
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Claude returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Claude returned an empty response."
            )

        return str(content)