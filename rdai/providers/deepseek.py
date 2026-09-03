from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class DeepseekProvider(BaseProvider):
    """DeepSeek provider adapter."""

    traits = (
        "coding",
        "logic",
        "fast",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)

    def fallback_models(self) -> tuple[str, ...]:
        """Return a safe fallback model for DeepSeek."""
        return ("deepseek-v4-flash",)

    def available_models(self) -> tuple[str, ...]:
        """Discover models currently exposed by DeepSeek."""

        if not self.is_available:
            return ()

        try:
            response = requests.get(
                "https://api.deepseek.com/models",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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

                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            # Discovery is best-effort. A failed discovery must not
            # break normal provider usage.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using DeepSeek chat completions."""

        if not self.is_available:
            raise ValueError("DeepSeek API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No DeepSeek model is configured or available."
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
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "DeepSeek returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("DeepSeek returned an empty response.")

        return str(content)