from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class DeepseekProvider(BaseProvider):
    """DeepSeek provider adapter.

    Model selection is discovery-based unless the user explicitly
    supplies a model.
    """

    traits = (
        "coding",
        "logic",
        "fast",
    )

    models_endpoint = "https://api.deepseek.com/models"
    chat_endpoint = "https://api.deepseek.com/v1/chat/completions"

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

                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using DeepSeek chat completions."""

        if not self.is_available:
            raise ValueError("DeepSeek API key is missing.")

        model = self.ensure_model()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        timeout = kwargs.pop("timeout", 15.0)

        payload.update(kwargs)

        response = requests.post(
            self.chat_endpoint,
            headers=headers,
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