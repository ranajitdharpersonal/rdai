from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class MistralProvider(BaseProvider):
    """Mistral provider adapter.

    Model selection is discovery-based unless the user explicitly
    supplies a model.
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
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        """Discover models that support chat completion."""

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

                if not isinstance(model_id, str):
                    continue

                model_id = model_id.strip()

                if not model_id:
                    continue

                if item.get("archived") is True:
                    continue

                capabilities = item.get("capabilities", {})

                if isinstance(capabilities, dict):
                    if capabilities.get("completion_chat") is False:
                        continue

                models.append(model_id)

            return tuple(models)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the Mistral API."""

        if not self.is_available:
            raise ValueError("Mistral API key is missing.")

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
                "Mistral returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Mistral returned an empty response."
            )

        return str(content)