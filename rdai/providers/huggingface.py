from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class HuggingfaceProvider(BaseProvider):
    """Hugging Face provider adapter."""

    traits = (
        "open-source",
        "flexible",
        "community",
    )

    models_endpoint = "https://huggingface.co/api/models"
    chat_endpoint = "https://router.huggingface.co/v1/chat/completions"

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
        """Discover text-generation models available through HF inference."""

        if not self.is_available:
            return ()

        try:
            response = requests.get(
                self.models_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                params={
                    "inference_provider": "hf-inference",
                    "pipeline_tag": "text-generation",
                    "limit": 100,
                },
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

                model_id = item.get("id")

                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through Hugging Face Inference Providers."""

        if not self.is_available:
            raise ValueError("HuggingFace API key is missing.")

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
                "HuggingFace returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "HuggingFace returned an empty response."
            )

        return str(content)