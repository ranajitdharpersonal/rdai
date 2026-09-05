from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class QwenProvider(BaseProvider):
    """Alibaba Cloud Qwen provider through Model Studio."""

    traits = (
        "multilingual",
        "efficient",
    )

    models_endpoint = (
        "https://dashscope.aliyuncs.com/"
        "api/v1/models"
    )

    chat_endpoint = (
        "https://dashscope.aliyuncs.com/"
        "api/v1/services/aigc/text-generation/generation"
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

    def available_models(self) -> tuple[str, ...]:
        """Discover text-generation models available to the account."""

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
                params={
                    "capabilities": "TG",
                    "page_no": 1,
                    "page_size": 100,
                },
                timeout=15.0,
            )
            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                return ()

            output = payload.get("output", {})

            if not isinstance(output, dict):
                return ()

            data = output.get("models", [])

            if not isinstance(data, list):
                return ()

            models: list[str] = []

            for item in data:
                if not isinstance(item, dict):
                    continue

                model_id = (
                    item.get("model")
                    or item.get("model_id")
                    or item.get("id")
                )

                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the DashScope API."""

        if not self.is_available:
            raise ValueError("Qwen API key is missing.")

        model = self.ensure_model()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
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
            content = data["output"]["text"]
        except (
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Qwen returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return str(content)