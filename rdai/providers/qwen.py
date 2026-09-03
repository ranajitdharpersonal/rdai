from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class QwenProvider(BaseProvider):
    """Alibaba Cloud Qwen provider through the DashScope API."""

    traits = (
        "multilingual",
        "efficient",
    )

    endpoint = (
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

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Qwen fallback model."""
        return ("qwen-turbo",)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the DashScope API."""

        if not self.is_available:
            raise ValueError("Qwen API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Qwen model is configured or available."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
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

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        try:
            data = response.json()
            content = data["output"]["text"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Qwen returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("Qwen returned an empty response.")

        return str(content)