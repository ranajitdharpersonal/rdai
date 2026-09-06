from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class QwenProvider(BaseProvider):
    """Alibaba Cloud Qwen provider through Model Studio.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Streaming uses the native DashScope SSE interface.
    """

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

            output = payload.get(
                "output",
                {},
            )

            if not isinstance(output, dict):
                return ()

            data = output.get(
                "models",
                [],
            )

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

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                ):
                    continue

                models.append(
                    model_id.strip()
                )

            return tuple(models)

        except Exception:
            return ()

    def filter_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Keep models suitable for text generation."""

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "rerank",
            "moderation",
            "tts",
            "text-to-speech",
            "asr",
            "speech",
        )

        for model in models:
            normalized = model.strip()

            if not normalized:
                continue

            lowered = normalized.lower()

            if any(
                marker in lowered
                for marker in excluded_markers
            ):
                continue

            filtered.append(normalized)

        return tuple(filtered)

    def rank_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Preserve Model Studio discovery order."""

        return tuple(models)

    def _headers(
        self,
        *,
        stream: bool = False,
    ) -> dict[str, str]:
        """Build authenticated DashScope headers."""

        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

        if stream:
            headers["X-DashScope-SSE"] = "enable"

        return headers

    @staticmethod
    def _timeout(
        kwargs: dict[str, Any],
    ) -> float:
        """Extract and validate the HTTP timeout."""

        timeout = kwargs.pop(
            "timeout",
            15.0,
        )

        if not isinstance(timeout, (int, float)):
            raise TypeError(
                "timeout must be an int or float."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        return float(timeout)

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response through DashScope."""

        if not self.is_available:
            raise ValueError(
                "Qwen API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        timeout = self._timeout(
            request_kwargs,
        )

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

        payload.update(request_kwargs)

        response = requests.post(
            self.chat_endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        try:
            data = response.json()
            output = data["output"]

            if isinstance(output, dict):
                content = output.get("text")

                if content is None:
                    choices = output.get(
                        "choices",
                        [],
                    )

                    if choices:
                        first = choices[0]

                        if isinstance(first, dict):
                            message = first.get(
                                "message",
                                {},
                            )

                            if isinstance(message, dict):
                                content = message.get(
                                    "content"
                                )
            else:
                content = None

        except (
            ValueError,
            KeyError,
            IndexError,
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

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream incremental text from DashScope."""

        if not self.is_available:
            raise ValueError(
                "Qwen API key is missing."
            )

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        timeout = self._timeout(
            request_kwargs,
        )

        request_kwargs.pop(
            "stream",
            None,
        )

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
            "parameters": {
                "incremental_output": True,
            },
        }

        extra_parameters = request_kwargs.pop(
            "parameters",
            None,
        )

        if isinstance(extra_parameters, dict):
            payload["parameters"].update(
                extra_parameters
            )

        payload.update(request_kwargs)

        response = requests.post(
            self.chat_endpoint,
            headers=self._headers(
                stream=True,
            ),
            json=payload,
            timeout=timeout,
            stream=True,
        )

        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True,
        ):
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode(
                    "utf-8",
                    errors="replace",
                )

            if line.startswith("data:"):
                raw_data = line[5:].strip()
            else:
                raw_data = line.strip()

            if not raw_data:
                continue

            if raw_data == "[DONE]":
                break

            try:
                event = requests.models.complexjson.loads(
                    raw_data
                )
            except (
                ValueError,
                TypeError,
            ):
                continue

            if not isinstance(event, dict):
                continue

            output = event.get(
                "output",
            )

            if not isinstance(output, dict):
                continue

            content = output.get(
                "text",
            )

            if content:
                yield str(content)
                continue

            choices = output.get(
                "choices",
                [],
            )

            if not isinstance(choices, list):
                continue

            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                delta = choice.get(
                    "delta",
                    {},
                )

                if not isinstance(delta, dict):
                    continue

                text = delta.get(
                    "content"
                )

                if text:
                    yield str(text)