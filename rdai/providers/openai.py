from __future__ import annotations

from typing import Any, Optional

from rdai.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter.

    Model selection is fully discovery-based unless the user explicitly
    supplies a model.
    """

    traits = (
        "coding",
        "reasoning",
        "industry-standard",
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
        """Discover models currently exposed by the OpenAI account."""

        if not self.is_available:
            return ()

        try:
            from openai import OpenAI
        except ImportError:
            return ()

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.models.list()

            models: list[str] = []

            for model in response.data:
                model_id = getattr(model, "id", None)

                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using OpenAI chat completions."""

        if not self.is_available:
            raise ValueError("OpenAI API key is missing.")

        model = self.ensure_model()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Please install the OpenAI SDK using: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenAI returned an unexpected response format."
            ) from exc

        if content is None:
            raise RuntimeError("OpenAI returned an empty response.")

        return str(content)