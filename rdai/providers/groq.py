from __future__ import annotations

from typing import Any, Optional

from rdai.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq provider adapter."""

    traits = (
        "ultra-fast",
        "groq-lpu",
        "low-latency",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)

    def fallback_models(self) -> tuple[str, ...]:
        """Return a safe fallback model for Groq."""
        return ("llama-3.1-8b-instant",)

    def available_models(self) -> tuple[str, ...]:
        """Discover active models currently exposed by Groq."""

        if not self.is_available:
            return ()

        try:
            from groq import Groq
        except ImportError:
            return ()

        try:
            client = Groq(api_key=self.api_key)
            response = client.models.list()

            models: list[str] = []

            for model in response.data:
                model_id = getattr(model, "id", None)
                is_active = getattr(model, "active", True)

                if (
                    isinstance(model_id, str)
                    and model_id.strip()
                    and is_active
                ):
                    models.append(model_id.strip())

            return tuple(models)

        except Exception:
            # Discovery is best-effort. A failed discovery must not make
            # the provider unusable.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using Groq chat completions."""

        if not self.is_available:
            raise ValueError("Groq API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Groq model is configured or available."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "Please install the Groq SDK using: pip install groq"
            ) from exc

        client = Groq(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError("Groq returned an empty response.")

        return content