from __future__ import annotations

from typing import Any, Optional

from google import genai

from rdai.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the google-genai SDK."""

    traits = (
        "coding",
        "reasoning",
        "multimodal",
        "fast",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)

        self.client = (
            genai.Client(api_key=self.api_key)
            if self.api_key
            else None
        )

    @property
    def is_available(self) -> bool:
        """Return True when Gemini has a usable API client."""
        return self.client is not None

    def fallback_models(self) -> tuple[str, ...]:
        """Return stable Gemini fallback candidates."""
        return ("gemini-3.8-flash",)

    def available_models(self) -> tuple[str, ...]:
        """Discover Gemini models supporting content generation."""

        if not self.is_available:
            return ()

        try:
            models = self.client.models.list()

            discovered: list[str] = []

            for model in models:
                model_name = getattr(model, "name", None)

                if not isinstance(model_name, str):
                    continue

                model_name = model_name.strip()

                if not model_name:
                    continue

                # The Gemini API may return names such as "models/foo".
                if model_name.startswith("models/"):
                    model_name = model_name[len("models/"):]

                supported_actions = getattr(
                    model,
                    "supported_actions",
                    (),
                )

                if supported_actions:
                    if "generateContent" not in supported_actions:
                        continue

                discovered.append(model_name)

            return tuple(discovered)

        except Exception:
            # Discovery is best-effort. Normal fallback resolution must
            # continue to work when model listing is unavailable.
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through the Gemini API."""

        if not self.is_available:
            raise ValueError("Gemini API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No Gemini model is configured or available."
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            **kwargs,
        )

        content = getattr(response, "text", None)

        if content is None:
            raise RuntimeError("Gemini returned an empty response.")

        return content