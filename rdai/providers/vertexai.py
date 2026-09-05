from __future__ import annotations

from typing import Any, Optional

from google import genai

from rdai.providers.base import BaseProvider


class VertexaiProvider(BaseProvider):
    """Google Vertex AI provider using the google-genai SDK."""

    traits = (
        "enterprise",
        "secure",
        "multimodal",
    )

    default_location = "us-central1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # ``api_key`` is retained for compatibility with the provider
        # interface. For Vertex AI it represents the GCP project ID.
        super().__init__(
            api_key=api_key,
            model=model,
        )

        self.client = (
            genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.default_location,
            )
            if self.project_id
            else None
        )

    @property
    def project_id(self) -> Optional[str]:
        """Return the configured GCP project ID."""
        if not self.api_key:
            return None

        project_id = self.api_key.strip()

        return project_id or None

    @property
    def is_available(self) -> bool:
        """Return True when a GCP project is configured."""
        return self.project_id is not None

    def available_models(self) -> tuple[str, ...]:
        """Discover Vertex AI publisher models supporting generation."""

        if not self.is_available or self.client is None:
            return ()

        try:
            models = self.client.models.list()

            discovered: list[str] = []

            for model in models:
                model_name = getattr(
                    model,
                    "name",
                    None,
                )

                if not isinstance(model_name, str):
                    continue

                model_name = model_name.strip()

                if not model_name:
                    continue

                supported_actions = getattr(
                    model,
                    "supported_actions",
                    (),
                )

                if supported_actions:
                    if "generateContent" not in supported_actions:
                        continue

                if model_name.startswith("models/"):
                    model_name = model_name[len("models/"):]

                discovered.append(model_name)

            return tuple(discovered)

        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through Vertex AI."""

        if not self.is_available or self.client is None:
            raise ValueError(
                "GCP Project ID is missing for VertexAI."
            )

        model = self.ensure_model()

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            **kwargs,
        )

        content = getattr(
            response,
            "text",
            None,
        )

        if content is None:
            raise RuntimeError(
                "VertexAI returned an empty response."
            )

        return str(content)