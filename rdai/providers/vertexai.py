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
        # ``api_key`` is retained for compatibility with the common provider
        # interface. For Vertex AI it represents the configured GCP project ID.
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Vertex AI Gemini model."""
        return ("gemini-2.5-flash",)

    @property
    def project_id(self) -> Optional[str]:
        """Return the configured GCP project ID."""
        if not self.api_key:
            return None

        project_id = self.api_key.strip()
        return project_id or None

    @property
    def is_available(self) -> bool:
        """Return True when a GCP project is configured.

        Actual Google Cloud credentials are resolved by the google-genai SDK
        when a request is made.
        """
        return self.project_id is not None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through Vertex AI."""

        if not self.is_available:
            raise ValueError(
                "GCP Project ID is missing for VertexAI."
            )

        if not self.model:
            raise RuntimeError(
                "No VertexAI model is configured or available."
            )

        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=kwargs.pop("location", self.default_location),
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            **kwargs,
        )

        content = getattr(response, "text", None)

        if content is None:
            raise RuntimeError(
                "VertexAI returned an empty response."
            )

        return content