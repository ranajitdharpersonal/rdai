from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

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
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        # ``api_key`` is retained for compatibility with the provider
        # interface. For Vertex AI it represents the configured GCP project ID.
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
    def project_id(self) -> str | None:
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
        """Discover Vertex AI models supporting text generation."""

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

                if not isinstance(
                    model_name,
                    str,
                ):
                    continue

                model_name = model_name.strip()

                if not model_name:
                    continue

                supported_actions = getattr(
                    model,
                    "supported_actions",
                    None,
                )

                if supported_actions is not None:
                    actions = {
                        str(action).strip().lower()
                        for action in supported_actions
                        if str(action).strip()
                    }

                    if actions and (
                        "generatecontent" not in actions
                    ):
                        continue

                if model_name.startswith(
                    "models/"
                ):
                    model_name = model_name[len("models/"):]

                discovered.append(
                    model_name
                )

            return tuple(discovered)

        except Exception:
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Keep models suitable for standard text generation."""

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "image-generation",
            "image_generation",
            "tts",
            "text-to-speech",
            "lyria",
            "robotics",
            "computer-use",
            "computer_use",
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

        return filtered

    def rank_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Preserve Vertex AI discovery order."""

        return models

    def _client(self) -> Any:
        """Return the configured Vertex AI client."""

        if not self.is_available or self.client is None:
            raise ValueError(
                "GCP Project ID is missing for VertexAI."
            )

        return self.client

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response through Vertex AI."""

        model = self.ensure_model()

        response = self._client().models.generate_content(
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

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks from Vertex AI."""

        model = self.ensure_model()

        stream = self._client().models.generate_content_stream(
            model=model,
            contents=prompt,
            **kwargs,
        )

        for chunk in stream:
            content = getattr(
                chunk,
                "text",
                None,
            )

            if content:
                yield str(content)