from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from google import genai

from rdai.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the google-genai SDK.

    Model selection is discovery-based unless the user explicitly supplies
    a model. Discovered models are filtered using provider metadata when
    available, rather than hardcoded model IDs.
    """

    traits = (
        "coding",
        "reasoning",
        "multimodal",
        "fast",
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

        self.client = (
            genai.Client(
                api_key=self.api_key,
            )
            if self.api_key
            else None
        )

    @property
    def is_available(self) -> bool:
        """Return True when the Gemini client is configured."""

        return self.client is not None

    def available_models(self) -> tuple[str, ...]:
        """Discover models that support text generation."""

        if not self.is_available:
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

                if model_name.startswith("models/"):
                    model_name = model_name[len("models/"):]

                supported_actions = getattr(
                    model,
                    "supported_actions",
                    None,
                )

                # Gemini exposes supported_actions for model capabilities.
                # Only require the capability check when metadata exists;
                # this keeps the adapter compatible with SDK objects that
                # may omit the field.
                if supported_actions is not None:
                    actions = {
                        str(action).strip().lower()
                        for action in supported_actions
                        if str(action).strip()
                    }

                    if "generatecontent" not in actions:
                        continue

                discovered.append(model_name)

            return tuple(discovered)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Remove clearly specialized non-text model families."""

        filtered: list[str] = []

        excluded_markers = (
            "embedding",
            "image-generation",
            "image_generation",
            "tts",
            "text-to-speech",
            "text_to_speech",
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
        """Preserve provider discovery order.

        Model capability metadata determines eligibility. RDAI deliberately
        does not promote a hardcoded Gemini model family as a default.
        """

        return models

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response using Gemini."""

        if not self.is_available:
            raise ValueError(
                "Gemini API key is missing."
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
                "Gemini returned an empty response."
            )

        return str(content)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks from Gemini."""

        if not self.is_available:
            raise ValueError(
                "Gemini API key is missing."
            )

        model = self.ensure_model()

        stream = self.client.models.generate_content_stream(
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