from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from google import genai

from rdai.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini provider using the google-genai SDK.

    Model selection is discovery-based unless the user explicitly supplies
    a model.
    """

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
        super().__init__(
            api_key=api_key,
            model=model,
        )

        self.client = (
            genai.Client(api_key=self.api_key)
            if self.api_key
            else None
        )

    @property
    def is_available(self) -> bool:
        """Return True when the Gemini client is configured."""
        return self.client is not None

    def available_models(self) -> tuple[str, ...]:
        """Discover Gemini models exposed by the configured API key."""

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

                discovered.append(model_name)

            return tuple(discovered)

        except Exception:
            # Discovery is best-effort. Never invent a model.
            return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Keep Gemini models suitable for standard text generation."""

        filtered: list[str] = []

        for model in models:
            normalized = model.strip()

            if not normalized:
                continue

            lowered = normalized.lower()

            # The current RDAI adapter uses generateContent, so models that
            # are clearly specialized for embeddings, image generation,
            # robotics, TTS, or other non-text tasks should not be selected.
            excluded_markers = (
                "embedding",
                "image",
                "tts",
                "lyria",
                "robotics",
                "computer-use",
                "computer_use",
                "deep-research",
                "deep_research",
                "antigravity",
            )

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
        """Prefer stable, general-purpose Flash models."""

        candidates = list(models)

        preferred_markers = (
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        )

        def score(model: str) -> tuple[int, int]:
            lowered = model.lower()

            for index, marker in enumerate(
                preferred_markers
            ):
                if marker in lowered:
                    return (
                        index,
                        candidates.index(model),
                    )

            # Unknown models remain usable candidates but are ranked after
            # known general-purpose Gemini families.
            return (
                len(preferred_markers),
                candidates.index(model),
            )

        return sorted(
            candidates,
            key=score,
        )

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Gemini."""

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