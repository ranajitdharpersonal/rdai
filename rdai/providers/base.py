from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from rdai.providers.model_resolver import normalize_model, resolve_model


class BaseProvider(ABC):
    """Abstract base class for all rdai providers.

    Model-resolution contract:

    1. An explicitly supplied model always wins.
    2. Without an explicit model, available models are discovered lazily.
    3. Provider-specific filtering/ranking may refine discovered candidates.
    4. A failed discovered model can be excluded during refresh.
    5. No hardcoded provider model is ever used as a fallback.

    Streaming contract:

    Providers may override ``stream()`` for native token streaming.
    The default implementation yields the complete ``generate()`` result
    as one chunk, keeping custom providers backward-compatible.
    """

    traits: Sequence[str] = ()

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self._requested_model = normalize_model(model)
        self._model: str | None = self._requested_model
        self._discovered_models: tuple[str, ...] | None = None
        self._excluded_models: set[str] = set()

    @property
    def is_available(self) -> bool:
        """Return True when the provider has the credentials it needs."""

        return bool(self.api_key)

    @property
    def model(self) -> str | None:
        """Return the active model, discovering one lazily when necessary."""

        requested_model = getattr(
            self,
            "_requested_model",
            None,
        )

        if requested_model is not None:
            return requested_model

        current_model = getattr(
            self,
            "_model",
            None,
        )

        if current_model is not None:
            return current_model

        return self._resolve_discovered_model()

    @model.setter
    def model(self, value: str | None) -> None:
        """Set the runtime model while preserving explicit user intent."""

        normalized = normalize_model(value)

        requested_model = getattr(
            self,
            "_requested_model",
            None,
        )

        if requested_model is not None:
            self._model = requested_model
            return

        self._model = normalized

    @property
    def requested_model(self) -> str | None:
        """Return the model explicitly supplied by the user, if any."""

        return getattr(
            self,
            "_requested_model",
            None,
        )

    def available_models(self) -> Iterable[str]:
        """Discover model identifiers currently exposed by the provider."""

        return ()

    def filter_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Filter discovered models to those usable by this provider."""

        return models

    def rank_models(
        self,
        models: Iterable[str],
    ) -> Iterable[str]:
        """Order discovered candidates before model selection."""

        return models

    def discover_models(
        self,
        force: bool = False,
    ) -> tuple[str, ...]:
        """Discover and cache normalized, filtered model candidates."""

        if not self.is_available:
            return ()

        cached_models = getattr(
            self,
            "_discovered_models",
            None,
        )

        if cached_models is not None and not force:
            return cached_models

        try:
            discovered = self.available_models()

            normalized_models: list[str] = []

            for candidate in discovered:
                normalized = normalize_model(candidate)

                if normalized is not None:
                    normalized_models.append(normalized)

            filtered_models = self.filter_models(
                normalized_models,
            )

            ranked_models = self.rank_models(
                filtered_models,
            )

            final_models: list[str] = []

            for candidate in ranked_models:
                normalized = normalize_model(candidate)

                if normalized is None:
                    continue

                if normalized in final_models:
                    continue

                if normalized in self._excluded_models:
                    continue

                final_models.append(normalized)

            self._discovered_models = tuple(final_models)

        except Exception:
            self._discovered_models = ()

        return self._discovered_models

    def resolve_model(
        self,
        requested_model: str | None = None,
        *,
        discover: bool = True,
    ) -> str | None:
        """Resolve the active model from explicit intent or discovery."""

        explicit = (
            normalize_model(requested_model)
            if requested_model is not None
            else self.requested_model
        )

        if explicit is not None:
            return explicit

        available_models: Iterable[str] = ()

        if discover:
            available_models = self.discover_models()

        return resolve_model(
            requested_model=None,
            available_models=available_models,
        )

    def _resolve_discovered_model(self) -> str | None:
        """Select and cache the first valid discovered candidate."""

        resolved = self.resolve_model(
            discover=True,
        )

        if resolved is not None:
            self._model = resolved

        return resolved

    def ensure_model(self) -> str:
        """Return a usable model or raise a clear configuration error."""

        model = self.model

        if model is None:
            provider_name = getattr(
                self,
                "name",
                self.__class__.__name__.replace(
                    "Provider",
                    "",
                ).lower(),
            )

            raise RuntimeError(
                f"No model is available for {provider_name}. "
                "Specify a model explicitly or ensure the provider "
                "can discover an available generation model."
            )

        return model

    def refresh_model(
        self,
        *,
        failed_model: str | None = None,
    ) -> str | None:
        """Refresh discovered models and select the next viable candidate."""

        if self.requested_model is not None:
            self._model = self.requested_model
            return self.requested_model

        candidate_to_exclude = normalize_model(
            failed_model,
        )

        if candidate_to_exclude is None:
            candidate_to_exclude = normalize_model(
                getattr(
                    self,
                    "_model",
                    None,
                )
            )

        if candidate_to_exclude is not None:
            self._excluded_models.add(
                candidate_to_exclude,
            )

        self._discovered_models = None
        self._model = None

        return self._resolve_discovered_model()

    def reset_model_exclusions(self) -> None:
        """Clear previously failed discovered-model exclusions."""

        self._excluded_models.clear()
        self._discovered_models = None

        if self.requested_model is None:
            self._model = None
        else:
            self._model = self.requested_model

    @property
    def excluded_models(self) -> tuple[str, ...]:
        """Return discovered models excluded during this provider lifetime."""

        return tuple(self._excluded_models)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response.

        Custom providers that do not implement native streaming remain
        compatible: the complete ``generate()`` response is yielded once.
        """

        yield self.generate(
            prompt,
            **kwargs,
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text through the provider."""

        pass