from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from rdai.providers.model_resolver import resolve_model


class BaseProvider(ABC):
    """Abstract base class for all AI providers in rdai.

    Model resolution follows a safe, lazy strategy:

    1. An explicitly supplied model always wins.
    2. Static provider fallback models are used during initialization.
    3. Runtime model discovery is opt-in and never happens during provider
       construction.
    4. Discovered models are cached to avoid repeated network requests.
    """

    # Traits help the Smart Router choose the best provider/model.
    traits: Sequence[str] = []

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key

        # Runtime discovery must NOT happen during construction.
        # This keeps provider initialization fast and network-independent.
        self.model = self.resolve_model(model, discover=False)

        # Cached result of runtime model discovery.
        # None means discovery has not happened yet.
        self._discovered_models: Optional[tuple[str, ...]] = None

    @property
    def is_available(self) -> bool:
        """Return True when the provider has the credentials it needs."""
        return bool(self.api_key)

    def available_models(self) -> Iterable[str]:
        """Return models discovered from the provider.

        Provider subclasses can override this to perform actual discovery.
        This method should only contain the provider-specific discovery logic;
        caching is handled by :meth:`discover_models`.
        """
        return ()

    def fallback_models(self) -> Iterable[str]:
        """Return provider-specific fallback models.

        Provider subclasses should override this with safe fallback
        candidates that can be used without runtime discovery.
        """
        return ()

    def discover_models(self, force: bool = False) -> tuple[str, ...]:
        """Discover and cache models exposed by the provider.

        Args:
            force: Re-run discovery even when a cached result exists.

        Returns:
            A tuple containing discovered model identifiers.

        Discovery is lazy and never happens automatically during provider
        construction.
        """

        if not self.is_available:
            return ()

        if self._discovered_models is not None and not force:
            return self._discovered_models

        try:
            discovered = self.available_models()

            normalized_models: list[str] = []

            for candidate in discovered:
                normalized = self._normalize_model(candidate)

                if normalized is not None:
                    normalized_models.append(normalized)

            self._discovered_models = tuple(normalized_models)

        except Exception:
            # Discovery failure must never make the provider unusable.
            # Keep an empty cache and let fallback models continue to work.
            self._discovered_models = ()

        return self._discovered_models

    def resolve_model(
        self,
        requested_model: Optional[str] = None,
        *,
        discover: bool = False,
    ) -> Optional[str]:
        """Resolve the model to use for this provider.

        Resolution order:

        1. Explicitly requested model.
        2. Discovered runtime model(s), when ``discover=True``.
        3. Provider fallback model(s).

        By default discovery is disabled so this method remains safe to call
        during provider construction.
        """

        available_models: Iterable[str] = ()

        if discover:
            available_models = self.discover_models()

        return resolve_model(
            requested_model=requested_model,
            available_models=available_models,
            fallback_models=self.fallback_models(),
        )

    @staticmethod
    def _normalize_model(model: Optional[str]) -> Optional[str]:
        """Normalize a model name without changing its meaning."""

        if model is None:
            return None

        if not isinstance(model, str):
            raise TypeError("model must be a string or None.")

        normalized = model.strip()
        return normalized or None

    def refresh_model(self) -> Optional[str]:
        """Refresh runtime discovery and resolve a model again.

        The provider's current model is replaced only when a valid model
        candidate is found. Explicit provider configuration is preserved
        because an explicit model continues to have highest priority.
        """

        resolved = self.resolve_model(discover=True)

        if resolved is not None:
            self.model = resolved

        return self.model

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text through the provider."""
        pass