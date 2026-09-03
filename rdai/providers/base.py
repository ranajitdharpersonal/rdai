from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from rdai.providers.model_resolver import normalize_model, resolve_model


class BaseProvider(ABC):
    """Abstract base class for all AI providers in rdai.

    Model resolution follows a safe, lazy strategy:

    1. An explicitly supplied model always wins.
    2. Static provider fallback models are used during initialization.
    3. Runtime discovery is opt-in and never happens during construction.
    4. Discovered models are cached to avoid repeated network requests.
    """

    # Traits help the Smart Router choose the best provider/model.
    traits: Sequence[str] = ()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key

        # Keep the user's explicit model separate so future discovery/refresh
        # can never accidentally override it.
        self._requested_model = normalize_model(model)

        # Runtime discovery must never happen during construction.
        # This keeps provider initialization fast and network-independent.
        self.model = self.resolve_model(discover=False)

        # None means discovery has not happened yet.
        # An empty tuple means discovery was attempted and found nothing.
        self._discovered_models: Optional[tuple[str, ...]] = None

    @property
    def is_available(self) -> bool:
        """Return True when the provider has the credentials it needs."""
        return bool(self.api_key)

    def available_models(self) -> Iterable[str]:
        """Return models discovered from the provider.

        Provider subclasses can override this with provider-specific
        discovery logic. Network access belongs here, not in ``__init__``.
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
            A tuple of normalized discovered model identifiers.

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
                normalized = normalize_model(candidate)

                if normalized is not None:
                    normalized_models.append(normalized)

            self._discovered_models = tuple(normalized_models)

        except Exception:
            # Discovery must never make the provider unusable.
            # An empty result allows fallback_models() to remain the safety net.
            self._discovered_models = ()

        return self._discovered_models

    def resolve_model(
        self,
        requested_model: Optional[str] = None,
        *,
        discover: bool = False,
    ) -> Optional[str]:
        """Resolve the model for this provider.

        Resolution order:

        1. Explicitly requested model.
        2. Discovered runtime model(s), when ``discover=True``.
        3. Provider fallback model(s).

        When ``requested_model`` is omitted, the model explicitly supplied
        during provider construction is preserved as the highest-priority
        choice.
        """

        explicit = (
            normalize_model(requested_model)
            if requested_model is not None
            else self._requested_model
        )

        if explicit is not None:
            return explicit

        available_models: Iterable[str] = ()

        if discover:
            available_models = self.discover_models()

        return resolve_model(
            requested_model=None,
            available_models=available_models,
            fallback_models=self.fallback_models(),
        )

    def refresh_model(self) -> Optional[str]:
        """Refresh discovered models and update the active model.

        An explicitly configured model always remains authoritative.
        Otherwise, a newly discovered model is preferred, with the provider
        fallback remaining available when discovery returns nothing.
        """

        if self._requested_model is not None:
            self.model = self._requested_model
            return self.model

        self.discover_models(force=True)

        resolved = self.resolve_model(discover=True)

        if resolved is not None:
            self.model = resolved

        return self.model

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text through the provider."""
        pass