from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional


class BaseProvider(ABC):
    """Abstract base class for all AI providers in rdai.

    Model resolution is intentionally lightweight at this layer:
    - an explicitly supplied model always wins;
    - otherwise, a provider may offer available/fallback models;
    - if nothing is available, ``None`` is returned so existing provider
      defaults remain backward compatible.
    """

    # Traits help the Smart Router choose the best model.
    traits: Sequence[str] = []

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = self.resolve_model(model)

    @property
    def is_available(self) -> bool:
        """Return True when the provider has the credentials it needs."""
        return bool(self.api_key)

    @staticmethod
    def normalize_model(model: Optional[str]) -> Optional[str]:
        """Normalize a model name without changing its meaning."""

        if model is None:
            return None

        if not isinstance(model, str):
            raise TypeError("model must be a string or None.")

        normalized = model.strip()
        return normalized or None

    def available_models(self) -> Iterable[str]:
        """Return models discovered or advertised by this provider.

        Providers can override this later when they support runtime model
        discovery. The base implementation deliberately returns nothing.
        """

        return ()

    def fallback_models(self) -> Iterable[str]:
        """Return provider-specific fallback models.

        Providers can override this later with safe fallback candidates.
        """

        return ()

    def resolve_model(self, requested_model: Optional[str] = None) -> Optional[str]:
        explicit = self.normalize_model(requested_model)
        if explicit is not None:
            return explicit

        for candidates in (
            self.available_models(),
            self.fallback_models(),
        ):
            for candidate in candidates:
                normalized = self.normalize_model(candidate)
                if normalized is not None:
                    return normalized
                
        return None

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text through the provider."""
        pass