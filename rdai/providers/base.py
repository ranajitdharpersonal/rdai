from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional

class BaseProvider(ABC):
    """Abstract Base Class for all AI Providers in rdai."""
    
    # Traits help the Smart Router choose the best model
    traits: Sequence[str] = []

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @property
    def is_available(self) -> bool:
        """Returns True if the provider has the necessary credentials to run."""
        return bool(self.api_key)

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """The core generation method that must be implemented."""
        pass