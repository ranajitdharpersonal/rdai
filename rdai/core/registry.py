from typing import Dict, List
from rdai.providers.base import BaseProvider

class ProviderRegistry:
    """
    Enterprise-grade Dynamic Registry for AI Providers.
    Manages instantiated provider adapters.
    """
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Registers a provider instance."""
        # Assuming BaseProvider has a 'name' attribute or we derive it from class name
        # For simplicity, we use the class name minus 'Provider' and lowercased
        name = provider.__class__.__name__.replace("Provider", "").lower()
        self._providers[name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Fetches a provider instance by name."""
        name = name.lower()
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' is not registered in the system.")
        return self._providers[name]
        
    def get_all_providers(self) -> Dict[str, BaseProvider]:
        """Returns all registered provider instances."""
        return self._providers

    def get_providers_by_trait(self, traits: List[str]) -> List[str]:
        """Finds providers that have ANY of the requested traits."""
        matching = []
        for name, provider in self._providers.items():
            if any(trait.lower() in [t.lower() for t in provider.traits] for trait in traits):
                matching.append(name)
        return matching