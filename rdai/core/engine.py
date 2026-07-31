import time
import logging
from typing import Any, Sequence, Optional, Dict
from rdai.providers.base import BaseProvider

# Logging setup for silent failover tracking
logger = logging.getLogger("rdai.engine")

class ProviderRegistry:
    """Stores and manages all AI provider instances."""
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        # Extract name from class (e.g., GeminiProvider -> gemini)
        name = provider.__class__.__name__.lower().replace("provider", "")
        self._providers[name] = provider

    def get(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name.lower())

    def all_available(self, order: Sequence[str] = None) -> list[BaseProvider]:
        """Returns only providers that have API keys (is_available == True)."""
        if order:
            return [
                self._providers[name] for name in order 
                if name in self._providers and self._providers[name].is_available
            ]
        return [p for p in self._providers.values() if p.is_available]


class Router:
    """Routes requests based on strategy and available providers."""
    
    def __init__(self, registry: ProviderRegistry, strategy: str = "manual", priority: Sequence[str] = None):
        self.registry = registry
        self.strategy = strategy
        self.priority = priority or []

    def get_route(self) -> list[BaseProvider]:
        """Determine the execution path."""
        # For V1, both smart and manual will respect the unbreakable priority chain
        return self.registry.all_available(self.priority)


class Failover:
    """The Unbreakable 3-Tier Circuit Breaker System."""
    
    def __init__(self, router: Router, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.router = router
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        # Tracks health of each provider
        self.circuit_breakers = {}

    def generate(self, prompt: str, traits: Sequence[str] = None, **kwargs: Any) -> str:
        providers = self.router.get_route()
        
        if not providers:
            raise RuntimeError("🚨 No available AI providers to route the request! Check your .env API keys.")

        for provider in providers:
            provider_name = provider.__class__.__name__
            
            # Check circuit breaker state
            state = self.circuit_breakers.get(provider_name, {"failures": 0, "last_failure": 0})
            
            # If threshold reached, check if it's time to try again
            if state["failures"] >= self.failure_threshold:
                if time.time() - state["last_failure"] < self.recovery_timeout:
                    # Circuit Open: Skip this model
                    continue 
                else:
                    # Circuit Half-Open: Let's try giving it another chance
                    state["failures"] = 0 

            # Attempt Generation
            try:
                return provider.generate(prompt, **kwargs)
                
            except Exception as e:
                logger.warning(f"⚠️ {provider_name} failed: {e}. Initiating Auto-Failover...")
                # Update failure state
                state["failures"] += 1
                state["last_failure"] = time.time()
                self.circuit_breakers[provider_name] = state

        # If loop finishes without returning, all providers in the chain crashed
        raise RuntimeError("🚨 ALL providers in the failover chain crashed. Circuit is completely broken.")