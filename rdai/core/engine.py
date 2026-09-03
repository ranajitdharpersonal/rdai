import time
import logging
from typing import Any, Sequence, Optional, Dict, List
from rdai.providers.base import BaseProvider

# Logging setup for silent failover tracking
logger = logging.getLogger("rdai.engine")

class ProviderRegistry:
    """Enterprise-grade Dynamic Registry for AI Providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Registers a provider instance."""
        name = getattr(provider, "name", None)
        if not name:
            name = provider.__class__.__name__.replace("Provider", "").lower()

        self._providers[name.lower()] = provider

    def get(self, name: str) -> Optional[BaseProvider]:
        """Fetches a provider instance by name."""
        return self._providers.get(name.lower())

    def all_available(self) -> List[BaseProvider]:
        """Returns all providers that have API keys (is_available == True)."""
        return [p for p in self._providers.values() if p.is_available]

    def get_providers_by_trait(self, traits: Sequence[str]) -> List[BaseProvider]:
        """Finds providers that have ANY of the requested traits."""
        matching = []
        for provider in self._providers.values():
            if provider.is_available and hasattr(provider, 'traits'):
                # Check if any requested trait matches the provider's traits
                if any(trait.lower() in [t.lower() for t in provider.traits] for trait in traits):
                    matching.append(provider)
        return matching


class Router:
    """Smart routing logic decoupled from specific models."""
    
    def __init__(self, registry: ProviderRegistry, strategy: str = "manual", priority: Sequence[str] = None):
        self.registry = registry
        self.strategy = strategy.lower()
        self.priority = list(priority) if priority else []

    def get_route(self, prompt: str = "", traits: Sequence[str] = None) -> List[BaseProvider]:
        """Determine the priority chain of providers to call based on strategy."""
        
        if self.strategy == "manual":
            if not self.priority:
                return self.registry.all_available()
            
            # Follow exact configured order
            route = []
            for name in self.priority:
                p = self.registry.get(name)
                if p and p.is_available:
                    route.append(p)
                    
            # 🎯 FIX: Append remaining available providers as fallback
            available = self.registry.all_available()
            for p in available:
                if p not in route:
                    route.append(p)
                    
            return route
            
        elif self.strategy == "smart":
            # Baseline is all active providers
            available = self.registry.all_available()
            chain = available.copy()
            
            requested_traits = list(traits) if traits else []
            prompt_lower = prompt.lower()
            
            # Intent detection from prompt if no explicit traits given
            if not requested_traits:
                if any(word in prompt_lower for word in ["code", "script", "python", "bug", "react"]):
                    requested_traits.append("coding")
                if any(word in prompt_lower for word in ["fast", "quick"]):
                    requested_traits.append("fast")
            
            # Reorder chain based on traits
            if requested_traits:
                best_matches = self.registry.get_providers_by_trait(requested_traits)
                for p in reversed(best_matches):  # Insert at front in priority
                    if p in chain:
                        chain.remove(p)
                        chain.insert(0, p)
                        
            return chain
            
        else:
            # Fallback to standard availability
            return self.registry.all_available()


class Failover:
    """The Unbreakable 3-Tier Circuit Breaker System."""
    
    def __init__(self, router: Router, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.router = router
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        # Tracks health of each provider
        self.circuit_breakers = {}

    def generate(self, prompt: str, traits: Sequence[str] = None, **kwargs: Any) -> str:
        """The magic shield that routes and catches errors silently."""
        providers = self.router.get_route(prompt, traits=traits)
        
        if not providers:
            raise RuntimeError("🚨 No available AI providers to route the request! Check your .env API keys.")

        for provider in providers:
            provider_name = provider.__class__.__name__
            
            # Check circuit breaker state
            state = self.circuit_breakers.get(provider_name, {"failures": 0, "last_failure": 0.0})
            
            # If threshold reached, check if it's time to probe again
            if state["failures"] >= self.failure_threshold:
                if time.time() - state["last_failure"] < self.recovery_timeout:
                    # Circuit Open: Skip this model
                    continue 
                else:
                    # Circuit Half-Open: Probe mode
                    state["failures"] = 0 

            # Attempt Generation
            try:
                response = provider.generate(prompt, **kwargs)
                
                # If success, reset any past failures
                state["failures"] = 0
                self.circuit_breakers[provider_name] = state
                return response
                
            except Exception as e:
                # 🛡️ THE SHIELD: Catch error, record it, and silently switch!
                logger.warning(f"⚠️ {provider_name} failed: {str(e)}. Initiating Auto-Failover...")
                print(f"⚠️ {provider_name.replace('Provider', '').upper()} failed. Switching to next brain...")
                
                # Update failure state
                state["failures"] += 1
                state["last_failure"] = time.time()
                self.circuit_breakers[provider_name] = state

        # If loop finishes without returning, all providers crashed
        raise RuntimeError("🚨 ALL providers in the failover chain crashed or are rate-limited. Circuit is completely broken.")