from typing import List, Sequence
from rdai.core.registry import ProviderRegistry

class Router:
    """Smart routing logic strictly decoupled from specific models."""
    
    def __init__(self, registry: ProviderRegistry, strategy: str, priority: Sequence[str]):
        self.registry = registry
        self.strategy = strategy.lower()
        self.priority = list(priority)

    def get_route(self, prompt: str, traits: Sequence[str] = None) -> List[str]:
        """
        Determines the priority chain of providers to call.
        """
        if self.strategy == "manual":
            if not self.priority:
                raise ValueError("🚨 Provider order must be configured when using 'manual' strategy.")
            return self.priority
            
        elif self.strategy == "smart":
            # Baseline is all active providers from registry
            all_available = list(self.registry.get_all_providers().keys())
            chain = all_available.copy()
            
            # If user explicitly requested traits (e.g., ["coding", "fast"])
            requested_traits = traits or []
            
            # Intent detection from prompt if no explicit traits given
            prompt_lower = prompt.lower()
            if not requested_traits:
                if any(word in prompt_lower for word in ["code", "script", "python", "bug", "react"]):
                    requested_traits.append("coding")
                if any(word in prompt_lower for word in ["fast", "quick"]):
                    requested_traits.append("fast")
            
            # Reorder chain based on traits
            if requested_traits:
                best_matches = self.registry.get_providers_by_trait(requested_traits)
                for p in reversed(best_matches):  # Insert at front in order
                    if p in chain:
                        chain.remove(p)
                        chain.insert(0, p)
                        
            return chain
            
        else:
            raise ValueError(f"🚨 Unknown strategy: '{self.strategy}'. Use 'smart' or 'manual'.")