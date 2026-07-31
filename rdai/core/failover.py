import time
import logging
from typing import Sequence, Any
from rdai.core.router import Router

logger = logging.getLogger("rdai.failover")

class Failover:
    """
    Enterprise-grade Circuit Breaker for AI Providers.
    Prevents cascading failures and handles timeouts/rate limits seamlessly.
    """
    def __init__(self, router: Router, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.router = router
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        # Tracks health: { "provider_name": {"failures": 0, "last_failure_time": 0.0} }
        self._health_status = {}

    def _is_circuit_open(self, provider_name: str) -> bool:
        """Checks if a provider is temporarily blocked due to repeated failures."""
        status = self._health_status.get(provider_name, {"failures": 0, "last_failure_time": 0.0})
        if status["failures"] >= self.failure_threshold:
            time_since_failure = time.time() - status["last_failure_time"]
            if time_since_failure < self.recovery_timeout:
                return True  # Circuit is open (provider is dead/skipped)
            else:
                return False # Time to probe again! (Half-open state)
        return False

    def _record_failure(self, provider_name: str):
        """Records a failure and updates the timestamp."""
        status = self._health_status.get(provider_name, {"failures": 0, "last_failure_time": 0.0})
        status["failures"] += 1
        status["last_failure_time"] = time.time()
        self._health_status[provider_name] = status

    def _record_success(self, provider_name: str):
        """Resets health on success."""
        if provider_name in self._health_status:
            self._health_status[provider_name] = {"failures": 0, "last_failure_time": 0.0}

    def generate(self, prompt: str, *, traits: Sequence[str] = None, **kwargs: Any) -> str:
        """The magic shield that routes and catches errors silently."""
        
        # 1. Ask the Router for the best sequence of brains
        chain = self.router.get_route(prompt, traits=traits)
        
        if not chain:
            raise RuntimeError("🚨 No providers available to route the request.")

        # 2. Iterate through the chain
        for provider_name in chain:
            if self._is_circuit_open(provider_name):
                # logger.warning(f"⚠️ {provider_name.upper()} circuit is open. Skipping...")
                continue

            provider = self.router.registry.get_provider(provider_name)
            
            try:
                # 3. Try calling the provider
                response = provider.generate(prompt, **kwargs)
                
                # If success, reset any past failures
                self._record_success(provider_name)
                return response
                
            except Exception as e:
                # 🛡️ THE SHIELD: Catch error, record it, and silently switch!
                self._record_failure(provider_name)
                print(f"⚠️ {provider_name.upper()} failed: {str(e)}. Switching to next brain...")
                continue

        # If loop finishes and everything failed
        raise RuntimeError("🚨 All AI providers in the failover chain have failed or are rate-limited!")