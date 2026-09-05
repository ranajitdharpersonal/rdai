from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Optional

from rdai.providers.base import BaseProvider

logger = logging.getLogger("rdai.engine")


class ProviderRegistry:
    """Registry for AI provider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""

        name = getattr(provider, "name", None)

        if not name:
            name = provider.__class__.__name__.replace(
                "Provider",
                "",
            ).lower()

        canonical_name = name.lower().strip()

        provider.name = canonical_name
        self._providers[canonical_name] = provider

    def get(
        self,
        name: str,
    ) -> Optional[BaseProvider]:
        """Return a provider by canonical name."""
        return self._providers.get(
            name.lower().strip()
        )

    def all_available(self) -> list[BaseProvider]:
        """Return all currently configured providers."""
        return [
            provider
            for provider in self._providers.values()
            if provider.is_available
        ]

    def get_providers_by_trait(
        self,
        traits: Sequence[str],
    ) -> list[BaseProvider]:
        """Return available providers matching any requested trait."""

        requested = {
            trait.lower().strip()
            for trait in traits
            if isinstance(trait, str) and trait.strip()
        }

        if not requested:
            return []

        matching: list[BaseProvider] = []

        for provider in self._providers.values():
            if not provider.is_available:
                continue

            provider_traits = {
                trait.lower().strip()
                for trait in provider.traits
                if isinstance(trait, str) and trait.strip()
            }

            if requested.intersection(provider_traits):
                matching.append(provider)

        return matching


class Router:
    """Provider routing policy."""

    def __init__(
        self,
        registry: ProviderRegistry,
        strategy: str = "manual",
        priority: Sequence[str] | None = None,
    ) -> None:
        self.registry = registry
        self.strategy = strategy.lower().strip()
        self.priority = [
            name.lower().strip()
            for name in (priority or [])
            if isinstance(name, str) and name.strip()
        ]

    def get_route(
        self,
        prompt: str = "",
        traits: Sequence[str] | None = None,
    ) -> list[BaseProvider]:
        """Return providers in the order they should be attempted."""

        if self.strategy == "manual":
            return self._manual_route()

        if self.strategy == "smart":
            return self._smart_route(
                prompt=prompt,
                traits=traits,
            )

        return self.registry.all_available()

    def _manual_route(self) -> list[BaseProvider]:
        """Build the configured manual provider order."""

        available = self.registry.all_available()

        if not self.priority:
            return available

        route: list[BaseProvider] = []

        for name in self.priority:
            provider = self.registry.get(name)

            if provider and provider.is_available:
                if provider not in route:
                    route.append(provider)

        for provider in available:
            if provider not in route:
                route.append(provider)

        return route

    def _smart_route(
        self,
        prompt: str,
        traits: Sequence[str] | None,
    ) -> list[BaseProvider]:
        """Rank providers by requested traits, then preserve availability order."""

        available = self.registry.all_available()

        if not available:
            return []

        requested_traits = [
            trait.strip().lower()
            for trait in (traits or [])
            if isinstance(trait, str) and trait.strip()
        ]

        prompt_lower = prompt.lower()

        if not requested_traits:
            if any(
                word in prompt_lower
                for word in (
                    "code",
                    "coding",
                    "script",
                    "python",
                    "bug",
                    "react",
                    "program",
                )
            ):
                requested_traits.append("coding")

            if any(
                word in prompt_lower
                for word in (
                    "fast",
                    "quick",
                    "speed",
                )
            ):
                requested_traits.append("fast")

            if any(
                word in prompt_lower
                for word in (
                    "reason",
                    "logic",
                    "analyze",
                    "analysis",
                )
            ):
                requested_traits.append("reasoning")

        if not requested_traits:
            return available

        best_matches = self.registry.get_providers_by_trait(
            requested_traits
        )

        route: list[BaseProvider] = []

        for provider in best_matches:
            if provider not in route:
                route.append(provider)

        for provider in available:
            if provider not in route:
                route.append(provider)

        return route


class Failover:
    """Resilient provider failover with transient-error circuit breaking."""

    def __init__(
        self,
        router: Router,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                "failure_threshold must be at least 1."
            )

        if recovery_timeout < 0:
            raise ValueError(
                "recovery_timeout cannot be negative."
            )

        self.router = router
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.circuit_breakers: dict[
            int,
            dict[str, float | int],
        ] = {}

    @staticmethod
    def _provider_key(
        provider: BaseProvider,
    ) -> int:
        return id(provider)

    @staticmethod
    def _status_code(
        error: Exception,
    ) -> Optional[int]:
        """Extract an HTTP-like status code from an exception."""

        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if isinstance(status_code, int):
            return status_code

        response = getattr(
            error,
            "response",
            None,
        )

        response_status = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(response_status, int):
            return response_status

        return None

    @classmethod
    def _is_transient_error(
        cls,
        error: Exception,
    ) -> bool:
        """Return True for retryable provider/service failures."""

        status_code = cls._status_code(error)

        if status_code is not None:
            return (
                status_code == 408
                or status_code == 429
                or status_code >= 500
            )

        if isinstance(
            error,
            (
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        error_text = str(error).lower()

        transient_markers = (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "connection aborted",
            "temporarily unavailable",
            "temporary failure",
            "service unavailable",
            "server error",
            "rate limit",
            "rate-limited",
            "rate limited",
            "too many requests",
            "resource exhausted",
            "overloaded",
            "network error",
        )

        return any(
            marker in error_text
            for marker in transient_markers
        )

    @classmethod
    def _is_model_error(
        cls,
        error: Exception,
    ) -> bool:
        """Return True for errors indicating model availability problems."""

        status_code = cls._status_code(error)

        if status_code == 404:
            return True

        error_text = str(error).lower()

        return any(
            marker in error_text
            for marker in (
                "model not found",
                "model_not_found",
                "unknown model",
                "invalid model",
                "model does not exist",
                "no such model",
                "model is not available",
                "model is unavailable",
                "model unavailable",
            )
        )

    @staticmethod
    def _has_explicit_model(
        provider: BaseProvider,
    ) -> bool:
        """Return True only when the user explicitly selected a model."""

        requested_model = getattr(
            provider,
            "requested_model",
            None,
        )

        if isinstance(requested_model, str):
            return bool(requested_model.strip())

        requested_model = getattr(
            provider,
            "_requested_model",
            None,
        )

        return bool(
            isinstance(requested_model, str)
            and requested_model.strip()
        )

    def _get_state(
        self,
        provider: BaseProvider,
    ) -> dict[str, float | int]:
        """Return the circuit state for one provider."""

        key = self._provider_key(provider)

        state = self.circuit_breakers.get(key)

        if state is None:
            state = {
                "failures": 0,
                "last_failure": 0.0,
            }

            self.circuit_breakers[key] = state

        return state

    def _circuit_allows_request(
        self,
        state: dict[str, float | int],
    ) -> bool:
        """Return whether a provider's circuit permits a request."""

        failures = int(
            state["failures"]
        )

        if failures < self.failure_threshold:
            return True

        last_failure = float(
            state["last_failure"]
        )

        return (
            time.monotonic() - last_failure
        ) >= self.recovery_timeout

    @staticmethod
    def _reset_state(
        state: dict[str, float | int],
    ) -> None:
        """Reset a provider circuit after successful generation."""

        state["failures"] = 0
        state["last_failure"] = 0.0

    def _record_failure(
        self,
        state: dict[str, float | int],
    ) -> None:
        """Record one transient provider failure."""

        state["failures"] = (
            int(state["failures"]) + 1
        )
        state["last_failure"] = time.monotonic()

    def _recover_model(
        self,
        provider: BaseProvider,
        failed_model: Optional[str],
    ) -> bool:
        """Refresh discovered models after a model-related generation failure.

        Explicitly selected user models are never replaced.
        Discovered models can be excluded so the next discovered candidate
        gets a chance.
        """

        if self._has_explicit_model(provider):
            return False

        refresh_model = getattr(
            provider,
            "refresh_model",
            None,
        )

        if not callable(refresh_model):
            return False

        previous_model = getattr(
            provider,
            "model",
            None,
        )

        try:
            refreshed_model = refresh_model(
                failed_model=failed_model,
            )
        except TypeError:
            # Backward compatibility for custom providers implementing the
            # older no-argument refresh_model() method.
            try:
                refreshed_model = refresh_model()
            except Exception as error:
                logger.warning(
                    "%s model refresh failed: %s",
                    provider.__class__.__name__,
                    error,
                )
                return False
        except Exception as error:
            logger.warning(
                "%s model refresh failed: %s",
                provider.__class__.__name__,
                error,
            )
            return False

        if not refreshed_model:
            return False

        if refreshed_model == previous_model:
            return False

        logger.info(
            "%s switched model from '%s' to '%s'.",
            provider.__class__.__name__,
            previous_model,
            refreshed_model,
        )

        return True

    def _try_provider(
        self,
        provider: BaseProvider,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> str:
        """Attempt one provider, including one model-recovery retry."""

        try:
            return provider.generate(
                prompt,
                **kwargs,
            )

        except Exception as error:
            if not self._is_model_error(error):
                raise

            failed_model = getattr(
                provider,
                "model",
                None,
            )

            logger.warning(
                "%s failed because of model '%s': %s",
                provider.__class__.__name__,
                failed_model,
                error,
            )

            recovered = self._recover_model(
                provider,
                failed_model=(
                    failed_model
                    if isinstance(failed_model, str)
                    else None
                ),
            )

            if not recovered:
                raise

            logger.info(
                "%s retrying with refreshed model '%s'.",
                provider.__class__.__name__,
                getattr(provider, "model", None),
            )

            return provider.generate(
                prompt,
                **kwargs,
            )

    def generate(
        self,
        prompt: str,
        traits: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response through the routed failover chain."""

        providers = self.router.get_route(
            prompt,
            traits=traits,
        )

        if not providers:
            raise RuntimeError(
                "🚨 No available AI providers to route "
                "the request! Check your .env API keys."
            )

        last_transient_error: Optional[Exception] = None
        attempted_provider = False

        for provider in providers:
            state = self._get_state(provider)

            if not self._circuit_allows_request(state):
                logger.info(
                    "Circuit open for %s; skipping provider.",
                    provider.__class__.__name__,
                )
                continue

            attempted_provider = True

            provider_name = provider.__class__.__name__

            try:
                response = self._try_provider(
                    provider,
                    prompt,
                    kwargs,
                )

                self._reset_state(state)

                return response

            except Exception as error:
                if self._is_transient_error(error):
                    last_transient_error = error

                    self._record_failure(state)

                    logger.warning(
                        "%s failed with transient error: "
                        "%s. Trying next provider.",
                        provider_name,
                        error,
                    )

                    continue

                if self._is_model_error(error):
                    logger.warning(
                        "%s still rejected model '%s' "
                        "after recovery attempt: %s",
                        provider_name,
                        getattr(provider, "model", None),
                        error,
                    )

                    continue

                logger.error(
                    "%s failed with non-transient error: %s",
                    provider_name,
                    error,
                )

                raise

        if not attempted_provider:
            raise RuntimeError(
                "🚨 All available providers are currently "
                "behind an open circuit. Please retry shortly."
            )

        if last_transient_error is not None:
            raise RuntimeError(
                "🚨 All providers in the failover chain "
                "failed with transient errors."
            ) from last_transient_error

        raise RuntimeError(
            "🚨 No provider completed the request."
        )