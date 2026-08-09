"""Focused failover tests using deterministic, local provider fakes."""

from __future__ import annotations

from typing import Any

# FIX: Ebar sob kichu unified engine theke import hocche
from rdai.core.engine import Failover, ProviderRegistry, Router
from rdai.providers.base import BaseProvider


class RateLimitedError(Exception):
    """Minimal HTTP-like error object that represents a 429 response."""
    status_code = 429


class FakeProvider(BaseProvider):
    # FIX: Test provider 'is_available' property
    is_available = True
    
    def __init__(self, name: str, outcomes: list[str | BaseException]) -> None:
        self.name = name
        self.traits = ["general"]
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.api_key = "fake-key"
        self.model = "fake-model"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        if not self._outcomes:
            return "default"
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _register(registry: ProviderRegistry, provider: FakeProvider) -> None:
    registry.register(provider)


def _failover_with(*providers: FakeProvider) -> Failover:
    registry = ProviderRegistry()
    for provider in providers:
        _register(registry, provider)
    router = Router(
        registry,
        strategy="manual",
        priority=[provider.name for provider in providers],
    )
    return Failover(router)


def test_rate_limit_failure_moves_to_next_provider() -> None:
    primary = FakeProvider("primary", [RateLimitedError("too many requests")])
    fallback = FakeProvider("fallback", ["served by fallback"])
    failover = _failover_with(primary, fallback)

    response = failover.generate("hello")

    assert response == "served by fallback"
    assert primary.calls
    assert fallback.calls == [("hello", {})]


def test_timeout_failure_moves_to_next_provider_and_preserves_kwargs() -> None:
    primary = FakeProvider("primary", [TimeoutError("network timed out")])
    fallback = FakeProvider("fallback", ["retry succeeded"])
    failover = _failover_with(primary, fallback)

    response = failover.generate("hello", temperature=0.2)

    assert response == "retry succeeded"
    assert primary.calls
    assert fallback.calls == [("hello", {"temperature": 0.2})]