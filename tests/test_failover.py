"""Focused failover tests using deterministic, local provider fakes."""

from __future__ import annotations

from typing import Any

from rdai.core.engine import Failover, ProviderRegistry, Router
from rdai.providers.base import BaseProvider


class RateLimitedError(Exception):
    """Minimal HTTP-like error object that represents a 429 response."""

    status_code = 429


class ModelNotFoundError(Exception):
    """Minimal HTTP-like error object that represents a 404 model error."""

    status_code = 404


class FakeProvider(BaseProvider):
    is_available = True

    def __init__(
        self,
        name: str,
        outcomes: list[str | BaseException],
        model: str = "fake-model",
        explicit_model: bool = False,
        refreshed_model: str = "refreshed-model",
    ) -> None:
        self.name = name
        self.traits = ["general"]
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.api_key = "fake-key"
        self.model = model
        self._requested_model = model if explicit_model else None
        self.refreshed_model = refreshed_model
        self.refresh_calls = 0
        self.refresh_failed_models: list[str | None] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))

        if not self._outcomes:
            return "default"

        outcome = self._outcomes.pop(0)

        if isinstance(outcome, BaseException):
            raise outcome

        return outcome

    def refresh_model(
        self,
        *,

        failed_model: str | None = None,
    ) -> str | None:


        self.refresh_calls += 1
        self.refresh_failed_models.append(failed_model)

        self.model = self.refreshed_model
        return self.model


def _register(
    registry: ProviderRegistry,
    provider: FakeProvider,
) -> None:
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
    primary = FakeProvider(
        "primary",
        [RateLimitedError("too many requests")],
    )
    fallback = FakeProvider(
        "fallback",
        ["served by fallback"],
    )

    failover = _failover_with(primary, fallback)

    response = failover.generate("hello")

    assert response == "served by fallback"
    assert primary.calls
    assert fallback.calls == [("hello", {})]


def test_timeout_failure_moves_to_next_provider_and_preserves_kwargs() -> None:
    primary = FakeProvider(
        "primary",
        [TimeoutError("network timed out")],
    )
    fallback = FakeProvider(
        "fallback",
        ["retry succeeded"],
    )

    failover = _failover_with(primary, fallback)

    response = failover.generate(
        "hello",
        temperature=0.2,
    )

    assert response == "retry succeeded"
    assert primary.calls
    assert fallback.calls == [
        ("hello", {"temperature": 0.2})
    ]


def test_model_error_refreshes_model_and_retries_same_provider() -> None:
    primary = FakeProvider(
        "primary",
        [
            ModelNotFoundError("model not found"),
            "recovered with new model",
        ],
        model="old-model",
        refreshed_model="new-model",
    )

    failover = _failover_with(primary)

    response = failover.generate("hello")

    assert response == "recovered with new model"
    assert primary.refresh_calls == 1
    assert primary.model == "new-model"
    assert len(primary.calls) == 2


def test_explicit_model_is_never_overridden() -> None:
    primary = FakeProvider(
        "primary",
        [ModelNotFoundError("model not found")],
        model="user-selected-model",
        explicit_model=True,
        refreshed_model="should-not-be-used",
    )

    fallback = FakeProvider(
        "fallback",
        ["fallback success"],
    )

    failover = _failover_with(primary, fallback)

    response = failover.generate("hello")

    assert response == "fallback success"
    assert primary.refresh_calls == 0
    assert primary.model == "user-selected-model"


def test_failed_model_recovery_moves_to_next_provider() -> None:
    primary = FakeProvider(
        "primary",
        [
            ModelNotFoundError("model not found"),
            ModelNotFoundError("model not found again"),
        ],
        model="old-model",
        refreshed_model="new-model",
    )

    fallback = FakeProvider(
        "fallback",
        ["served by fallback"],
    )

    failover = _failover_with(primary, fallback)

    response = failover.generate("hello")

    assert response == "served by fallback"
    assert primary.refresh_calls == 1
    assert len(primary.calls) == 2
    assert fallback.calls == [("hello", {})]


def test_model_error_does_not_increment_circuit_failures() -> None:
    primary = FakeProvider(
        "primary",
        [
            ModelNotFoundError("model not found"),
            ModelNotFoundError("model not found again"),
        ],
        model="old-model",
        refreshed_model="new-model",
    )

    fallback = FakeProvider(
        "fallback",
        ["fallback success"],
    )

    failover = _failover_with(primary, fallback)

    response = failover.generate("hello")

    assert response == "fallback success"

    state = failover.circuit_breakers[id(primary)]

    assert state["failures"] == 0


def test_model_error_passes_failed_model_to_refresh() -> None:
    primary = FakeProvider(
        "primary",
        [
            ModelNotFoundError("model not found"),
            "recovered",
        ],
        model="old-model",
        refreshed_model="new-model",
    )

    failover = _failover_with(primary)

    response = failover.generate("hello")

    assert response == "recovered"
    assert primary.refresh_failed_models == ["old-model"]
