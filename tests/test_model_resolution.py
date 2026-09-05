"""Tests for the base provider model-resolution contract."""

from __future__ import annotations

from typing import Any

import pytest

from rdai.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """Local provider used to test model resolution without network calls."""

    def __init__(
        self,
        *,
        api_key: str | None = "fake-key",
        model: str | None = None,
        discovered: tuple[str, ...] = (),
    ) -> None:
        self._discovered = discovered
        self.discovery_calls = 0

        super().__init__(
            api_key=api_key,
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        self.discovery_calls += 1
        return self._discovered

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return f"fake: {prompt}"


def test_explicit_model_always_wins() -> None:
    provider = FakeProvider(
        model="requested-model",
        discovered=("discovered-model",),
    )

    assert provider.model == "requested-model"
    assert provider.discovery_calls == 0


def test_no_explicit_model_discovers_lazily() -> None:
    provider = FakeProvider(
        discovered=(
            "discovered-model",
            "another-model",
        ),
    )

    assert provider.model == "discovered-model"
    assert provider.discovery_calls == 1


def test_discovered_models_are_cached() -> None:
    provider = FakeProvider(
        discovered=("discovered-model",),
    )

    assert provider.model == "discovered-model"
    assert provider.model == "discovered-model"

    assert provider.discovery_calls == 1


def test_forced_discovery_refreshes_cache() -> None:
    provider = FakeProvider(
        discovered=("first-model",),
    )

    assert provider.model == "first-model"
    assert provider.discovery_calls == 1

    provider._discovered = ("second-model",)

    assert provider.refresh_model() == "second-model"
    assert provider.discovery_calls == 2


def test_model_is_none_when_discovery_returns_nothing() -> None:
    provider = FakeProvider(
        discovered=(),
    )

    assert provider.model is None
    assert provider.discovery_calls == 1


def test_failed_discovery_does_not_create_a_fallback_model() -> None:
    class FailingProvider(FakeProvider):
        def available_models(self) -> tuple[str, ...]:
            self.discovery_calls += 1
            raise RuntimeError("discovery failed")

    provider = FailingProvider()

    assert provider.model is None
    assert provider.discovery_calls == 1


def test_explicit_model_does_not_trigger_discovery() -> None:
    provider = FakeProvider(
        model="explicit-model",
        discovered=("discovered-model",),
    )

    assert provider.model == "explicit-model"
    assert provider.discovery_calls == 0


def test_explicit_model_remains_authoritative_after_refresh() -> None:
    provider = FakeProvider(
        model="explicit-model",
        discovered=("discovered-model",),
    )

    assert provider.refresh_model() == "explicit-model"
    assert provider.model == "explicit-model"
    assert provider.discovery_calls == 0


def test_whitespace_model_is_normalized() -> None:
    provider = FakeProvider(
        model="  my-model  ",
    )

    assert provider.model == "my-model"


def test_blank_model_is_treated_as_missing() -> None:
    provider = FakeProvider(
        model="   ",
        discovered=("discovered-model",),
    )

    assert provider.model == "discovered-model"


def test_invalid_model_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="model must be a string or None",
    ):
        FakeProvider(model=123)  # type: ignore[arg-type]


def test_unavailable_provider_does_not_trigger_discovery() -> None:
    provider = FakeProvider(
        api_key=None,
        discovered=("discovered-model",),
    )

    assert provider.is_available is False
    assert provider.model is None
    assert provider.discovery_calls == 0


def test_discover_models_can_be_called_directly() -> None:
    provider = FakeProvider(
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.discover_models() == (
        "model-a",
        "model-b",
    )

    assert provider.discovery_calls == 1


def test_force_discovery_can_replace_cached_models() -> None:
    provider = FakeProvider(
        discovered=("model-a",),
    )

    assert provider.discover_models() == ("model-a",)

    provider._discovered = (
        "model-b",
        "model-c",
    )

    assert provider.discover_models(
        force=True,
    ) == (
        "model-b",
        "model-c",
    )

    assert provider.discovery_calls == 2


def test_ensure_model_returns_discovered_model() -> None:
    provider = FakeProvider(
        discovered=("discovered-model",),
    )

    assert provider.ensure_model() == "discovered-model"


def test_ensure_model_raises_when_no_model_is_available() -> None:
    provider = FakeProvider(
        discovered=(),
    )

    with pytest.raises(
        RuntimeError,
        match="No model is available for fake",
    ):
        provider.ensure_model()


def test_discovery_failure_causes_ensure_model_to_raise() -> None:
    class FailingProvider(FakeProvider):
        def available_models(self) -> tuple[str, ...]:
            self.discovery_calls += 1
            raise RuntimeError("network failure")

    provider = FailingProvider()

    with pytest.raises(
        RuntimeError,
        match="No model is available for failing",
    ):
        provider.ensure_model()


def test_model_property_discovers_only_when_needed() -> None:
    provider = FakeProvider(
        discovered=("discovered-model",),
    )

    assert provider.discovery_calls == 0

    _ = provider.model

    assert provider.discovery_calls == 1


def test_runtime_model_assignment_is_supported() -> None:
    provider = FakeProvider()

    assert provider.model is None

    provider.model = "runtime-model"

    assert provider.model == "runtime-model"


def test_runtime_model_assignment_does_not_override_explicit_model() -> None:
    provider = FakeProvider(
        model="explicit-model",
    )

    provider.model = "runtime-model"

    assert provider.model == "explicit-model"


def test_resolve_model_with_explicit_argument_does_not_discover() -> None:
    provider = FakeProvider(
        discovered=("discovered-model",),
    )

    assert provider.resolve_model(
        requested_model="requested-model",
        discover=True,
    ) == "requested-model"

    assert provider.discovery_calls == 0


def test_resolve_model_discovers_when_no_explicit_model_exists() -> None:
    provider = FakeProvider(
        discovered=("discovered-model",),
    )

    assert provider.resolve_model(
        discover=True,
    ) == "discovered-model"

    assert provider.discovery_calls == 1


def test_refresh_model_excludes_failed_discovered_model() -> None:
    provider = FakeProvider(
        discovered=(
            "first-model",
            "second-model",
            "third-model",
        ),
    )

    assert provider.model == "first-model"

    assert provider.refresh_model(
        failed_model="first-model",
    ) == "second-model"

    assert provider.model == "second-model"
