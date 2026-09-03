"""Tests for the base provider model-resolution contract."""

from __future__ import annotations

from typing import Any

import pytest

from rdai.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """Local provider used to test model resolution without any network call."""

    def __init__(
        self,
        *,
        api_key: str = "fake-key",
        model: str | None = None,
        discovered: tuple[str, ...] = (),
        fallbacks: tuple[str, ...] = (),
    ) -> None:
        self._discovered = discovered
        self._fallbacks = fallbacks
        super().__init__(api_key=api_key, model=model)

    def available_models(self) -> tuple[str, ...]:
        return self._discovered

    def fallback_models(self) -> tuple[str, ...]:
        return self._fallbacks

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"fake: {prompt}"


def test_explicit_model_always_wins() -> None:
    provider = FakeProvider(
        model="requested-model",
        discovered=("discovered-model",),
        fallbacks=("fallback-model",),
    )

    assert provider.model == "requested-model"


def test_discovered_model_is_used_when_no_explicit_model_exists() -> None:
    provider = FakeProvider(
        discovered=("discovered-model", "another-model"),
        fallbacks=("fallback-model",),
    )

    assert provider.model == "discovered-model"


def test_fallback_model_is_used_when_no_discovered_model_exists() -> None:
    provider = FakeProvider(
        fallbacks=("fallback-model",),
    )

    assert provider.model == "fallback-model"


def test_model_is_none_when_no_candidate_exists() -> None:
    provider = FakeProvider()

    assert provider.model is None


def test_whitespace_model_is_normalized() -> None:
    provider = FakeProvider(model="  my-model  ")

    assert provider.model == "my-model"


def test_blank_model_is_treated_as_missing() -> None:
    provider = FakeProvider(
        model="   ",
        discovered=("discovered-model",),
    )

    assert provider.model == "discovered-model"


def test_invalid_model_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="model must be a string or None"):
        FakeProvider(model=123)  # type: ignore[arg-type]