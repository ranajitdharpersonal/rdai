"""Focused tests for the public AI facade."""

from __future__ import annotations

from typing import Any

import pytest

from rdai import AI
from rdai.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    is_available = True

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.traits = ["general"]
        self.api_key = "fake-key"
        self.model = "fake-model"
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        return f"{self.name}: {prompt}"


def test_ai_accepts_injected_providers() -> None:
    provider = FakeProvider("custom")

    ai = AI(providers=[provider])

    response = ai.generate("hello", temperature=0.2)

    assert response == "custom: hello"
    assert provider.calls == [
        ("hello", {"temperature": 0.2})
    ]


def test_ai_rejects_empty_prompt() -> None:
    ai = AI(providers=[FakeProvider()])

    with pytest.raises(ValueError, match="non-empty"):
        ai.generate("   ")


def test_ai_respects_explicit_strategy() -> None:
    provider = FakeProvider()

    ai = AI(
        providers=[provider],
        strategy="manual",
    )

    assert ai.strategy == "manual"
    assert ai.router.strategy == "manual"


def test_ai_accepts_explicit_model_mapping_without_breaking_provider() -> None:
    provider = FakeProvider()

    ai = AI(
        providers=[provider],
        models={"fake": "custom-model"},
    )

    response = ai.generate("hello")

    assert response == "fake: hello"
    assert provider.model == "fake-model"