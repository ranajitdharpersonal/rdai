"""Focused routing tests using providers that never touch external SDKs."""

from __future__ import annotations

from typing import Any

# FIX: Ebar sob kichu unified engine theke import hocche
from rdai.core.engine import ProviderRegistry, Router
from rdai.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """Small in-memory provider used to exercise routing decisions only."""
    
    # FIX: Test provider gulo jate 'is_available' property pass kore
    is_available = True

    def __init__(self, name: str, traits: list[str]) -> None:
        self.name = name
        self.traits = traits
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Simulate base class requirements
        self.api_key = "fake-key"
        self.model = "fake-model"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        return f"{self.name}: {prompt}"


def _register(registry: ProviderRegistry, provider: FakeProvider) -> None:
    registry.register(provider)


def _registry_with_providers() -> ProviderRegistry:
    registry = ProviderRegistry()
    _register(registry, FakeProvider("gemini", ["general", "fast"]))
    _register(registry, FakeProvider("groq", ["coding", "fast"]))
    _register(registry, FakeProvider("openai", ["creative", "reasoning"]))
    return registry


def test_smart_router_prioritizes_provider_matching_requested_trait() -> None:
    registry = _registry_with_providers()
    router = Router(registry, strategy="smart")

    # FIX: unified get_route() use kora hocche
    chain = router.get_route("Write a Python function.", traits=["coding"])
    selected = chain[0] if chain else None

    assert selected is not None
    assert selected.name == "groq"
    assert {provider.name for provider in chain} == {"gemini", "groq", "openai"}


def test_manual_router_honors_priority_and_keeps_remaining_fallbacks() -> None:
    registry = _registry_with_providers()
    router = Router(
        registry,
        strategy="manual",
        priority=["openai", "groq"],
    )

    # FIX: unified get_route() use kora hocche
    chain = router.get_route("Any request, regardless of traits.")
    selected = chain[0] if chain else None

    assert selected is not None
    assert selected.name == "openai"
    assert [provider.name for provider in chain[:2]] == ["openai", "groq"]
    assert {provider.name for provider in chain} == {"gemini", "groq", "openai"}