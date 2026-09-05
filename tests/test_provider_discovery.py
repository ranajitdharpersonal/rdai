"""Provider discovery contract tests."""

from __future__ import annotations

from typing import Any

from rdai.providers.base import BaseProvider


class FakeDiscoveryProvider(BaseProvider):
    """Deterministic provider used to validate shared discovery behavior."""

    def __init__(
        self,
        *,
        model: str | None = None,
        discovered: tuple[str, ...] = (),
    ) -> None:
        self._available = discovered
        self.discovery_calls = 0

        super().__init__(
            api_key="fake-key",
            model=model,
        )

    def available_models(self) -> tuple[str, ...]:
        self.discovery_calls += 1
        return self._available

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return f"fake: {prompt}"


def test_no_explicit_model_uses_discovery() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.model == "model-a"
    assert provider.discovery_calls == 1


def test_explicit_model_bypasses_discovery() -> None:
    provider = FakeDiscoveryProvider(
        model="user-model",
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.model == "user-model"
    assert provider.discovery_calls == 0


def test_discovery_result_is_cached() -> None:
    provider = FakeDiscoveryProvider(
        discovered=("model-a",),
    )

    assert provider.model == "model-a"
    assert provider.model == "model-a"

    assert provider.discovery_calls == 1


def test_refresh_excludes_failed_discovered_model() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.model == "model-a"

    assert provider.refresh_model(
        failed_model="model-a",
    ) == "model-b"

    assert provider.model == "model-b"
    assert provider.excluded_models == ("model-a",)


def test_refresh_does_not_replace_explicit_model() -> None:
    provider = FakeDiscoveryProvider(
        model="user-model",
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.refresh_model(
        failed_model="user-model",
    ) == "user-model"

    assert provider.model == "user-model"
    assert provider.excluded_models == ()


def test_empty_discovery_returns_no_model() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(),
    )

    assert provider.model is None


def test_discovery_failure_returns_no_model() -> None:
    class FailingProvider(FakeDiscoveryProvider):
        def available_models(self) -> tuple[str, ...]:
            self.discovery_calls += 1
            raise RuntimeError("provider unavailable")

    provider = FailingProvider()

    assert provider.model is None


def test_provider_filter_can_remove_candidates() -> None:
    class FilteringProvider(FakeDiscoveryProvider):
        def filter_models(
            self,
            models: tuple[str, ...] | list[str],
        ) -> tuple[str, ...]:
            return tuple(
                model
                for model in models
                if "invalid" not in model
            )

    provider = FilteringProvider(
        discovered=(
            "invalid-model",
            "valid-model",
        ),
    )

    assert provider.model == "valid-model"


def test_provider_rank_can_change_candidate_order() -> None:
    class RankingProvider(FakeDiscoveryProvider):
        def rank_models(
            self,
            models: tuple[str, ...] | list[str],
        ) -> tuple[str, ...]:
            return tuple(
                sorted(
                    models,
                    key=lambda model: (
                        0
                        if model.startswith("preferred")
                        else 1
                    ),
                )
            )

    provider = RankingProvider(
        discovered=(
            "ordinary-model",
            "preferred-model",
        ),
    )

    assert provider.model == "preferred-model"


def test_failed_model_is_not_reused_after_refresh() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "model-a",
            "model-b",
            "model-c",
        ),
    )

    assert provider.model == "model-a"

    assert provider.refresh_model(
        failed_model="model-a",
    ) == "model-b"

    assert provider.refresh_model(
        failed_model="model-b",
    ) == "model-c"

    assert provider.model == "model-c"


def test_reset_model_exclusions_allows_previous_model_again() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "model-a",
            "model-b",
        ),
    )

    assert provider.model == "model-a"

    assert provider.refresh_model(
        failed_model="model-a",
    ) == "model-b"

    provider.reset_model_exclusions()

    assert provider.model == "model-a"
    assert provider.excluded_models == ()


def test_duplicate_discovered_models_are_removed() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "model-a",
            "model-a",
            "model-b",
        ),
    )

    assert provider.discover_models() == (
        "model-a",
        "model-b",
    )


def test_whitespace_candidates_are_normalized() -> None:
    provider = FakeDiscoveryProvider(
        discovered=(
            "  model-a  ",
            "   ",
            "",
            "model-b",
        ),
    )

    assert provider.discover_models() == (
        "model-a",
        "model-b",
    )