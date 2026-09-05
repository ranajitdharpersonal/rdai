"""Tests for the standalone model resolver."""

from __future__ import annotations

import pytest

from rdai.providers.model_resolver import (
    normalize_model,
    resolve_model,
)


def test_explicit_model_wins() -> None:
    assert resolve_model(
        "requested",
        available_models=("discovered",),
    ) == "requested"


def test_discovered_model_wins_when_no_explicit_model() -> None:
    assert resolve_model(
        None,
        available_models=("discovered",),
    ) == "discovered"


def test_none_is_returned_when_no_model_exists() -> None:
    assert resolve_model(
        None,
        available_models=(),
    ) is None


def test_first_valid_discovered_model_is_selected() -> None:
    assert resolve_model(
        None,
        available_models=(
            "",
            "   ",
            "first-valid",
            "second",
        ),
    ) == "first-valid"


def test_model_is_normalized() -> None:
    assert normalize_model("  gpt-test  ") == "gpt-test"


def test_blank_model_becomes_none() -> None:
    assert normalize_model("   ") is None


def test_none_model_stays_none() -> None:
    assert normalize_model(None) is None


def test_non_string_model_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="model must be a string or None",
    ):
        normalize_model(123)  # type: ignore[arg-type]


def test_explicit_model_is_normalized_before_resolution() -> None:
    assert resolve_model(
        "  requested-model  ",
        available_models=("discovered-model",),
    ) == "requested-model"