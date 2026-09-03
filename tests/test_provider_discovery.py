"""Deterministic tests for provider model discovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rdai.providers.deepseek import DeepseekProvider
from rdai.providers.groq import GroqProvider
from rdai.providers.openai import OpenAIProvider


def test_openai_available_models_uses_models_api() -> None:
    fake_client = MagicMock()
    fake_client.models.list.return_value.data = [
        SimpleNamespace(id="model-a"),
        SimpleNamespace(id="model-b"),
    ]

    with patch("openai.OpenAI", return_value=fake_client):
        provider = OpenAIProvider(api_key="test-key")
        models = provider.available_models()

    assert models == ("model-a", "model-b")


def test_groq_available_models_filters_inactive_models() -> None:
    fake_client = MagicMock()
    fake_client.models.list.return_value.data = [
        SimpleNamespace(id="active", active=True),
        SimpleNamespace(id="inactive", active=False),
    ]

    with patch("groq.Groq", return_value=fake_client):
        provider = GroqProvider(api_key="test-key")
        models = provider.available_models()

    assert models == ("active",)


def test_deepseek_available_models_reads_data_field() -> None:
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"id": "model-a"},
            {"id": "model-b"},
        ]
    }
    response.raise_for_status.return_value = None

    with patch("requests.get", return_value=response) as mocked_get:
        provider = DeepseekProvider(api_key="test-key")
        models = provider.available_models()

    assert models == ("model-a", "model-b")
    mocked_get.assert_called_once()


def test_discovery_returns_empty_when_provider_is_unavailable() -> None:
    assert OpenAIProvider().available_models() == ()
    assert GroqProvider().available_models() == ()
    assert DeepseekProvider().available_models() == ()