"""Tests for the public rdai generate CLI command."""

from __future__ import annotations

from collections.abc import Iterator

from typer.testing import CliRunner

from rdai.cli.main import app

runner = CliRunner()


def test_generate_command_uses_ai_runtime(monkeypatch) -> None:
    """The CLI should call AI.generate() and render the response."""

    calls: list[str] = []

    class FakeAI:
        def generate(self, prompt: str) -> str:
            calls.append(prompt)
            return "hello from rdai"

    monkeypatch.setattr(
        "rdai.AI",
        FakeAI,
    )

    result = runner.invoke(
        app,
        ["generate", "Say hello"],
    )

    assert result.exit_code == 0
    assert "hello from rdai" in result.output
    assert calls == ["Say hello"]


def test_generate_command_requires_prompt() -> None:
    """The CLI should reject a missing prompt."""

    result = runner.invoke(
        app,
        ["generate"],
    )

    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_generate_command_streams_response(monkeypatch) -> None:
    """The CLI --stream flag should use AI.stream() and render chunks."""

    calls: list[str] = []

    class FakeAI:
        def generate(self, prompt: str) -> str:
            raise AssertionError(
                "generate() must not be called in streaming mode"
            )

        def stream(self, prompt: str) -> Iterator[str]:
            calls.append(prompt)

            yield "hello"
            yield " "
            yield "from stream"

    monkeypatch.setattr(
        "rdai.AI",
        FakeAI,
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "Say hello",
            "--stream",
        ],
    )

    assert result.exit_code == 0
    assert "hello from stream" in result.output
    assert calls == ["Say hello"]


def test_generate_command_stream_flag_is_optional(monkeypatch) -> None:
    """The CLI should preserve normal generate behaviour without --stream."""

    calls: list[str] = []

    class FakeAI:
        def generate(self, prompt: str) -> str:
            calls.append(prompt)
            return "normal response"

        def stream(self, prompt: str) -> Iterator[str]:
            raise AssertionError(
                "stream() must not be called without --stream"
            )

    monkeypatch.setattr(
        "rdai.AI",
        FakeAI,
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "Hello",
        ],
    )

    assert result.exit_code == 0
    assert "normal response" in result.output
    assert calls == ["Hello"]