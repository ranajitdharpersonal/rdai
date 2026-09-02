"""Tests for the public rdai generate CLI command."""

from __future__ import annotations

from typer.testing import CliRunner

from rdai.cli.main import app


runner = CliRunner()


def test_generate_command_uses_ai_runtime(monkeypatch) -> None:
    """The CLI should call AI.generate() and render the response."""

    import rdai.cli.commands.generate as generate_module

    calls: list[str] = []

    class FakeAI:
        def generate(self, prompt: str) -> str:
            calls.append(prompt)
            return "hello from rdai"

    monkeypatch.setattr(
        "rdai.AI",
        FakeAI,
    )

    result = runner.invoke(app, ["generate", "Say hello"])

    assert result.exit_code == 0
    assert "hello from rdai" in result.output
    assert calls == ["Say hello"]


def test_generate_command_requires_prompt() -> None:
    """The CLI should reject a missing prompt."""

    result = runner.invoke(app, ["generate"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output