from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values
from rich.align import Align
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rdai import _BUILTIN_PROVIDER_CLASSES, __version__
from rdai.config.discovery import (
    API_KEY_PLACEHOLDERS,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_ENV_KEYS,
)

console = Console()


def print_mini_header() -> None:
    header = (
        f"[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] "
        f"[bold white]v{__version__}[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] "
        "[bold white]Ranajit Dhar[/bold white] | "
        "[bold yellow]🌐 Website:[/bold yellow] "
        "[bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )

    console.print(Align.center(header))
    console.print()


def _usable_value(
    provider: str,
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()

    if not stripped:
        return None

    placeholder = API_KEY_PLACEHOLDERS.get(provider)

    if (
        placeholder
        and stripped.casefold() == placeholder.casefold()
    ):
        return None

    return stripped


def _classify_error(error: Exception) -> str:
    error_text = str(error).lower()

    status_code = getattr(error, "status_code", None)

    if status_code == 401 or any(
        marker in error_text
        for marker in (
            "401",
            "unauthorized",
            "authentication",
            "invalid api key",
            "invalid key",
        )
    ):
        return "[red]❌ INVALID CREDENTIALS[/red]"

    if status_code == 403 or any(
        marker in error_text
        for marker in (
            "403",
            "forbidden",
            "permission denied",
            "access denied",
        )
    ):
        return "[red]🚫 ACCESS DENIED[/red]"

    if status_code == 429 or any(
        marker in error_text
        for marker in (
            "429",
            "rate limit",
            "rate-limited",
            "too many requests",
            "quota",
            "resource exhausted",
        )
    ):
        return "[yellow]⏳ RATE LIMITED[/yellow]"

    if status_code == 408 or any(
        marker in error_text
        for marker in (
            "timeout",
            "timed out",
        )
    ):
        return "[yellow]⏱ TIMEOUT[/yellow]"

    if any(
        marker in error_text
        for marker in (
            "connection",
            "network",
            "dns",
            "connection reset",
            "connection refused",
        )
    ):
        return "[red]🌐 NETWORK ERROR[/red]"

    if status_code == 404 or any(
        marker in error_text
        for marker in (
            "model not found",
            "model_not_found",
            "unknown model",
            "invalid model",
        )
    ):
        return "[yellow]⚠️ MODEL ERROR[/yellow]"

    clean_error = " ".join(
        str(error).split()
    )

    if not clean_error:
        clean_error = error.__class__.__name__

    return (
        "[red]🔥 ERROR: "
        f"{clean_error[:40]}[/red]"
    )


def _load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        return {}

    values = dotenv_values(env_path)

    return {
        key: value.strip()
        for key, value in values.items()
        if isinstance(value, str) and value.strip()
    }


def run_doctor() -> None:
    """Run provider credential and live API diagnostics."""

    console.print()
    print_mini_header()

    env_path = Path(".env").resolve()

    if not env_path.is_file():
        console.print(
            Align.center(
                "[red]❌ .env file not found! "
                "Run 'rdai init' first.[/red]\n"
            )
        )
        raise typer.Exit(code=1)

    env_values = _load_env_values(env_path)

    configured: list[
        tuple[str, str, str]
    ] = []

    for provider, env_var in PROVIDER_ENV_KEYS.items():
        value = _usable_value(
            provider,
            env_values.get(env_var),
        )

        if value is not None:
            display_name = PROVIDER_DISPLAY_NAMES.get(
                provider,
                provider.replace("_", " ").title(),
            )

            configured.append(
                (
                    provider,
                    display_name,
                    value,
                )
            )

    if not configured:
        console.print(
            Align.center(
                "[yellow]⚠️ No configured provider "
                "credentials found in .env.[/yellow]\n"
            )
        )
        raise typer.Exit(code=0)

    table = Table(
        title="🩺 rdai Provider Diagnostics",
        title_style="bold cyan",
        border_style="cyan",
    )

    table.add_column(
        "Provider",
        style="cyan",
        no_wrap=True,
    )
    table.add_column(
        "Credential",
        justify="center",
    )
    table.add_column(
        "Live Check",
        justify="center",
    )
    table.add_column(
        "Latency",
        justify="right",
    )

    results: list[
        tuple[str, str, str, str]
    ] = []

    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}"
        ),
        transient=True,
    ) as progress:
        task = progress.add_task(
            (
                "[cyan]Checking configured providers "
                f"({len(configured)})...[/cyan]"
            ),
            total=len(configured),
        )

        for (
            provider_name,
            display_name,
            credential,
        ) in configured:
            provider_class = (
                _BUILTIN_PROVIDER_CLASSES.get(
                    provider_name
                )
            )

            if provider_class is None:
                results.append(
                    (
                        display_name,
                        "[green]✔ DETECTED[/green]",
                        "[yellow]⚠️ UNKNOWN[/yellow]",
                        "[dim]-[/dim]",
                    )
                )
                progress.advance(task)
                continue

            start = time.monotonic()

            try:
                adapter = provider_class(
                    api_key=credential
                )

                adapter.generate("Reply OK")

                latency_ms = (
                    time.monotonic() - start
                ) * 1000

                results.append(
                    (
                        display_name,
                        "[green]✔ DETECTED[/green]",
                        "[green]🟢 ALIVE[/green]",
                        f"[green]{latency_ms:.0f}ms[/green]",
                    )
                )

            except NotImplementedError:
                results.append(
                    (
                        display_name,
                        "[green]✔ DETECTED[/green]",
                        "[blue]🔍 FORMAT OK[/blue]",
                        "[dim]-[/dim]",
                    )
                )

            except Exception as error:
                results.append(
                    (
                        display_name,
                        "[green]✔ DETECTED[/green]",
                        _classify_error(error),
                        "[dim]-[/dim]",
                    )
                )

            progress.advance(task)

    for result in results:
        table.add_row(*result)

    console.print()
    console.print(Align.center(table))
    console.print(
        Align.center(
            "\n[dim]Credentials are never printed. "
            "Live Check sends a minimal test request "
            "to the configured provider.[/dim]\n"
        )
    )