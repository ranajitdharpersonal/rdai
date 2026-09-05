from __future__ import annotations

import os
from pathlib import Path

import questionary
import typer
import yaml
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rdai.config.discovery import (
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_ENV_KEYS,
)

console = Console()

INIT_CONTENT = r"""[bold cyan]
██████╗ ██████╗  █████╗ ██╗
██╔══██╗██╔══██╗██╔══██╗██║
██████╔╝██║  ██║███████║██║
██╔══██╗██║  ██║██╔══██║██║
██║  ██║██████╔╝██║  ██║██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝[/bold cyan]

[bold yellow]⚙️  SYSTEM INITIALIZATION SEQUENCE[/bold yellow]
[bold white]Version: v1.0.2[/bold white]

[dim]────────────────────────────────────────────────────────[/dim]
[bold yellow]👑 Created by :[/bold yellow] [bold white]Ranajit Dhar[/bold white]
[bold yellow]🌐 Website    :[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]
[dim]────────────────────────────────────────────────────────[/dim]

[bold white]Configure your Multi-Brain AI Orchestrator.[/bold white]
[dim]Select providers and define your failover strategy below.[/dim]
"""


def _display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(
        provider,
        provider.replace("_", " ").title(),
    )


def _provider_choices() -> list[str]:
    return [
        _display_name(provider)
        for provider in PROVIDER_ENV_KEYS
    ]


def _provider_from_display(display_name: str) -> str:
    for provider, configured_name in PROVIDER_DISPLAY_NAMES.items():
        if configured_name == display_name:
            return provider

    normalized = (
        display_name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return normalized


def _write_env_file(
    providers: list[str],
    env_path: Path,
) -> None:
    lines: list[str] = []

    for provider in providers:
        env_key = PROVIDER_ENV_KEYS.get(provider)

        if env_key is None:
            continue

        lines.append(f"{env_key}=")

    env_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_config_file(
    strategy: str,
    provider_order: list[str],
    config_path: Path,
) -> None:
    config_data = {
        "strategy": strategy,
        "providers": provider_order,
    }

    with config_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as config_file:
        yaml.safe_dump(
            config_data,
            config_file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def setup_init() -> None:
    """Initialize the rdai multi-brain AI environment."""

    console.print()

    config_path = Path("rdai.yaml")
    env_path = Path(".env")

    if config_path.exists() or env_path.exists():
        console.print(
            "[yellow]⚠️ Warning: Existing configuration "
            "(rdai.yaml or .env) detected![/yellow]"
        )

        overwrite = questionary.confirm(
            "Do you want to overwrite your current setup?"
        ).ask()

        if not overwrite:
            console.print(
                "[green]✔ Setup cancelled. "
                "Your existing configuration is safe![/green]\n"
            )
            raise typer.Exit(code=0)

    console.print(
        Panel(
            Align.center(INIT_CONTENT),
            border_style="cyan",
            title="[bold cyan] rdai (Ranajit Dhar AI) Setup [/bold cyan]",
        )
    )
    console.print()

    selected_display_names = questionary.checkbox(
        "🧠 Select the AI providers you want to use "
        "(Press <Space> to select, <Enter> to submit):",
        choices=_provider_choices(),
        validate=lambda result: (
            "❌ Please select at least one provider!"
            if not result
            else True
        ),
        style=questionary.Style(
            [("highlighted", "fg:cyan bold")]
        ),
    ).ask()

    if not selected_display_names:
        console.print(
            "[red]🚨 No providers selected. Setup cancelled.[/red]"
        )
        raise typer.Exit(code=0)

    providers = [
        _provider_from_display(display_name)
        for display_name in selected_display_names
    ]

    strategy_choice = questionary.select(
        "🚀 How do you want to route your AI requests?",
        choices=[
            "🧠 Smart Auto Routing "
            "(Automatically chooses the best provider)",
            "👑 Manual Priority "
            "(You define the provider order)",
        ],
    ).ask()

    if not strategy_choice:
        console.print(
            "[red]🚨 No routing strategy selected. "
            "Setup cancelled.[/red]"
        )
        raise typer.Exit(code=0)

    strategy = (
        "smart"
        if "Smart" in strategy_choice
        else "manual"
    )

    provider_order = list(providers)

    if strategy == "manual" and len(providers) > 1:
        console.print(
            "\n[yellow]Configure your unbreakable "
            "failover chain:[/yellow]"
        )

        primary_display = questionary.select(
            "👑 Choose your Primary Provider:",
            choices=selected_display_names,
        ).ask()

        if not primary_display:
            console.print(
                "[red]🚨 No primary provider selected. "
                "Setup cancelled.[/red]"
            )
            raise typer.Exit(code=0)

        primary_provider = _provider_from_display(
            primary_display
        )

        fallbacks = [
            provider
            for provider in providers
            if provider != primary_provider
        ]

        provider_order = [
            primary_provider,
            *fallbacks,
        ]

    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}"
        ),
        transient=True,
    ) as progress:
        progress.add_task(
            description="Creating configuration...",
            total=None,
        )

        _write_env_file(
            providers,
            env_path,
        )

        _write_config_file(
            strategy,
            provider_order,
            config_path,
        )

    console.print(
        "\n[bold green]🎉 Setup completed successfully![/bold green]"
    )
    console.print(
        "✔ [cyan].env[/cyan] created. "
        "Please add your credentials there."
    )
    console.print(
        "✔ [cyan]rdai.yaml[/cyan] created with "
        f"[bold]{strategy}[/bold] strategy."
    )

    if strategy == "manual":
        chain_str = " ➔ ".join(
            _display_name(provider)
            for provider in provider_order
        )

        console.print(
            "✔ Failover Chain: "
            f"[bold yellow]{chain_str}[/bold yellow]"
        )

    console.print(
        "\n[white]Run[/white] "
        "[bold cyan]rdai doctor[/bold cyan] "
        "[white]to verify your setup![/white]\n"
    )