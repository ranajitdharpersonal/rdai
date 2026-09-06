from __future__ import annotations

import typer
from rich.align import Align
from rich.console import Console

from rdai import __version__
from rdai.config.loader import ConfigError, load_config

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

    console.print(
        Align.center(header)
    )
    console.print()


def run_config() -> None:
    """View the current rdai routing configuration."""

    console.print()
    print_mini_header()

    try:
        config = load_config()

    except ConfigError as error:
        console.print(
            Align.center(
                f"[red]❌ Invalid rdai.yaml:[/red] {error}\n"
            )
        )
        raise typer.Exit(code=1) from error

    strategy = config.strategy.upper()

    if config.providers:
        chain = " ➔ ".join(
            provider.replace("_", " ").title()
            for provider in config.providers
        )
    else:
        chain = "Automatic provider discovery"

    config_text = (
        "[bold white]⚙️  Current System Configuration:[/bold white]\n\n"
        f"  • Routing Strategy : [bold cyan]{strategy}[/bold cyan]\n"
        f"  • Provider Chain   : [bold yellow]{chain}[/bold yellow]\n"
    )

    console.print(
        Align.center(config_text)
    )
    console.print()