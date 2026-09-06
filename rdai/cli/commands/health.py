from __future__ import annotations

from pathlib import Path

import requests
import typer
from rich.align import Align
from rich.console import Console
from rich.table import Table

from rdai import __version__
from rdai.config.discovery import discover_api_keys
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


def check_internet() -> bool:
    """Verify that outbound HTTPS connectivity is available."""

    try:
        response = requests.get(
            "https://www.google.com/generate_204",
            timeout=3,
        )
        return response.status_code < 500

    except requests.RequestException:
        return False


def run_health() -> None:
    """Check overall rdai configuration and network readiness."""

    console.print()
    print_mini_header()

    table = Table(
        title="🩺 System Health Report",
        title_style="bold cyan",
        border_style="cyan",
    )

    table.add_column(
        "Component",
        style="white",
    )
    table.add_column(
        "Status",
        justify="center",
    )

    env_path = Path(".env").resolve()
    config_path = Path("rdai.yaml").resolve()

    keys = discover_api_keys(
        env_path,
    )

    if keys:
        env_status = (
            f"[green]✔ {len(keys)} PROVIDER(S) READY[/green]"
        )
    elif env_path.is_file():
        env_status = "[yellow]⚠️ NO USABLE CREDENTIALS[/yellow]"
    else:
        env_status = "[yellow]⚠️ NOT CONFIGURED[/yellow]"

    table.add_row(
        "Provider Credentials",
        env_status,
    )

    config_error: ConfigError | None = None

    try:
        config = load_config(
            config_path,
            load_environment=False,
        )
        config_status = "[green]✔ OK[/green]"

    except ConfigError as error:
        config = None
        config_error = error
        config_status = "[red]❌ INVALID[/red]"

    table.add_row(
        "Routing Configuration",
        config_status,
    )

    if config is not None and config.providers:
        failover_status = (
            f"[green]✔ READY ({len(config.providers)} PROVIDER(S))[/green]"
        )
    elif keys:
        failover_status = "[green]✔ READY (AUTO DISCOVERY)[/green]"
    else:
        failover_status = "[yellow]⚠️ WAITING FOR PROVIDERS[/yellow]"

    table.add_row(
        "Failover Engine",
        failover_status,
    )

    network_status = (
        "[green]✔ CONNECTED[/green]"
        if check_internet()
        else "[red]❌ OFFLINE[/red]"
    )

    table.add_row(
        "Network Connection",
        network_status,
    )

    console.print(
        Align.center(table)
    )

    if config_error is not None:
        console.print(
            Align.center(
                "\n[red]Configuration error:[/red] "
                f"{config_error}\n"
            )
        )

    console.print()