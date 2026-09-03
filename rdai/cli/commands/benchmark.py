from __future__ import annotations

import time

import typer
from rich.align import Align
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from rdai import _BUILTIN_PROVIDER_CLASSES, __version__
from rdai.config.discovery import (
    PROVIDER_DISPLAY_NAMES,
    discover_api_keys,
)

console = Console()


def print_mini_header() -> None:
    header = (
        "[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] "
        f"[bold white]v{__version__}[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] "
        "[bold white]Ranajit Dhar[/bold white] | "
        "[bold yellow]🌐 Website:[/bold yellow] "
        "[bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )

    console.print(Align.center(header))
    console.print()


def run_benchmark() -> None:
    """Run a real latency test independently for every configured provider."""

    print_mini_header()

    console.print(
        "\n[bold cyan]⚡ Initiating REAL System Benchmark...[/bold cyan]\n"
    )

    keys = discover_api_keys()

    if not keys:
        console.print(
            "[red]🚨 ALERT: No API credentials detected.[/red]"
        )
        console.print(
            "[yellow]Please run 'rdai init' and configure "
            "at least one provider before running a benchmark.[/yellow]\n"
        )
        raise typer.Exit(code=1)

    configured_names = [
        PROVIDER_DISPLAY_NAMES.get(
            provider,
            provider.replace("_", " ").title(),
        )
        for provider in keys
    ]

    console.print(
        "[yellow]🔍 Found configured providers: "
        f"{', '.join(configured_names)}[/yellow]\n"
    )

    results: list[tuple[str, int | None, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=False,
    ) as progress:

        task = progress.add_task(
            "[cyan]Pinging real networks...[/cyan]",
            total=len(keys),
        )

        for provider_name, api_key in keys.items():
            display_name = PROVIDER_DISPLAY_NAMES.get(
                provider_name,
                provider_name.replace("_", " ").title(),
            )

            provider_class = _BUILTIN_PROVIDER_CLASSES.get(
                provider_name
            )

            if provider_class is None:
                console.print(
                    f"  [yellow]⚠ {display_name}: "
                    "No built-in provider adapter found.[/yellow]"
                )
                results.append(
                    (display_name, None, "unavailable")
                )
                progress.advance(task)
                continue

            try:
                # IMPORTANT:
                # Benchmark calls this provider directly.
                # No AI facade, no router, no failover.
                provider = provider_class(
                    api_key=api_key
                )

                start_time = time.monotonic()

                provider.generate(
                    "Reply with a single word: OK."
                )

                latency_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                results.append(
                    (
                        display_name,
                        latency_ms,
                        "success",
                    )
                )

                console.print(
                    f"  [green]✔ {display_name} responded "
                    f"in {latency_ms}ms[/green]"
                )

            except Exception as error:
                results.append(
                    (
                        display_name,
                        None,
                        str(error),
                    )
                )

                console.print(
                    f"  [red]✖ {display_name} failed "
                    "to respond.[/red]"
                )
                console.print(
                    f"    [dim]{str(error)[:120]}[/dim]"
                )

            progress.advance(task)

    successful = [
        result
        for result in results
        if result[1] is not None
    ]

    failed = len(results) - len(successful)

    console.print()

    if successful:
        fastest = min(
            successful,
            key=lambda result: result[1] or float("inf"),
        )

        console.print(
            f"[bold green]🏆 Fastest: "
            f"{fastest[0]} — {fastest[1]}ms[/bold green]"
        )

    console.print(
        f"[dim]Completed: {len(successful)} succeeded, "
        f"{failed} failed.[/dim]"
    )

    if failed == 0:
        console.print(
            "\n[bold green]✅ Real Benchmark Complete![/bold green]\n"
        )
    else:
        console.print(
            "\n[bold yellow]⚠️ Benchmark completed with "
            f"{failed} failure(s).[/bold yellow]\n"
        )