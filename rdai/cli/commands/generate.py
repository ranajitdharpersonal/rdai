"""CLI command for generating text through the rdai runtime."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_generate(
    prompt: str = typer.Argument(
        ...,
        help="The prompt to send through the rdai provider router.",
    ),
) -> None:
    """Generate a response through rdai's configured provider chain."""

    # Import lazily so the rest of the CLI remains lightweight and keeps
    # existing command startup behavior unchanged.
    from rdai import AI

    try:
        ai = AI()
        response = ai.generate(prompt)

    except ValueError as error:
        console.print(f"[red]❌ Invalid request:[/red] {error}")
        raise typer.Exit(code=2) from error

    except RuntimeError as error:
        console.print(
            Panel(
                str(error),
                title="[bold red]RDAI Generation Failed[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from error

    except Exception as error:
        # Keep unexpected provider/runtime failures user-friendly in the CLI.
        console.print(
            Panel(
                f"{error}",
                title="[bold red]Unexpected Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from error

    console.print()
    console.print(
        Panel(
            response,
            title="[bold cyan]RDAI Response[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()