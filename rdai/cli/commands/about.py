import typer
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

console = Console()

def print_mini_header():
    header = (
        "[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] [bold white]v1.0.2[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] [bold white]Ranajit Dhar[/bold white] | [bold yellow]🌐 Website:[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )
    console.print(Align.center(header))
    console.print()

def run_about():
    """Learn about the rdai orchestrator and architecture."""
    console.print()
    print_mini_header()
    
    about_text = (
        "[bold white]The Quantum-Ready, Self-Healing, Multi-Brain AI OS[/bold white]\n\n"
        "[cyan]rdai[/cyan] is not just a routing tool; it is a master orchestration engine.\n"
        "Designed to eliminate AI downtime, it features a limitless circuit breaker\n"
        "mechanism that silently auto-failovers across any number of AI providers.\n\n"
        "[bold green]Key Features:[/bold green]\n"
        " 🧠 [yellow]Limitless Multi-Brain Orchestration[/yellow]\n"
        " ⚡ [yellow]Zero-Downtime Smart Auto-Routing[/yellow]\n"
        " 🛠️ [yellow]Bring-Your-Own-Model (BYOM) Ready[/yellow]"
    )
    
    console.print(Panel(Align.center(about_text), border_style="cyan"))
    console.print()