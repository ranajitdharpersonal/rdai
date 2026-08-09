import typer
import os
import requests
from rich.console import Console
from rich.table import Table
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

def check_internet():
    """Verify actual internet connection."""
    try:
        requests.get("https://1.1.1.1", timeout=3)
        return True
    except requests.RequestException:
        return False

def run_health():
    """Check overall system health and network readiness."""
    console.print()
    print_mini_header()
    
    table = Table(title="🩺 System Health Report", title_style="bold cyan", border_style="cyan")
    table.add_column("Component", style="white")
    table.add_column("Status", justify="center")
    
    # Check .env
    env_status = "[green]✔ OK[/green]" if os.path.exists(".env") else "[red]❌ MISSING[/red]"
    table.add_row("Environment File (.env)", env_status)
    
    # Check Config
    yaml_status = "[green]✔ OK[/green]" if os.path.exists("rdai.yaml") else "[red]❌ MISSING[/red]"
    table.add_row("Config File (rdai.yaml)", yaml_status)
    
    # 🎯 FIX: Real Network Check
    net_status = "[green]✔ CONNECTED[/green]" if check_internet() else "[red]❌ OFFLINE[/red]"
    
    table.add_row("Failover Engine", "[green]✔ STANDBY[/green]" if env_status == "[green]✔ OK[/green]" else "[yellow]⚠️ BLOCKED[/yellow]")
    table.add_row("Network Connection", net_status)
    
    console.print(Align.center(table))
    console.print()