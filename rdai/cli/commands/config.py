import typer
import yaml
import os
from rich.console import Console
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

def run_config():
    """View your current routing configuration."""
    console.print()
    print_mini_header()
    
    if not os.path.exists("rdai.yaml"):
        console.print(Align.center("[red]❌ rdai.yaml not found! Run 'rdai init' first.[/red]\n"))
        raise typer.Exit()
        
    with open("rdai.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Formatting output centrally
    config_text = "[bold white]⚙️  Current System Configuration:[/bold white]\n\n"
    config_text += f"  • Routing Strategy : [bold cyan]{config.get('strategy', 'unknown').upper()}[/bold cyan]\n"
    
    # 🎯 FIX: Check both new and old config keys
    providers = config.get("providers", config.get("provider_order", []))
    chain = " ➔ ".join([p.capitalize() for p in providers])
    config_text += f"  • Failover Chain   : [bold yellow]{chain}[/bold yellow]\n"
    
    console.print(Align.center(config_text))
    console.print()