import typer
import time
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

# 🎯 Importing ALL commands
from rdai.cli.commands.init import setup_init 
from rdai.cli.commands.doctor import run_doctor
from rdai.cli.commands.about import run_about
from rdai.cli.commands.config import run_config
from rdai.cli.commands.benchmark import run_benchmark
from rdai.cli.commands.health import run_health

console = Console()
app = typer.Typer(
    name="rdai",
    help="Multi-Brain AI Orchestrator and Router.",
    add_completion=False,
)

# Panel er bhetorer ashol content
LOGO_CONTENT = r"""[bold cyan]
██████╗ ██████╗  █████╗ ██╗
██╔══██╗██╔══██╗██╔══██╗██║
██████╔╝██║  ██║███████║██║
██╔══██╗██║  ██║██╔══██║██║
██║  ██║██████╔╝██║  ██║██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝[/bold cyan]

[bold white]🚀 One Interface. Any AI. Unbreakable Auto-Failover.[/bold white]

[dim]────────────────────────────────────────────────────────[/dim]
[bold yellow]👑 Created by :[/bold yellow] [bold white]Ranajit Dhar[/bold white]
[bold yellow]🌐 Website    :[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]
[dim]────────────────────────────────────────────────────────[/dim]

[yellow]⚡ Get Started:[/yellow]
  [bold green]rdai init[/bold green]    [white]- Setup your AI providers & strategy[/white]
  [bold green]rdai doctor[/bold green]  [white]- Check your API keys and health[/white]

[italic dim]Type [bold cyan]rdai --help[/bold cyan] for all commands.[/italic dim]
"""

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        
        # 🎬 ANIMATION PHASE
        console.print()
        animation_frames = [
            "                           [bold bright_cyan]●[/bold bright_cyan]",
            "\n                        [bold bright_cyan]●──●[/bold bright_cyan]",
            "\n                     [bold bright_cyan]●──●──●[/bold bright_cyan]",
            "\n                  [bold bright_cyan]●──●──●──●[/bold bright_cyan]"
        ]
        
        for frame in animation_frames:
            console.print(frame)
            time.sleep(0.3)
            
        time.sleep(0.4)
        console.print("\n                 [bold green]🧠  Network Established[/bold green]\n")
        time.sleep(0.5)
        
        # 📦 PANEL WITH BORDER
        panel = Panel(
            Align.center(LOGO_CONTENT),
            border_style="cyan",
            padding=(1, 4),
            title="[bold bright_cyan] RDAI SYSTEM DASHBOARD [/bold bright_cyan]",
            title_align="center"
        )
        console.print(panel)
        console.print()

# 🎯 Wiring the exact safe functions
app.command(name="init", help="Setup your AI providers & strategy")(setup_init)
app.command(name="doctor", help="Check your API keys")(run_doctor)
app.command(name="about", help="Learn about the rdai orchestrator")(run_about)
app.command(name="config", help="View your current routing config")(run_config)
app.command(name="benchmark", help="Run a latency test on models")(run_benchmark)
app.command(name="health", help="Check system health and readiness")(run_health)

if __name__ == "__main__":
    app()