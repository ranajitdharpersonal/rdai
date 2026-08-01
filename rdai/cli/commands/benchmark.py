import time
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.align import Align  # 👈 Added this missing import!
from rdai.config.discovery import discover_api_keys

console = Console()

def print_mini_header():
    header = (
        "[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] [bold white]v1.0.1[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] [bold white]Ranajit Dhar[/bold white] | [bold yellow]🌐 Website:[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )
    console.print(Align.center(header))
    console.print()

def run_benchmark():
    """Run a real latency test on configured models."""
    print_mini_header()  # 👈 Called the header function here!
    
    console.print("\n[bold cyan]⚡ Initiating REAL System Benchmark...[/bold cyan]\n")
    
    # 🛑 1. Check for real keys first! No more fake simulation!
    keys = discover_api_keys()
    
    if not keys:
        console.print("[red]🚨 ALERT: No API keys detected in .env file![/red]")
        console.print("[yellow]Please run 'rdai init' and add real keys before running a benchmark.[/yellow]\n")
        raise typer.Exit()

    console.print(f"[yellow]🔍 Found active keys for: {', '.join(keys.keys()).title()}[/yellow]\n")
    
    # 2. Dynamic import to avoid circular dependencies
    from rdai import AI
    
    # ⏳ 3. Real Progress and Real Pinging
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=False,
    ) as progress:
        
        task = progress.add_task("[cyan]Pinging real networks...[/cyan]", total=len(keys))
        
        for provider_name, api_key in keys.items():
            try:
                # We inject only one provider at a time to test its true latency
                tester = AI(strategy="manual", providers=[provider_name]) 
                
                start_time = time.time()
                # 📡 REAL API CALL!
                tester.generate("Reply with a single word: OK.") 
                end_time = time.time()
                
                latency = int((end_time - start_time) * 1000)
                console.print(f"  [green]✔ {provider_name.capitalize()} actually responded in {latency}ms[/green]")
                
            except Exception as e:
                console.print(f"  [red]✖ {provider_name.capitalize()} failed to respond. (Invalid Key or Timeout)[/red]")
                
            progress.advance(task)

    console.print("\n[bold green]✅ Real Benchmark Complete![/bold green]\n")