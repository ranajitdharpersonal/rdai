import time
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.align import Align
from rdai.config.discovery import discover_api_keys

console = Console()

def print_mini_header():
    header = (
        "[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] [bold white]v1.0.2[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] [bold white]Ranajit Dhar[/bold white] | [bold yellow]🌐 Website:[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )
    console.print(Align.center(header))
    console.print()

def run_benchmark():
    """Run a real latency test on configured models."""
    print_mini_header()
    
    console.print("\n[bold cyan]⚡ Initiating REAL System Benchmark...[/bold cyan]\n")
    
    keys = discover_api_keys()
    
    if not keys:
        console.print("[red]🚨 ALERT: No API keys detected in .env file![/red]")
        console.print("[yellow]Please run 'rdai init' and add real keys before running a benchmark.[/yellow]\n")
        raise typer.Exit()

    console.print(f"[yellow]🔍 Found active keys for: {', '.join(keys.keys()).title()}[/yellow]\n")
    
    # Dynamic import
    from rdai import AI
    
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
                # 🎯 FIX: Test real loaded AI engine using manual override
                import yaml
                temp_yaml = {"strategy": "manual", "providers": [provider_name]}
                with open("temp_benchmark.yaml", "w") as f:
                    yaml.dump(temp_yaml, f)
                    
                tester = AI(config_path="temp_benchmark.yaml") 
                
                start_time = time.time()
                tester.generate("Reply with a single word: OK.") 
                end_time = time.time()
                
                latency = int((end_time - start_time) * 1000)
                console.print(f"  [green]✔ {provider_name.capitalize()} actually responded in {latency}ms[/green]")
                
                import os
                if os.path.exists("temp_benchmark.yaml"):
                    os.remove("temp_benchmark.yaml")
                
            except Exception as e:
                console.print(f"  [red]✖ {provider_name.capitalize()} failed to respond. (Error: {e})[/red]")
                
            progress.advance(task)

    console.print("\n[bold green]✅ Real Benchmark Complete![/bold green]\n")