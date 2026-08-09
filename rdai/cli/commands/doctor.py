import typer
import os
import time
from rich.console import Console
from rich.table import Table
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import dotenv_values

# Amader core theke model class gulo niye aschi live test er jonno
from rdai import _BUILTIN_PROVIDER_CLASSES

console = Console()

def print_mini_header():
    header = (
        "[bold cyan]🚀 rdai (Ranajit Dhar AI)[/bold cyan] [bold white]v1.0.2[/bold white]\n"
        "[bold yellow]👑 Created by:[/bold yellow] [bold white]Ranajit Dhar[/bold white] | [bold yellow]🌐 Website:[/bold yellow] [bold cyan]https://ranajitdhar.in[/bold cyan]\n"
        "[dim]────────────────────────────────────────────────────────────[/dim]"
    )
    console.print(Align.center(header))
    console.print()

def run_doctor():
    """Scan full .env file and perform LIVE API diagnostics."""
    console.print()
    print_mini_header()
    
    env_path = ".env"
    if not os.path.exists(env_path):
        console.print(Align.center("[red]❌ .env file not found! Run 'rdai init' first.[/red]\n"))
        raise typer.Exit()
        
    # 🎯 FIX: Ebar amra rdai.yaml noy, direct .env scan korchi
    env_vars = dotenv_values(env_path)
    
    # Filter only keys that end with _API_KEY
    api_keys_to_test = {k: v for k, v in env_vars.items() if k.endswith("_API_KEY") and "CUSTOM" not in k}
    
    if not api_keys_to_test:
        console.print(Align.center("[yellow]⚠️ No provider API keys found in .env file to test![/yellow]\n"))
        raise typer.Exit()
        
    # 📊 The Ultimate Full-Scan Diagnostic Table
    table = Table(title="🩺 Full .env API Key Diagnostics", title_style="bold cyan", border_style="cyan")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Key Config", justify="center")
    table.add_column("Live Check", justify="center")
    table.add_column("Latency", justify="right")
    
    results = []
    
    # ⏳ Progress Spinner for Live Pinging
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Scanning .env and pinging {len(api_keys_to_test)} providers...", total=len(api_keys_to_test))
        
        for env_var, key in api_keys_to_test.items():
            # Extract provider name (e.g., GEMINI_API_KEY -> Gemini)
            p_name = env_var.replace("_API_KEY", "").capitalize()
            
            # 1. Blank Key Check
            if not key or len(key.strip()) < 5:
                results.append((p_name, "[red]❌ MISSING[/red]", "[dim]-[/dim]", "[dim]-[/dim]"))
                progress.advance(task)
                continue
                
            key_status = "[green]✔ DETECTED[/green]"
            
            # 2. Match with our SDK classes
            provider_class = _BUILTIN_PROVIDER_CLASSES.get(p_name.lower())
            
            if not provider_class:
                results.append((p_name, key_status, "[yellow]⚠️ UNKNOWN[/yellow]", "[dim]-[/dim]"))
                progress.advance(task)
                continue
                
            # 3. Live Ping Execution
            start_time = time.time()
            try:
                # Attempt connection
                adapter = provider_class(api_key=key.strip())
                # Send a tiny prompt to verify authentication
                adapter.generate("Reply OK") 
                
                latency = (time.time() - start_time) * 1000
                results.append((p_name, key_status, "[green]🟢 ALIVE[/green]", f"[green]{latency:.0f}ms[/green]"))
                
            except NotImplementedError:
                # Skeleton models er jonno graceful status
                results.append((p_name, key_status, "[blue]🔍 FORMAT OK (SDK Pending)[/blue]", "[dim]-[/dim]"))
            except Exception as e:
                # Real error classification
                error_str = str(e).lower()
                if "401" in error_str or "auth" in error_str or "unauthorized" in error_str:
                    diag_status = "[red]❌ INVALID API KEY[/red]"
                elif "403" in error_str or "forbidden" in error_str:
                    diag_status = "[red]🚫 ACCESS DENIED[/red]"
                elif "429" in error_str or "rate" in error_str or "quota" in error_str:
                    diag_status = "[yellow]⏳ RATE LIMITED[/yellow]"
                elif "timeout" in error_str:
                    diag_status = "[yellow]⏱ TIMEOUT[/yellow]"
                elif "connection" in error_str or "network" in error_str:
                    diag_status = "[red]🌐 NETWORK ERROR[/red]"
                else:
                    # Truncate unexpected errors so they fit in the table
                    clean_err = str(e).replace('\n', ' ')[:20]
                    diag_status = f"[red]🔥 ERROR: {clean_err}...[/red]"
                    
                results.append((p_name, key_status, diag_status, "[dim]-[/dim]"))
                
            progress.advance(task)
            
    # Load data into table
    for res in results:
        table.add_row(*res)
        
    console.print(Align.center(table))
    console.print(Align.center("\n[dim]Note: 'INVALID API KEY' indicates authentication failure. 'FORMAT OK' means SDK is pending.[/dim]\n"))