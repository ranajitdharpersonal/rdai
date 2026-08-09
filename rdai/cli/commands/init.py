import os
import sys
import typer
import questionary
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

INIT_CONTENT = r"""[bold cyan]
██████╗ ██████╗  █████╗ ██╗
██╔══██╗██╔══██╗██╔══██╗██║
██████╔╝██║  ██║███████║██║
██╔══██╗██║  ██║██╔══██║██║
██║  ██║██████╔╝██║  ██║██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝[/bold cyan]

[bold yellow]⚙️  SYSTEM INITIALIZATION SEQUENCE[/bold yellow]
[bold white]Version: v1.0.2[/bold white]

[dim]────────────────────────────────────────────────────────[/dim]
[bold yellow]👑 Created by :[/bold yellow] [bold white]Ranajit Dhar[/bold white]
[bold yellow]🌐 Website    :[/bold yellow] [bold cyan][https://ranajitdhar.in](https://ranajitdhar.in)[/bold cyan]
[dim]────────────────────────────────────────────────────────[/dim]

[bold white]Configure your Multi-Brain AI Orchestrator.[/bold white]
[dim]Select providers and define your failover strategy below.[/dim]
"""

def setup_init():
    """Initialize the rdai multi-brain AI environment."""
    
    console.print()
    
    # 🛑 1st Step: Check for existing configuration to prevent accidental overwrite
    if os.path.exists("rdai.yaml") or os.path.exists(".env"):
        console.print("[yellow]⚠️ Warning: Existing configuration (rdai.yaml or .env) detected![/yellow]")
        overwrite = questionary.confirm("Do you want to overwrite your current setup?").ask()
        
        if not overwrite:
            console.print("[green]✔ Setup cancelled. Your existing configuration is safe![/green]\n")
            raise typer.Exit()
            
    # Note: Dependencies are now managed via pyproject.toml
    
    # 🎨 The Dashboard Face for Init with Creator Details
    console.print(Panel(
        Align.center(INIT_CONTENT),
        border_style="cyan",
        title="[bold cyan] rdai (Ranajit Dhar AI) Setup [/bold cyan]"
    ))
    console.print()
    
    # ⚙️ The Setup Questions
    providers = questionary.checkbox(
        "🧠 Select the AI providers you want to use (Press <Space> to select, <Enter> to submit):",
        choices=[
            "Gemini",
            "VertexAI",
            "OpenAI", 
            "Claude",
            "AWS_Bedrock",
            "Groq", 
            "DeepSeek", 
            "Qwen",
            "Llama",
            "Mistral",
            "HuggingFace",
            "Custom (Bring Your Own Model)"
        ],
        validate=lambda result: "❌ Please select at least one provider! (Use SPACE to select before pressing ENTER)" if len(result) == 0 else True,
        style=questionary.Style([("highlighted", "fg:cyan bold")])
    ).ask()
    
    if not providers:
        console.print("[red]🚨 No providers selected. Setup cancelled.[/red]")
        raise typer.Exit()

    strategy_choice = questionary.select(
        "🚀 How do you want to route your AI requests?",
        choices=[
            "🧠 Smart Auto Routing (Automatically chooses the best provider)",
            "👑 Manual Priority (You define the provider order)"
        ]
    ).ask()
    
    strategy = "smart" if "Smart" in strategy_choice else "manual"
    
    # 🎯 Formatting provider names perfectly
    provider_order = []
    for p in providers:
        if "Custom" in p:
            provider_order.append("custom")
        else:
            provider_order.append(p.lower())
    
    if strategy == "manual" and len(providers) > 1:
        console.print("\n[yellow]Configure your unbreakable failover chain:[/yellow]")
        
        # Format the display names for the priority selection
        display_names = ["Custom" if "Custom" in p else p for p in providers]
        
        primary_display = questionary.select(
            "👑 Choose your Primary Provider:",
            choices=display_names
        ).ask()
        
        fallbacks = [p for p in display_names if p != primary_display]
        provider_order = [primary_display.lower()] + [f.lower() for f in fallbacks]
        
    # ⏳ Magic loading animation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Creating configuration...", total=None)
        
        env_content = ""
        for p in providers:
            if "Custom" in p:
                env_content += "\n# 🛠️ Bring Your Own Model Configuration\n"
                env_content += "# CUSTOM_API_KEY=your_key_here\n"
                env_content += "# CUSTOM_ENDPOINT=[https://your-custom-ai.com/v1](https://your-custom-ai.com/v1)\n"
            # 🎯 FIX: Explicitly handle VertexAI environment key naming
            elif p.lower() == "vertexai":
                env_content += "VERTEXAI_PROJECT_ID=\n"
            else:
                env_content += f"{p.upper()}_API_KEY=\n"
        
        with open(".env", "w") as f:
            f.write(env_content)
            
        config_data = {
            "strategy": strategy,
            # 🎯 FIX: Using canonical key 'providers' for rdai.yaml
            "providers": provider_order
        }
        with open("rdai.yaml", "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
    # 🎉 Success Message
    console.print("\n[bold green]🎉 Setup completed successfully![/bold green]")
    console.print("✔ [cyan].env[/cyan] created. Please add your API keys there.")
    console.print(f"✔ [cyan]rdai.yaml[/cyan] created with [bold]{strategy}[/bold] strategy.")
    
    if strategy == "manual":
        chain_str = " ➔ ".join([p.capitalize() for p in provider_order])
        console.print(f"✔ Failover Chain: [bold yellow]{chain_str}[/bold yellow]")
    
    console.print("\n[white]Run[/white] [bold cyan]rdai doctor[/bold cyan] [white]to verify your setup![/white]\n")