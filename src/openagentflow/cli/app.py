"""
CLI application for Open Agent Flow.

Provides commands for running agents, viewing traces, and managing the framework.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██████╗ ███████╗███╗   ██╗                         ║
║  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║                         ║
║  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║                         ║
║  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║                         ║
║  ╚██████╔╝██║     ███████╗██║ ╚████║                         ║
║   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝                         ║
║                                                               ║
║   █████╗  ██████╗ ███████╗███╗   ██╗████████╗               ║
║  ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝               ║
║  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                  ║
║  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                  ║
║  ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                  ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                  ║
║                                                               ║
║  ███████╗██╗      ██████╗ ██╗    ██╗                        ║
║  ██╔════╝██║     ██╔═══██╗██║    ██║                        ║
║  █████╗  ██║     ██║   ██║██║ █╗ ██║                        ║
║  ██╔══╝  ██║     ██║   ██║██║███╗██║                        ║
║  ██║     ███████╗╚██████╔╝╚███╔███╔╝                        ║
║  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝                         ║
║                                                               ║
║  Distributed Agentic AI Workflows                            ║
║  Graph-Native Reasoning Traces                               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""


@click.group()
@click.version_option(package_name="openagentflow")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress output")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """Open Agent Flow - Distributed Agentic AI Workflows."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    if verbose > 0 and not quiet:
        console.print(BANNER, style="cyan")


@cli.command()
@click.argument("agent_path")
@click.option("-i", "--input", "input_data", help="Input for the agent (JSON string)")
@click.option("-m", "--model", default=None, help="Override model")
@click.option("--max-iterations", type=int, default=None, help="Override max iterations")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--trace/--no-trace", default=True, help="Record reasoning trace")
@click.pass_context
def run(
    ctx: click.Context,
    agent_path: str,
    input_data: str | None,
    model: str | None,
    max_iterations: int | None,
    stream: bool,
    trace: bool,
) -> None:
    """
    Run an agent from a Python file.

    AGENT_PATH can be:
    - A Python file: agent.py (runs first @agent decorated function)
    - A specific function: agent.py::researcher
    """
    import asyncio
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    quiet = ctx.obj.get("quiet", False)

    # Parse agent path
    if "::" in agent_path:
        file_path, func_name = agent_path.split("::", 1)
    else:
        file_path = agent_path
        func_name = None

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        sys.exit(1)

    # Load the module
    spec = importlib.util.spec_from_file_location("agent_module", path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Error:[/red] Could not load module: {file_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    # Find the agent function
    from openagentflow.core.agent import get_all_agents

    agents = get_all_agents()
    if not agents:
        console.print("[red]Error:[/red] No @agent decorated functions found")
        sys.exit(1)

    if func_name:
        if func_name not in agents:
            console.print(f"[red]Error:[/red] Agent '{func_name}' not found")
            sys.exit(1)
        agent_spec = agents[func_name]
    else:
        # Use first agent
        agent_spec = list(agents.values())[0]

    # Parse input
    if input_data:
        try:
            input_dict = json.loads(input_data)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON input: {e}")
            sys.exit(1)
    else:
        input_dict = {}

    if not quiet:
        console.print(f"[bold]Running agent:[/bold] {agent_spec.name}")
        console.print(f"[dim]Model:[/dim] {agent_spec.model.model_id}")

    # Run the agent
    async def execute():
        func = agent_spec.func
        if hasattr(func, "_async_call"):
            return await func._async_call(**input_dict)
        else:
            return await func(**input_dict)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Executing agent...", total=None)
        result = asyncio.run(execute())
        progress.update(task, completed=True)

    # Display result
    if not quiet:
        if hasattr(result, "status"):
            status_color = "green" if result.status.name == "SUCCEEDED" else "red"
            console.print(f"\n[{status_color}]Status:[/{status_color}] {result.status.name}")
            if result.output:
                console.print(Panel(str(result.output), title="Output"))
            if result.error:
                console.print(f"[red]Error:[/red] {result.error}")
            console.print(f"[dim]Duration:[/dim] {result.duration_ms:.2f}ms")
        else:
            console.print(Panel(str(result), title="Output"))


@cli.command()
@click.argument("trace_id")
@click.option("-f", "--format", "fmt", type=click.Choice(["tree", "json", "timeline"]), default="tree")
@click.pass_context
def trace(ctx: click.Context, trace_id: str, fmt: str) -> None:
    """View an agent reasoning trace."""
    console.print(f"[yellow]Trace viewing not yet implemented[/yellow]")
    console.print(f"Trace ID: {trace_id}")
    console.print(f"Format: {fmt}")


@cli.command()
@click.argument("trace_id")
@click.option("--step/--no-step", default=False, help="Step through trace interactively")
@click.pass_context
def replay(ctx: click.Context, trace_id: str, step: bool) -> None:
    """Replay an agent execution for debugging."""
    console.print(f"[yellow]Trace replay not yet implemented[/yellow]")
    console.print(f"Trace ID: {trace_id}")


@cli.command()
@click.option("-p", "--port", type=int, default=8080, help="Server port")
@click.option("-h", "--host", default="127.0.0.1", help="Server host")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str, reload: bool) -> None:
    """Start the Open Agent Flow API server."""
    console.print(f"[bold]Starting server on {host}:{port}[/bold]")

    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error:[/red] uvicorn not installed. Run: pip install openagentflow[server]")
        return

    try:
        from openagentflow.server.app import create_app
    except ImportError:
        console.print("[red]Error:[/red] Server module not found. Run: pip install openagentflow[server]")
        return

    app = create_app()

    if not ctx.obj.get("quiet", False):
        console.print(f"[green]Dashboard:[/green] http://{host}:{port}")
        console.print(f"[green]API docs:[/green]  http://{host}:{port}/api/docs")
        console.print("[dim]Press Ctrl+C to stop[/dim]")

    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from openagentflow import __version__

    console.print(f"[bold]Open Agent Flow[/bold] v{__version__}")
    console.print("[dim]Distributed Agentic AI Workflows[/dim]")


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    from openagentflow.config import get_settings

    settings = get_settings()
    console.print(Panel(str(settings.to_dict()), title="Configuration"))


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
