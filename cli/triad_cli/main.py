"""
TRIAD CLI — Beautiful terminal interface for the multi-agent system.
"""

import asyncio
import json
import os
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="TRIAD — Multi-Agent Consensus System CLI")
console = Console()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to deliberate on"),
    max_rounds: int = typer.Option(3, "--rounds", "-r", help="Maximum deliberation rounds"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream real-time progress"),
):
    """Submit a question to TRIAD."""
    if stream:
        asyncio.run(_stream_ask(question, max_rounds))
    else:
        _sync_ask(question, max_rounds)

@app.command()
def health():
    """Check TRIAD system health."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health")
        data = resp.json()
        
        table = Table(title="TRIAD System Health", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        
        table.add_row("Backend", "[green]OK[/green]" if data["status"] == "ok" else "[red]ERROR[/red]")
        table.add_row("Neo4j", f"[green]{data['neo4j']}[/green]" if "connected" in data["neo4j"] else f"[red]{data['neo4j']}[/red]")
        
        for agent, status in data["agents"].items():
            table.add_row(f"Agent: {agent.upper()}", status)
            
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error connecting to TRIAD:[/red] {e}")

async def _stream_ask(question: str, max_rounds: int):
    ws_url = BACKEND_URL.replace("http", "ws") + "/ws"
    
    console.print(Panel(f"[bold cyan]QUERY:[/bold cyan] {question}", box=box.DOUBLE))
    
    agent_status = {
        "axiom": {"pos": "", "conf": 0},
        "prism": {"pos": "", "conf": 0},
        "forge": {"pos": "", "conf": 0}
    }

    import websockets
    try:
        async with websockets.connect(ws_url) as websocket:
            await websocket.send(json.dumps({"question": question, "max_rounds": max_rounds}))
            
            with Live(console=console, refresh_per_second=4) as live:
                while True:
                    try:
                        msg = await websocket.recv()
                        event = json.loads(msg)
                        etype = event["type"]
                        data = event["data"]
                        
                        if etype == "round_start":
                            live.console.print(f"\n[bold yellow]─── ROUND {data['round']}/{data['max_rounds']} ───[/bold yellow]")
                        
                        elif etype == "agent_thinking":
                            live.console.print(f" ▸ [dim]{data['agent'].upper()} is thinking...[/dim]")
                        
                        elif etype == "agent_response":
                            agent = data["agent"]
                            agent_status[agent]["pos"] = data["position"]
                            agent_status[agent]["conf"] = data["confidence"]
                            
                            color = "cyan" if agent == "axiom" else "magenta" if agent == "prism" else "orange3"
                            live.console.print(f"[{color}][bold]{agent.upper()} ({(data['confidence']*100):.0f}%):[/bold] {data['position']}[/{color}]")
                        
                        elif etype == "consensus_reached":
                            status = data["status"].upper()
                            live.console.print(f"\n[bold green]◆ CONSENSUS REACHED: {status}[/bold green]")
                            live.console.print(f"[dim]Agreed: {', '.join(data['agreeing']).upper()}[/dim]")
                        
                        elif etype == "deadlock":
                            live.console.print("\n[bold red]✖ DEADLOCK: NO CONSENSUS REACHED[/bold red]")
                        
                        elif etype == "final_answer":
                            live.update(Panel(
                                Markdown(data["final_answer"] or "No consensus reached."),
                                title="[bold green]TRIAD FINAL STANCE[/bold green]",
                                border_style="green",
                                box=box.HEAVY
                            ))
                            break
                            
                    except websockets.ConnectionClosed:
                        break
    except Exception as e:
        console.print(f"[red]Connection error:[/red] {e}")

def _sync_ask(question: str, max_rounds: int):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="TRIAD is deliberating...", total=None)
        try:
            resp = httpx.post(f"{BACKEND_URL}/ask", json={"question": question, "max_rounds": max_rounds}, timeout=120)
            data = resp.json()
            
            console.print(Panel(
                Markdown(data["final_answer"] or "System entered deadlock. View Web UI for resolution."),
                title=f"[bold green]TRIAD OUTPUT ({data['status'].upper()})[/bold green]",
                subtitle=f"Deliberated for {data['duration_seconds']}s",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    app()
