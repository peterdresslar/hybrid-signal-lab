"""Typer CLI for the Colony testbed."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="colony",
    help="Colony: collective control testbed for multi-agent LLM systems.",
)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task prompt for the agents."),
    agents: int = typer.Option(5, "--agents", "-n", help="Total number of agents (N)."),
    batch_size: int = typer.Option(
        5, "--batch-size", "-k", help="Agents per round (K)."
    ),
    rounds: int = typer.Option(
        0,
        "--rounds",
        "-r",
        help="Number of rounds to run. 0 = one full cycle (every agent gets a turn).",
    ),
    model: str = typer.Option(
        "qwen3.5:2b", "--model", "-m", help="Ollama model name."
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Base sampling temperature."
    ),
    temp_spread: float = typer.Option(
        0.2,
        "--temp-spread",
        help="Temperature variation across agents. "
        "Agent i gets temperature + spread * (i / (N-1) - 0.5).",
    ),
    decay_rate: float = typer.Option(
        0.95, "--decay-rate", "-d", help="Signal decay rate per second."
    ),
    signal_dim: int = typer.Option(
        16, "--signal-dim", help="Dimensionality of signal vectors."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path to write results JSON."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
):
    """Run the colony with the given configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from colony.agent import OllamaAgent
    from colony.buffer import SignalBuffer
    from colony.colony import Colony

    # create agents with varied temperatures (homogeneous consortium
    # with controlled temperature variation, following Ghosh et al.)
    agent_list = []
    for i in range(agents):
        if agents > 1:
            t = temperature + temp_spread * (i / (agents - 1) - 0.5)
        else:
            t = temperature

        agent = OllamaAgent(
            agent_id=f"agent-{i:03d}",
            model=model,
            temperature=t,
            signal_dim=signal_dim,
        )
        agent_list.append(agent)

    buffer = SignalBuffer(decay_rate=decay_rate)
    colony = Colony(agents=agent_list, buffer=buffer, batch_size=batch_size)

    typer.echo(
        f"Colony initialized: N={colony.n_agents}, K={colony.batch_size}, "
        f"decay={decay_rate}, model={model}"
    )

    if rounds == 0:
        results = colony.full_cycle(prompt)
    else:
        results = colony.run(prompt, rounds=rounds)

    # summarize
    for r in results:
        typer.echo(f"\nRound {r.round_number} [{', '.join(r.batch)}]:")
        for agent_id, resp in r.responses.items():
            preview = resp.text[:80].replace("\n", " ")
            typer.echo(
                f"  {agent_id}: entropy={resp.signal.entropy:.3f} "
                f'"{preview}..."'
            )

    # optionally write full results
    if output:
        out_data = []
        for r in results:
            round_data = {
                "round": r.round_number,
                "batch": r.batch,
                "responses": {
                    aid: {
                        "text": resp.text,
                        "entropy": resp.signal.entropy,
                        "signal_vector": resp.signal.vector.tolist(),
                        "timestamp": resp.signal.timestamp,
                    }
                    for aid, resp in r.responses.items()
                },
            }
            out_data.append(round_data)

        output.write_text(json.dumps(out_data, indent=2))
        typer.echo(f"\nResults written to {output}")


@app.command()
def status():
    """Check that dependencies are available."""
    import shutil

    typer.echo("Checking dependencies...")

    # ollama
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        typer.echo(f"  ollama: {ollama_bin}")
    else:
        typer.echo("  ollama: NOT FOUND (install from https://ollama.com)")

    # ollama python package
    try:
        import ollama

        typer.echo(f"  ollama (python): {ollama.__version__}")
    except ImportError:
        typer.echo("  ollama (python): NOT FOUND (uv add ollama)")

    # numpy
    try:
        import numpy

        typer.echo(f"  numpy: {numpy.__version__}")
    except ImportError:
        typer.echo("  numpy: NOT FOUND (uv add numpy)")

    typer.echo("Done.")


def main():
    app()
