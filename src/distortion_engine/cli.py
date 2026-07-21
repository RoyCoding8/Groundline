import asyncio
import json
from pathlib import Path

import typer
import uvicorn

from distortion_engine import __version__
from distortion_engine.config import (
    load_env,
    load_experiment_request,
    load_run_request,
    load_yaml,
)
from distortion_engine.events.store import FileEventStore
from distortion_engine.experiments.runner import ExperimentRequest, ExperimentRunner
from distortion_engine.policy.factory import build_policy
from distortion_engine.policy.models import AgentPolicy
from distortion_engine.replay.engine import ReplayEngine
from distortion_engine.simulation.runner import RunRequest, SimulationRunner

app = typer.Typer(
    help=f"The Distortion Engine {__version__}: computed truth, strategic reports.",
    no_args_is_help=True,
)


@app.command()
def validate(config: Path) -> None:
    """Validate a scenario or experiment configuration."""
    data = load_yaml(config)
    kind = "experiment" if "seeds" in data else "run"
    if kind == "experiment":
        ExperimentRequest.model_validate(data)
    else:
        RunRequest.model_validate(data)
    typer.echo(f"valid {kind}: {config}")


@app.command()
def run(
    config: Path = typer.Option(...),
    seed: int = typer.Option(...),
    policy: str = typer.Option("fixture"),
    artifacts: Path = typer.Option(Path("artifacts")),
    model: str | None = typer.Option(None),
) -> None:
    """Run one seeded company trajectory."""
    load_env()
    request = load_run_request(config, seed=seed)
    selected_policy = _policy(policy, model, artifacts)
    result = asyncio.run(
        SimulationRunner().run(request, selected_policy, FileEventStore(artifacts))
    )
    typer.echo(str(result.run_directory.resolve()))


@app.command()
def replay(run_directory: Path) -> None:
    """Replay a finalized Run without model calls."""
    result = asyncio.run(ReplayEngine().replay(run_directory))
    typer.echo(result.model_dump_json())
    if not result.equivalent:
        raise typer.Exit(1)


@app.command()
def experiment(
    config: Path = typer.Option(...),
    policy: str = typer.Option("fixture"),
    artifacts: Path = typer.Option(Path("artifacts")),
    model: str | None = typer.Option(None),
) -> None:
    """Execute a paired Experiment."""
    load_env()
    request = load_experiment_request(config)
    selected_policy = _policy(policy, model, artifacts)
    result = asyncio.run(
        ExperimentRunner().run(request, selected_policy, FileEventStore(artifacts))
    )
    typer.echo(str(result.analysis_path.resolve()))


@app.command()
def analyze(experiment_directory: Path) -> None:
    """Analyze a completed Experiment."""
    analysis = json.loads((experiment_directory / "analysis.json").read_text(encoding="utf-8"))
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))


@app.command()
def serve(
    artifacts: Path = typer.Option(Path("artifacts")),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the read-only query application."""
    load_env()
    from distortion_engine.api.app import create_app

    uvicorn.run(create_app(artifacts), host=host, port=port)


def _policy(name: str, model: str | None, artifacts: Path) -> AgentPolicy:
    try:
        return build_policy(name, model=model, artifacts=artifacts)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
