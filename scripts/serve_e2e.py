from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import uvicorn

from distortion_engine.api.app import create_app
from distortion_engine.config import load_experiment_request
from distortion_engine.events.store import FileEventStore
from distortion_engine.experiments.runner import ExperimentRunner
from distortion_engine.policy.fixture import FixturePolicy

ROOT = Path(__file__).resolve().parents[1]


async def prepare(artifacts: Path) -> None:
    request = load_experiment_request(ROOT / "configs" / "e2e.yaml")
    await ExperimentRunner().run(request, FixturePolicy(), FileEventStore(artifacts))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="distortion-e2e-") as temporary:
        artifacts = Path(temporary)
        asyncio.run(prepare(artifacts))
        uvicorn.run(create_app(artifacts), host="127.0.0.1", port=4173)
