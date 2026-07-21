from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from distortion_engine import __version__
from distortion_engine.api.jobs import ExperimentJobManager, ExperimentLaunchConflict
from distortion_engine.api.models import (
    ArtifactErrorResponse,
    EvidenceResponse,
    ExperimentDetailResponse,
    ExperimentSummaryResponse,
    HealthResponse,
    JobStatus,
    LaunchExperiment,
    MessageErrorResponse,
    RunDetailResponse,
    ValidationErrorResponse,
)
from distortion_engine.config import load_env
from distortion_engine.events.artifacts import ArtifactCorruptError, verify_run_artifacts
from distortion_engine.events.models import Event, RunManifest
from distortion_engine.experiments.runner import ExperimentRequest


class ArtifactRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def _json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ArtifactCorruptError(
                "unreadable_artifact", path.name, f"cannot read artifact {path.name}"
            ) from error
        except json.JSONDecodeError as error:
            raise ArtifactCorruptError(
                "invalid_json", path.name, f"artifact {path.name} is not valid JSON"
            ) from error

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, Any]]:
        try:
            values = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        except OSError as error:
            raise ArtifactCorruptError(
                "unreadable_artifact", path.name, f"cannot read artifact {path.name}"
            ) from error
        except json.JSONDecodeError as error:
            raise ArtifactCorruptError(
                "invalid_json", path.name, f"artifact {path.name} is not valid JSON"
            ) from error
        if any(not isinstance(value, dict) for value in values):
            raise ArtifactCorruptError(
                "invalid_structure",
                path.name,
                f"artifact {path.name} has an invalid structure",
            )
        return cast(list[dict[str, Any]], values)

    @classmethod
    def _object(cls, path: Path) -> dict[str, Any]:
        value = cls._json(path)
        if not isinstance(value, dict):
            raise ArtifactCorruptError(
                "invalid_structure",
                path.name,
                f"artifact {path.name} has an invalid structure",
            )
        return value

    def run_directories(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if not self.root.exists():
            return found
        for manifest_path in self.root.rglob("manifest.json"):
            if "quarantine" in manifest_path.parts:
                continue
            artifacts = verify_run_artifacts(manifest_path.parent)
            found[artifacts.manifest.run_id] = artifacts.directory
        return found

    def list_runs(self) -> list[dict[str, Any]]:
        manifests = [
            verify_run_artifacts(path).manifest.model_dump(mode="json")
            for path in self.run_directories().values()
        ]
        return sorted(manifests, key=lambda manifest: manifest["run_id"])

    def run(self, run_id: str) -> Path:
        directory = self.run_directories().get(run_id)
        if directory is None:
            raise KeyError(run_id)
        return directory

    def experiments(self) -> list[dict[str, Any]]:
        experiments_root = self.root / "experiments"
        if not experiments_root.exists():
            return []
        return [
            {"name": directory.name, "analysis": self._object(directory / "analysis.json")}
            for directory in sorted(experiments_root.iterdir())
            if directory.is_dir() and (directory / "analysis.json").exists()
        ]

    def experiment(self, name: str) -> dict[str, Any]:
        experiments_root = (self.root / "experiments").resolve()
        directory = (experiments_root / name).resolve()
        if directory.parent != experiments_root:
            raise KeyError(name)
        if not directory.is_dir() or not (directory / "analysis.json").exists():
            raise KeyError(name)
        rows = self._jsonl(directory / "run-index.jsonl")
        completed = [row for row in rows if row.get("status", "completed") == "completed"]
        failures = [row for row in rows if row.get("status") == "failed"]
        request_path = directory / "experiment-request.json"
        request = self.experiment_request(name) if request_path.exists() else None
        return {
            "name": name,
            "analysis": self._object(directory / "analysis.json"),
            "request": request.model_dump(mode="json") if request is not None else None,
            "runs": completed,
            "failures": failures,
        }

    def experiment_request(self, name: str) -> ExperimentRequest | None:
        experiments_root = (self.root / "experiments").resolve()
        directory = (experiments_root / name).resolve()
        if directory.parent != experiments_root:
            raise KeyError(name)
        path = directory / "experiment-request.json"
        if not path.exists():
            return None
        try:
            return ExperimentRequest.model_validate(self._json(path))
        except ValidationError as error:
            raise ArtifactCorruptError(
                "invalid_structure",
                path.name,
                f"artifact {path.name} has an invalid structure",
            ) from error


def create_app(artifacts: Path = Path("artifacts")) -> FastAPI:
    load_env()
    repository = ArtifactRepository(artifacts)
    jobs = ExperimentJobManager(artifacts)
    app = FastAPI(title="The Distortion Engine API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.router.add_event_handler("startup", jobs.recover)

    @app.exception_handler(ArtifactCorruptError)
    async def corrupt_artifact_handler(_: Any, error: ArtifactCorruptError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": error.code,
                    "artifact": error.filename,
                    "message": error.message,
                }
            },
        )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/api/runs",
        response_model=list[RunManifest],
        responses={500: {"model": ArtifactErrorResponse}},
    )
    def list_runs() -> list[dict[str, Any]]:
        return repository.list_runs()

    @app.get(
        "/api/runs/{run_id}",
        response_model=RunDetailResponse,
        responses={
            404: {"model": MessageErrorResponse},
            500: {"model": ArtifactErrorResponse},
        },
    )
    def run_detail(run_id: str) -> dict[str, Any]:
        try:
            directory = repository.run(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error
        artifacts = verify_run_artifacts(directory)
        return {
            "manifest": artifacts.manifest.model_dump(mode="json"),
            "request": artifacts.request.model_dump(mode="json"),
            "metrics": artifacts.metrics.model_dump(mode="json"),
        }

    @app.get(
        "/api/runs/{run_id}/timeline",
        response_model=list[Event],
        responses={
            404: {"model": MessageErrorResponse},
            422: {"model": ValidationErrorResponse},
            500: {"model": ArtifactErrorResponse},
        },
    )
    def run_timeline(
        run_id: str,
        department: str | None = None,
        depth: int | None = Query(default=None, ge=0),
        tick: int | None = Query(default=None, ge=0),
        kind: str | None = None,
        actor_id: str | None = None,
    ) -> list[dict[str, Any]]:
        directory = _run_directory(repository, run_id)
        nodes = _evidence_nodes(directory)
        return [
            node["event"]
            for node in nodes
            if _matches(node, department, depth, tick, kind, actor_id)
        ]

    @app.get(
        "/api/runs/{run_id}/evidence",
        response_model=EvidenceResponse,
        responses={
            404: {"model": MessageErrorResponse},
            422: {"model": ValidationErrorResponse},
            500: {"model": ArtifactErrorResponse},
        },
    )
    def run_evidence(
        run_id: str,
        department: str | None = None,
        depth: int | None = Query(default=None, ge=0),
        tick: int | None = Query(default=None, ge=0),
        kind: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        directory = _run_directory(repository, run_id)
        nodes = [
            node
            for node in _evidence_nodes(directory)
            if _matches(node, department, depth, tick, kind, actor_id)
        ]
        return {
            "filters": {
                "department": department,
                "depth": depth,
                "tick": tick,
                "kind": kind,
                "actor_id": actor_id,
            },
            "nodes": nodes,
        }

    @app.get(
        "/api/experiments",
        response_model=list[ExperimentSummaryResponse],
        responses={500: {"model": ArtifactErrorResponse}},
    )
    def list_experiments() -> list[dict[str, Any]]:
        return repository.experiments()

    @app.get(
        "/api/experiments/{name}",
        response_model=ExperimentDetailResponse,
        responses={
            404: {"model": MessageErrorResponse},
            500: {"model": ArtifactErrorResponse},
        },
    )
    def experiment_detail(name: str) -> dict[str, Any]:
        try:
            return repository.experiment(name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="experiment not found") from error

    @app.post(
        "/api/experiments",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobStatus,
        responses={
            409: {"model": MessageErrorResponse},
            422: {"model": ValidationErrorResponse},
            500: {"model": ArtifactErrorResponse},
            503: {"model": MessageErrorResponse},
        },
    )
    async def launch_experiment(launch: LaunchExperiment) -> JobStatus:
        existing_request = repository.experiment_request(launch.experiment.name)
        if existing_request is not None and existing_request != launch.experiment:
            raise HTTPException(
                status_code=409,
                detail="experiment name already belongs to a different request",
            )
        if jobs.conflicts(launch):
            raise HTTPException(
                status_code=409,
                detail="experiment name already belongs to a different launch",
            )
        if launch.policy != "fixture":
            effective_model = (launch.model or "").strip() or os.environ.get(
                "DISTORTION_MODEL", ""
            ).strip()
            if not effective_model:
                raise HTTPException(
                    status_code=503,
                    detail="model is required: supply request body or set DISTORTION_MODEL",
                )
        try:
            return await jobs.launch(launch)
        except ExperimentLaunchConflict as error:
            raise HTTPException(
                status_code=409,
                detail="experiment name already belongs to a different launch",
            ) from error

    @app.get(
        "/api/jobs/{job_id}",
        response_model=JobStatus,
        responses={404: {"model": MessageErrorResponse}},
    )
    async def job_status(job_id: str) -> JobStatus:
        await jobs.recover()
        try:
            return jobs.snapshot(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    frontend = next(
        (
            candidate
            for candidate in (
                Path(__file__).resolve().parents[1] / "frontend",
                Path(__file__).resolve().parents[3] / "frontend" / "dist",
            )
            if (candidate / "index.html").is_file()
        ),
        None,
    )
    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


def _run_directory(repository: ArtifactRepository, run_id: str) -> Path:
    try:
        return repository.run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error


def _evidence_nodes(directory: Path) -> list[dict[str, Any]]:
    artifacts = verify_run_artifacts(directory)
    agents = artifacts.request.organization.by_id
    depths = {metric.agent_id: metric.depth for metric in artifacts.metrics.distortion}
    nodes = []
    for typed_event in artifacts.events:
        event = typed_event.model_dump(mode="json")
        agent = agents.get(typed_event.actor_id)
        nodes.append(
            {
                "sequence": typed_event.sequence,
                "kind": typed_event.kind,
                "tick": typed_event.tick,
                "actor_id": typed_event.actor_id,
                "department": agent.department if agent else "World",
                "depth": (
                    depths.get(typed_event.actor_id)
                    if agent and typed_event.actor_id is not None
                    else None
                ),
                "causes": list(typed_event.causes),
                "evidence_refs": typed_event.payload.get("evidence_refs", []),
                "event": event,
            }
        )
    return nodes


def _matches(
    node: dict[str, Any],
    department: str | None,
    depth: int | None,
    tick: int | None,
    kind: str | None,
    actor_id: str | None,
) -> bool:
    return (
        (department is None or node["department"] == department)
        and (depth is None or node["depth"] == depth)
        and (tick is None or node["tick"] == tick)
        and (kind is None or node["kind"] == kind)
        and (actor_id is None or node["actor_id"] == actor_id)
    )


# Module-level ASGI app so `uvicorn distortion_engine.api.app:app` (used by the
# TUI launcher's "Launch Web UI" and "Backend only" actions) resolves. The CLI
# `serve` command builds its own instance via create_app(artifacts).
app = create_app()
