from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from distortion_engine.events.artifacts import RunArtifactMetrics
from distortion_engine.events.models import Event, RunManifest
from distortion_engine.experiments.runner import ExperimentRequest
from distortion_engine.simulation.runner import RunRequest


class LaunchExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: ExperimentRequest
    policy: Literal["fixture", "record", "locked"] = "fixture"
    model: str = Field(default="", max_length=100)


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    experiment: str
    status: Literal["queued", "running", "completed", "failed"]
    completed_runs: int
    failed_runs: int
    total_runs: int
    error: str | None = None


class ArtifactErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    artifact: str
    message: str


class ArtifactErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: ArtifactErrorDetail


class MessageErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str


class ValidationErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    detail: list[dict[str, Any]]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok"]


class RunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: RunManifest
    request: RunRequest
    metrics: RunArtifactMetrics


class EvidenceFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    department: str | None
    depth: int | None
    tick: int | None
    kind: str | None
    actor_id: str | None


class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    kind: str
    tick: int
    actor_id: str | None
    department: str
    depth: int | None
    causes: tuple[int, ...]
    evidence_refs: tuple[Any, ...]
    event: Event


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filters: EvidenceFilters
    nodes: tuple[EvidenceNode, ...]


class ExperimentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    analysis: dict[str, Any]


class ExperimentDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    analysis: dict[str, Any]
    request: ExperimentRequest | None
    runs: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
