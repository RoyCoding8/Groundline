from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, ValidationError

from distortion_engine.events.models import DecisionLedgerEntry, Event, RunManifest
from distortion_engine.events.store import canonical_hash, canonical_json  # noqa: F401
from distortion_engine.metrics.engine import DistortionMetric
from distortion_engine.metrics.outcomes import RunOutcomes, calculate_run_outcomes

if TYPE_CHECKING:
    from distortion_engine.simulation.runner import RunRequest

ArtifactErrorCode = Literal[
    "missing_artifact",
    "unreadable_artifact",
    "invalid_json",
    "invalid_structure",
    "unfinalized_manifest",
    "identity_mismatch",
    "hash_mismatch",
    "count_mismatch",
    "sequence_mismatch",
    "provenance_mismatch",
    "content_mismatch",
]


class ArtifactCorruptError(RuntimeError):
    def __init__(self, code: ArtifactErrorCode, filename: str, message: str) -> None:
        self.code = code
        self.filename = filename
        self.message = message
        super().__init__(message)


class RunArtifactMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    distortion: tuple[DistortionMetric, ...]
    outcomes: RunOutcomes


class VerifiedRunArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    directory: Path
    manifest: RunManifest
    request: Any
    events: tuple[Event, ...]
    decisions: tuple[DecisionLedgerEntry, ...]
    metrics: RunArtifactMetrics


def verify_run_artifacts(
    directory: Path,
    *,
    expected_request: RunRequest | None = None,
    expected_policy_fingerprint: str | None = None,
    expected_engine_fingerprint: str | None = None,
) -> VerifiedRunArtifacts:
    from distortion_engine.simulation.runner import RunRequest

    directory = directory.resolve()
    manifest = _model_file(directory / "manifest.json", RunManifest)
    if not manifest.finalized:
        _raise("unfinalized_manifest", "manifest.json", "run manifest is not finalized")
    if directory.name != manifest.run_id:
        _raise("identity_mismatch", "manifest.json", "run directory does not match manifest run_id")

    request = _model_file(directory / "request.json", RunRequest)
    events, event_bytes = _jsonl_models(directory / "events.jsonl", Event)
    decisions, decision_bytes = _jsonl_models(directory / "decisions.jsonl", DecisionLedgerEntry)
    metrics = _model_file(directory / "metrics.json", RunArtifactMetrics)

    request_hash = canonical_hash(request.model_dump(mode="json"))
    if request_hash != manifest.request_hash:
        _raise("hash_mismatch", "request.json", "request hash does not match manifest")
    if request.seed != manifest.seed:
        _raise("identity_mismatch", "request.json", "request seed does not match manifest")
    if expected_request is not None and request != expected_request:
        _raise("identity_mismatch", "request.json", "request does not match expected run request")
    if (
        expected_policy_fingerprint is not None
        and manifest.policy_fingerprint != expected_policy_fingerprint
    ):
        _raise(
            "identity_mismatch",
            "manifest.json",
            "policy fingerprint does not match expected policy",
        )
    if (
        expected_engine_fingerprint is not None
        and manifest.engine_fingerprint != expected_engine_fingerprint
    ):
        _raise(
            "identity_mismatch",
            "manifest.json",
            "engine fingerprint does not match expected engine",
        )

    if hashlib.sha256(event_bytes).hexdigest() != manifest.event_hash:
        _raise("hash_mismatch", "events.jsonl", "event ledger hash does not match manifest")
    if hashlib.sha256(decision_bytes).hexdigest() != manifest.decision_hash:
        _raise(
            "hash_mismatch",
            "decisions.jsonl",
            "decision ledger hash does not match manifest",
        )
    if canonical_hash(metrics.model_dump(mode="json")) != manifest.metrics_hash:
        _raise("hash_mismatch", "metrics.json", "metrics hash does not match manifest")

    if len(events) != manifest.event_count:
        _raise("count_mismatch", "events.jsonl", "event count does not match manifest")
    if len(decisions) != manifest.decision_count:
        _raise(
            "count_mismatch",
            "decisions.jsonl",
            "decision count does not match manifest",
        )
    _verify_event_sequence(events)
    _verify_decision_sequence(decisions, manifest.policy)

    from distortion_engine.policy.models import PolicyDecision

    try:
        for decision in decisions:
            PolicyDecision.model_validate(decision.decision)
    except ValidationError as error:
        _raise(
            "invalid_structure",
            "decisions.jsonl",
            "decision payload has an invalid structure",
            error,
        )

    try:
        metric_events = tuple(
            DistortionMetric.model_validate(event.payload)
            for event in events
            if event.kind == "metric"
        )
    except ValidationError as error:
        _raise(
            "invalid_structure",
            "events.jsonl",
            "metric event payload has an invalid structure",
            error,
        )
    if metric_events != metrics.distortion:
        _raise(
            "content_mismatch",
            "metrics.json",
            "distortion metrics do not match metric ledger events",
        )
    try:
        reconstructed_outcomes = calculate_run_outcomes(
            scenario=request.scenario,
            organization=request.organization,
            events=events,
            metrics=metrics.distortion,
            specification=request.outcome_specification,
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        _raise(
            "invalid_structure",
            "events.jsonl",
            "event payloads cannot reconstruct canonical run outcomes",
            error,
        )
    if reconstructed_outcomes != metrics.outcomes:
        _raise(
            "content_mismatch",
            "metrics.json",
            "run outcomes do not match canonical events and metrics",
        )

    return VerifiedRunArtifacts(
        directory=directory,
        manifest=manifest,
        request=request,
        events=events,
        decisions=decisions,
        metrics=metrics,
    )


def _model_file(path: Path, model: type[BaseModel]) -> Any:
    raw = _read_bytes(path)
    try:
        return model.model_validate_json(raw)
    except json.JSONDecodeError as error:
        _raise("invalid_json", path.name, f"artifact {path.name} is not valid JSON", error)
    except ValidationError as error:
        code: ArtifactErrorCode = (
            "invalid_json"
            if any(detail["type"] == "json_invalid" for detail in error.errors())
            else "invalid_structure"
        )
        description = "is not valid JSON" if code == "invalid_json" else "has an invalid structure"
        _raise(code, path.name, f"artifact {path.name} {description}", error)


def _jsonl_models(path: Path, model: type[BaseModel]) -> tuple[tuple[Any, ...], bytes]:
    raw = _read_bytes(path)
    rows: list[Any] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _raise("unreadable_artifact", path.name, f"cannot decode artifact {path.name}", error)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            _raise(
                "invalid_structure",
                path.name,
                f"artifact {path.name} contains an empty row at line {line_number}",
            )
        try:
            rows.append(model.model_validate_json(line))
        except json.JSONDecodeError as error:
            _raise(
                "invalid_json",
                path.name,
                f"artifact {path.name} contains invalid JSON at line {line_number}",
                error,
            )
        except ValidationError as error:
            code: ArtifactErrorCode = (
                "invalid_json"
                if any(detail["type"] == "json_invalid" for detail in error.errors())
                else "invalid_structure"
            )
            description = "invalid JSON" if code == "invalid_json" else "an invalid row"
            _raise(
                code,
                path.name,
                f"artifact {path.name} contains {description} at line {line_number}",
                error,
            )
    return tuple(rows), raw


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        _raise("missing_artifact", path.name, f"artifact {path.name} is missing")
    try:
        return path.read_bytes()
    except OSError as error:
        _raise("unreadable_artifact", path.name, f"cannot read artifact {path.name}", error)


def _verify_event_sequence(events: tuple[Event, ...]) -> None:
    for expected, event in enumerate(events):
        if event.sequence != expected:
            _raise(
                "sequence_mismatch",
                "events.jsonl",
                "event sequences must be contiguous and zero-based",
            )
        if any(cause < 0 or cause >= event.sequence for cause in event.causes):
            _raise(
                "provenance_mismatch",
                "events.jsonl",
                "event causes must reference earlier events",
            )


def _verify_decision_sequence(decisions: tuple[DecisionLedgerEntry, ...], policy: str) -> None:
    for expected, decision in enumerate(decisions):
        if decision.sequence != expected:
            _raise(
                "sequence_mismatch",
                "decisions.jsonl",
                "decision sequences must be contiguous and zero-based",
            )
        if decision.policy != policy:
            _raise(
                "provenance_mismatch",
                "decisions.jsonl",
                "decision policy does not match manifest",
            )


def _raise(
    code: ArtifactErrorCode,
    filename: str,
    message: str,
    cause: Exception | None = None,
) -> NoReturn:
    error = ArtifactCorruptError(code, filename, message)
    if cause is None:
        raise error
    raise error from cause
