from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    kind: str
    tick: int
    actor_id: str | None = None
    causes: tuple[int, ...] = ()
    payload: dict[str, Any]


class DecisionLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    agent_id: str
    tick: int
    context_hash: str
    policy: str
    decision: dict[str, Any]


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    seed: int
    request_hash: str
    event_hash: str
    decision_hash: str
    metrics_hash: str
    event_count: int
    decision_count: int
    finalized: bool
    policy: str
    policy_fingerprint: str
    engine_fingerprint: str
