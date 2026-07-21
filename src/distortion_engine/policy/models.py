from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from distortion_engine.domain.reports import Report
from distortion_engine.observation.engine import Observation
from distortion_engine.organization.models import AgentConfig
from distortion_engine.world.models import OperationalHealth, WorldAction


class AgentMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turns: int = 0
    concerns: tuple[str, ...] = ()
    trust: dict[str, float] = Field(default_factory=dict)


class VerificationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reported_by: str
    scope: tuple[str, ...]
    observed_health: OperationalHealth


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: AgentConfig
    depth: int
    tick: int
    scope: tuple[str, ...]
    observation: Observation
    incoming_reports: tuple[Report, ...] = ()
    verification_evidence: tuple[VerificationEvidence, ...] = ()
    memory: AgentMemory = AgentMemory()
    incentive_pressure: float = Field(ge=0, le=1)
    attention_budget: int = Field(ge=0)
    release_window_open: bool = False


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report: Report
    actions: tuple[WorldAction, ...] = ()
    memory: AgentMemory
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """The model-facing schema; identity and scope are supplied by the engine."""

    model_config = ConfigDict(extra="forbid")

    health: OperationalHealth
    confidence: float = Field(ge=0, le=1)
    escalate: bool
    explanation: str = Field(min_length=1, max_length=800)
    actions: tuple[WorldAction, ...] = ()
    concerns: tuple[str, ...] = Field(default=(), max_length=8)


class AgentPolicy(Protocol):
    name: str

    async def decide(self, context: AgentContext) -> PolicyDecision: ...


@runtime_checkable
class TruthAwarePolicy(Protocol):
    name: str

    async def decide_with_truth(
        self, context: AgentContext, truth: OperationalHealth
    ) -> PolicyDecision: ...
