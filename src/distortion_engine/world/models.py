from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distortion_engine.organization.models import AgentRole

type ActionKind = Literal[
    "work",
    "test",
    "remediate",
    "release",
    "delay_release",
    "reduce_scope",
    "investigate",
    "add_qa_capacity",
]
type RejectionCode = Literal[
    "tick_ordering_violation",
    "unauthorized_action",
    "item_required",
    "unknown_item",
    "dependency_blocked",
    "invalid_action_state",
    "unsupported_action",
]


class OperationalHarmMaxima(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_delay: float = Field(default=100, gt=0)
    escaped_defects: float = Field(default=10, gt=0)
    incident: float = Field(default=100, gt=0)
    remediation: float = Field(default=100, gt=0)
    scope_loss: float = Field(default=1, gt=0)


class WorkItemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    department: str = Field(min_length=1)
    business_value: float = Field(gt=0)
    effort: float = Field(gt=0)
    deadline_tick: int = Field(ge=1)
    dependencies: tuple[str, ...] = ()


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_ticks: int = Field(ge=1)
    shock_tick: int = Field(ge=1)
    shock_item_id: str
    shock_severity: float = Field(gt=0, le=10)
    work_items: tuple[WorkItemConfig, ...]
    harm_maxima: OperationalHarmMaxima = Field(default_factory=OperationalHarmMaxima)

    @model_validator(mode="after")
    def validate_scenario(self) -> "ScenarioConfig":
        ids = {item.id for item in self.work_items}
        if len(ids) != len(self.work_items) or self.shock_item_id not in ids:
            raise ValueError("scenario requires unique work items and a known shock item")
        if self.shock_tick > self.max_ticks:
            raise ValueError("shock tick must occur during the scenario")
        if any(
            dependency not in ids for item in self.work_items for dependency in item.dependencies
        ):
            raise ValueError("work item dependency must exist")
        remaining = {item.id: set(item.dependencies) for item in self.work_items}
        while remaining:
            ready = {item_id for item_id, dependencies in remaining.items() if not dependencies}
            if not ready:
                raise ValueError("work item dependencies must be acyclic")
            for item_id in ready:
                del remaining[item_id]
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return self


class WorkItemState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    department: str
    business_value: float
    initial_effort: float
    effort_remaining: float
    deadline_tick: int
    dependencies: tuple[str, ...] = ()
    latent_defect_severity: float = 0.0
    discovered_defect_severity: float = 0.0
    released: bool = False
    removed_scope: bool = False

    @property
    def complete(self) -> bool:
        return self.removed_scope or self.effort_remaining <= 0


class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tick: int
    items: tuple[WorkItemState, ...]
    seed: int
    seed_tape: tuple[float, ...]
    incident_severity: float = 0.0
    incident_duration: int = 0
    remediation_cost: float = 0.0
    staffing_cost: float = 0.0

    def unresolved_dependencies(self, item: WorkItemState) -> tuple[str, ...]:
        by_id = {candidate.id: candidate for candidate in self.items}
        return tuple(
            dependency for dependency in item.dependencies if not by_id[dependency].complete
        )


class WorldAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ActionKind
    item_id: str | None = None
    amount: float = Field(default=0.0, ge=0)


class ActorAction(BaseModel):
    """A policy action bound to the authoritative organization actor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: str = Field(min_length=1)
    actor_role: AgentRole
    action: WorldAction


class TransitionRejection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: RejectionCode
    reason: str = Field(min_length=1)
    actor_id: str | None = None
    actor_role: AgentRole | None = None
    action_kind: ActionKind | None = None
    item_id: str | None = None


class OperationalHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    progress: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    schedule: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)

    @property
    def score(self) -> float:
        return (
            0.20 * self.progress
            + 0.30 * self.quality
            + 0.20 * self.schedule
            + 0.30 * self.reliability
        )


class TransitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: WorldState
    events: tuple[str, ...] = ()
    rejections: tuple[TransitionRejection, ...] = ()
