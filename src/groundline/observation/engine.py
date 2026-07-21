from pydantic import BaseModel, ConfigDict

from groundline.organization.models import AgentConfig
from groundline.world.models import WorldState


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    tick: int
    scope: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    effort_remaining: float | None = None
    schedule_risk: float | None = None
    discovered_defect_severity: float | None = None
    detected_defect_severity: float | None = None
    detected_defects: dict[str, float] | None = None
    unresolved_dependencies: dict[str, tuple[str, ...]] | None = None
    incident_severity: float | None = None


class ObservationEngine:
    def observe(self, state: WorldState, agent: AgentConfig, scope: tuple[str, ...]) -> Observation:
        items = [item for item in state.items if item.id in scope and not item.removed_scope]
        if not items:
            raise ValueError("observation scope contains no active work items")
        refs = tuple(f"world:{state.tick}:item:{item.id}" for item in items)
        if agent.role != "contributor":
            return Observation(
                agent_id=agent.id,
                tick=state.tick,
                scope=scope,
                evidence_refs=refs,
                incident_severity=state.incident_severity,
            )
        if agent.department == "Product":
            risk = sum(
                max(0.0, item.effort_remaining - max(0, item.deadline_tick - state.tick))
                for item in items
            )
            return Observation(
                agent_id=agent.id,
                tick=state.tick,
                scope=scope,
                evidence_refs=refs,
                schedule_risk=risk,
                incident_severity=state.incident_severity,
            )
        if agent.department == "Engineering":
            unresolved = {
                item.id: dependencies
                for item in items
                if (dependencies := state.unresolved_dependencies(item))
            }
            return Observation(
                agent_id=agent.id,
                tick=state.tick,
                scope=scope,
                evidence_refs=refs,
                effort_remaining=sum(item.effort_remaining for item in items),
                discovered_defect_severity=sum(item.discovered_defect_severity for item in items),
                unresolved_dependencies=unresolved,
                incident_severity=state.incident_severity,
            )
        if agent.department == "QA":
            detection = agent.skills.get("defect_detection", 0.5)
            detected_by_item = {
                item.id: item.latent_defect_severity
                for item in items
                if detection >= 0.5 and item.latent_defect_severity > 0
            }
            detected = sum(detected_by_item.values())
            return Observation(
                agent_id=agent.id,
                tick=state.tick,
                scope=scope,
                evidence_refs=refs,
                discovered_defect_severity=sum(item.discovered_defect_severity for item in items),
                detected_defect_severity=detected,
                detected_defects=detected_by_item,
                incident_severity=state.incident_severity,
            )
        return Observation(
            agent_id=agent.id,
            tick=state.tick,
            scope=scope,
            evidence_refs=refs,
            incident_severity=state.incident_severity,
        )
