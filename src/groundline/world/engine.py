import random

from groundline.organization.models import AgentRole
from groundline.world.models import (
    ActionKind,
    ActorAction,
    OperationalHealth,
    RejectionCode,
    ScenarioConfig,
    TransitionRejection,
    TransitionResult,
    WorkItemState,
    WorldState,
)

_ALLOWED_ACTIONS: dict[AgentRole, frozenset[ActionKind]] = {
    "contributor": frozenset({"work", "test"}),
    "manager": frozenset(),
    "executive": frozenset(
        {
            "remediate",
            "release",
            "delay_release",
            "reduce_scope",
            "investigate",
            "add_qa_capacity",
        }
    ),
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _reject(
    state: WorldState,
    attributed: ActorAction,
    code: RejectionCode,
    reason: str,
) -> TransitionResult:
    return TransitionResult(
        state=state,
        rejections=(
            TransitionRejection(
                code=code,
                reason=reason,
                actor_id=attributed.actor_id,
                actor_role=attributed.actor_role,
                action_kind=attributed.action.kind,
                item_id=attributed.action.item_id,
            ),
        ),
    )


class WorldEngine:
    def initialize(self, scenario: ScenarioConfig, seed: int) -> WorldState:
        rng = random.Random(seed)
        seed_tape = tuple(rng.random() for _ in range(64))
        items = tuple(
            WorkItemState(
                id=item.id,
                department=item.department,
                business_value=item.business_value,
                initial_effort=item.effort,
                effort_remaining=item.effort,
                deadline_tick=item.deadline_tick,
                dependencies=item.dependencies,
            )
            for item in scenario.work_items
        )
        return WorldState(tick=0, items=items, seed=seed, seed_tape=seed_tape)

    def advance(
        self, state: WorldState, tick: int, scenario: ScenarioConfig | None = None
    ) -> TransitionResult:
        if tick != state.tick + 1:
            return TransitionResult(
                state=state,
                rejections=(
                    TransitionRejection(
                        code="tick_ordering_violation",
                        reason="tick ordering violation",
                    ),
                ),
            )
        items = state.items
        events: list[str] = []
        if scenario is not None and tick == scenario.shock_tick:
            shock_draw = state.seed_tape[(tick - 1) % len(state.seed_tape)]
            realized_shock = min(10.0, scenario.shock_severity * (0.8 + 0.4 * shock_draw))
            items = tuple(
                item.model_copy(
                    update={"latent_defect_severity": item.latent_defect_severity + realized_shock}
                )
                if item.id == scenario.shock_item_id
                else item
                for item in items
            )
            events.append(f"shock:{scenario.shock_item_id}:{realized_shock}")
        incident = state.incident_severity
        if any(item.released for item in items):
            incident = min(
                10.0,
                sum(
                    item.latent_defect_severity + item.discovered_defect_severity
                    for item in items
                    if item.released
                ),
            )
        return TransitionResult(
            state=state.model_copy(
                update={
                    "tick": tick,
                    "items": items,
                    "incident_severity": incident,
                    "incident_duration": state.incident_duration + (1 if incident > 0 else 0),
                }
            ),
            events=tuple(events),
        )

    def apply(self, state: WorldState, actions: tuple[ActorAction, ...]) -> TransitionResult:
        for attributed in actions:
            if attributed.action.kind not in _ALLOWED_ACTIONS[attributed.actor_role]:
                return _reject(
                    state,
                    attributed,
                    "unauthorized_action",
                    f"{attributed.actor_role} cannot perform {attributed.action.kind}",
                )
        current = state
        events: list[str] = []
        for attributed in actions:
            result = self._apply_one(current, attributed)
            if result.rejections:
                return TransitionResult(state=state, rejections=result.rejections)
            current = result.state
            events.extend(result.events)
        return TransitionResult(state=current, events=tuple(events))

    def _apply_one(self, state: WorldState, attributed: ActorAction) -> TransitionResult:
        action = attributed.action
        if action.kind == "delay_release":
            return TransitionResult(state=state, events=("release_delayed",))
        if action.kind == "add_qa_capacity":
            return TransitionResult(
                state=state.model_copy(
                    update={"staffing_cost": state.staffing_cost + action.amount}
                ),
                events=(f"qa_capacity:{action.amount}",),
            )
        if action.kind == "investigate":
            return TransitionResult(state=state, events=("investigation_started",))
        if action.kind == "release":
            blocked: list[tuple[str, tuple[str, ...]]] = []
            for item in state.items:
                dependencies = state.unresolved_dependencies(item)
                if not item.removed_scope and dependencies:
                    blocked.append((item.id, dependencies))
            if blocked:
                item_id, dependencies = blocked[0]
                return _reject(
                    state,
                    attributed,
                    "dependency_blocked",
                    f"{item_id} is blocked by incomplete dependencies: {', '.join(dependencies)}",
                )
            released_items = tuple(
                item.model_copy(update={"released": True}) if not item.removed_scope else item
                for item in state.items
            )
            return TransitionResult(
                state=state.model_copy(update={"items": released_items}),
                events=("company_release",),
            )
        if action.item_id is None:
            return _reject(
                state,
                attributed,
                "item_required",
                f"{action.kind} requires an item",
            )
        index = next((i for i, item in enumerate(state.items) if item.id == action.item_id), None)
        if index is None:
            return _reject(
                state,
                attributed,
                "unknown_item",
                f"unknown item {action.item_id}",
            )
        item = state.items[index]
        update: dict[str, float | bool] = {}
        event_amount = action.amount
        if action.kind == "work":
            dependencies = state.unresolved_dependencies(item)
            if dependencies:
                return _reject(
                    state,
                    attributed,
                    "dependency_blocked",
                    f"{item.id} is blocked by incomplete dependencies: {', '.join(dependencies)}",
                )
            update["effort_remaining"] = max(0.0, item.effort_remaining - action.amount)
        elif action.kind == "test":
            discovered = min(item.latent_defect_severity, action.amount)
            update["latent_defect_severity"] = item.latent_defect_severity - discovered
            update["discovered_defect_severity"] = item.discovered_defect_severity + discovered
        elif action.kind == "remediate":
            if item.discovered_defect_severity <= 0:
                return _reject(
                    state,
                    attributed,
                    "invalid_action_state",
                    f"{item.id} has no discovered defects to remediate",
                )
            removed = min(item.discovered_defect_severity, action.amount)
            update["discovered_defect_severity"] = item.discovered_defect_severity - removed
            event_amount = removed
        elif action.kind == "reduce_scope":
            update["removed_scope"] = True
        else:
            return _reject(
                state,
                attributed,
                "unsupported_action",
                f"unsupported action {action.kind}",
            )
        items = list(state.items)
        items[index] = item.model_copy(update=update)
        remediation_cost = state.remediation_cost
        if action.kind == "remediate":
            remediation_cost += event_amount
        return TransitionResult(
            state=state.model_copy(
                update={"items": tuple(items), "remediation_cost": remediation_cost}
            ),
            events=(f"{action.kind}:{item.id}:{event_amount}",),
        )

    def truth(self, state: WorldState, scope: tuple[str, ...]) -> OperationalHealth:
        selected = [item for item in state.items if item.id in scope and not item.removed_scope]
        if not selected:
            raise ValueError("truth scope contains no active work items")
        total_value = sum(item.business_value for item in selected)

        def weighted(values: list[float]) -> float:
            return (
                sum(
                    value * item.business_value
                    for value, item in zip(values, selected, strict=True)
                )
                / total_value
            )

        progress = weighted(
            [1.0 - item.effort_remaining / item.initial_effort for item in selected]
        )
        quality = weighted(
            [
                _clamp(1.0 - (item.latent_defect_severity + item.discovered_defect_severity) / 10.0)
                for item in selected
            ]
        )
        schedule = weighted(
            [
                _clamp(
                    1.0
                    - max(0.0, item.effort_remaining - max(0, item.deadline_tick - state.tick))
                    / item.initial_effort
                )
                for item in selected
            ]
        )
        reliability = _clamp(1.0 - state.incident_severity / 10.0)
        return OperationalHealth(
            progress=progress,
            quality=quality,
            schedule=schedule,
            reliability=reliability,
        )
