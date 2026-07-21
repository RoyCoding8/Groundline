import pytest

from groundline.organization.models import AgentRole
from groundline.world.engine import WorldEngine
from groundline.world.models import (
    ActionKind,
    ActorAction,
    ScenarioConfig,
    WorkItemConfig,
    WorldAction,
)


def scenario() -> ScenarioConfig:
    return ScenarioConfig(
        max_ticks=4,
        shock_tick=1,
        shock_item_id="release",
        shock_severity=6.0,
        work_items=[
            WorkItemConfig(
                id="release",
                department="Engineering",
                business_value=1.0,
                effort=4.0,
                deadline_tick=4,
            )
        ],
    )


def attributed(
    role: AgentRole,
    kind: ActionKind,
    *,
    item_id: str | None = None,
    amount: float = 0,
) -> ActorAction:
    return ActorAction(
        actor_id=f"the-{role}",
        actor_role=role,
        action=WorldAction(kind=kind, item_id=item_id, amount=amount),
    )


def test_world_is_deterministic_and_actions_have_worked_consequences() -> None:
    engine = WorldEngine()
    initial = engine.initialize(scenario(), seed=42)
    repeated = engine.initialize(scenario(), seed=42)
    work = attributed("contributor", "work", item_id="release", amount=2)

    assert initial.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert len(initial.seed_tape) == 64

    worked = engine.apply(initial, (work,))
    shocked = engine.advance(worked.state, tick=1, scenario=scenario())
    repeated_shock = engine.advance(engine.apply(repeated, (work,)).state, 1, scenario())
    alternate_shock = engine.advance(
        engine.apply(engine.initialize(scenario(), seed=7), (work,)).state,
        tick=1,
        scenario=scenario(),
    )
    truth = engine.truth(shocked.state, ("release",))

    item = shocked.state.items[0]
    assert item.effort_remaining == 2.0
    assert item.latent_defect_severity == repeated_shock.state.items[0].latent_defect_severity
    assert item.latent_defect_severity != alternate_shock.state.items[0].latent_defect_severity
    assert truth.progress == 0.5
    assert truth.quality == 1 - item.latent_defect_severity / 10
    assert truth.schedule == 1.0
    assert truth.reliability == 1.0


def test_invalid_action_is_rejected_without_partial_mutation() -> None:
    engine = WorldEngine()
    initial = engine.initialize(scenario(), seed=7)

    result = engine.apply(
        initial,
        (attributed("executive", "remediate", item_id="release", amount=3),),
    )

    assert result.state == initial
    assert result.events == ()
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.code == "invalid_action_state"
    assert rejection.actor_id == "the-executive"
    assert rejection.actor_role == "executive"
    assert rejection.action_kind == "remediate"
    assert rejection.item_id == "release"


def test_report_language_has_no_world_entry_point() -> None:
    assert "report" not in WorldEngine.apply.__annotations__


def test_global_release_turns_unremediated_defects_into_incident_harm() -> None:
    engine = WorldEngine()
    shocked = engine.advance(engine.initialize(scenario(), 9), 1, scenario()).state

    released = engine.apply(shocked, (attributed("executive", "release"),))
    harmed = engine.advance(released.state, 2, scenario()).state
    delayed = engine.apply(shocked, (attributed("executive", "delay_release"),))
    safe = engine.advance(delayed.state, 2, scenario()).state

    assert all(item.released for item in released.state.items)
    assert harmed.incident_severity > 0
    assert harmed.incident_duration == 1
    assert safe.incident_severity == 0


@pytest.mark.parametrize(
    ("role", "kind"),
    [
        ("contributor", "remediate"),
        ("contributor", "release"),
        ("contributor", "delay_release"),
        ("contributor", "reduce_scope"),
        ("contributor", "investigate"),
        ("contributor", "add_qa_capacity"),
        ("manager", "work"),
        ("manager", "test"),
        ("manager", "release"),
        ("executive", "work"),
        ("executive", "test"),
    ],
)
def test_world_rejects_unauthorized_role_actions(role: AgentRole, kind: ActionKind) -> None:
    engine = WorldEngine()
    initial = engine.initialize(scenario(), seed=7)
    item_id = "release" if kind in {"work", "test", "remediate", "reduce_scope"} else None

    result = engine.apply(initial, (attributed(role, kind, item_id=item_id, amount=1),))

    assert result.state == initial
    assert result.events == ()
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.code == "unauthorized_action"
    assert rejection.actor_role == role
    assert rejection.action_kind == kind


def test_unauthorized_action_rolls_back_an_entire_batch() -> None:
    engine = WorldEngine()
    initial = engine.initialize(scenario(), seed=7)

    result = engine.apply(
        initial,
        (
            attributed("contributor", "work", item_id="release", amount=1),
            attributed("contributor", "release"),
        ),
    )

    assert result.state == initial
    assert result.events == ()
    assert result.rejections[0].code == "unauthorized_action"


def test_executive_remediation_charges_only_performed_work() -> None:
    engine = WorldEngine()
    shocked = engine.advance(engine.initialize(scenario(), seed=7), 1, scenario()).state
    discovered = engine.apply(
        shocked,
        (attributed("contributor", "test", item_id="release", amount=10),),
    ).state
    available = discovered.items[0].discovered_defect_severity

    remediated = engine.apply(
        discovered,
        (attributed("executive", "remediate", item_id="release", amount=10),),
    )

    assert remediated.rejections == ()
    assert remediated.state.items[0].discovered_defect_severity == 0
    assert remediated.state.remediation_cost == available
    assert remediated.events == (f"remediate:release:{available}",)


def dependency_scenario(*, linked: bool = True) -> ScenarioConfig:
    return ScenarioConfig(
        max_ticks=3,
        shock_tick=1,
        shock_item_id="service",
        shock_severity=1,
        work_items=(
            WorkItemConfig(
                id="foundation",
                department="Engineering",
                business_value=1,
                effort=1,
                deadline_tick=1,
            ),
            WorkItemConfig(
                id="service",
                department="Engineering",
                business_value=1,
                effort=2,
                deadline_tick=2,
                dependencies=("foundation",) if linked else (),
            ),
        ),
    )


def test_scenario_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        ScenarioConfig(
            max_ticks=2,
            shock_tick=1,
            shock_item_id="a",
            shock_severity=1,
            work_items=(
                WorkItemConfig(
                    id="a",
                    department="Engineering",
                    business_value=1,
                    effort=1,
                    deadline_tick=1,
                    dependencies=("b",),
                ),
                WorkItemConfig(
                    id="b",
                    department="Engineering",
                    business_value=1,
                    effort=1,
                    deadline_tick=1,
                    dependencies=("a",),
                ),
            ),
        )


def test_work_waits_for_dependencies_and_can_follow_completion_in_one_batch() -> None:
    engine = WorldEngine()
    initial = engine.initialize(dependency_scenario(), seed=7)

    blocked = engine.apply(
        initial,
        (attributed("contributor", "work", item_id="service", amount=1),),
    )

    assert blocked.state == initial
    assert blocked.events == ()
    assert blocked.rejections[0].code == "dependency_blocked"
    assert "foundation" in blocked.rejections[0].reason

    progressed = engine.apply(
        initial,
        (
            attributed("contributor", "work", item_id="foundation", amount=1),
            attributed("contributor", "work", item_id="service", amount=1),
        ),
    )

    assert progressed.rejections == ()
    assert progressed.state.items[0].effort_remaining == 0
    assert progressed.state.items[1].effort_remaining == 1
    assert progressed.events == ("work:foundation:1.0", "work:service:1.0")


def test_release_waits_for_active_item_dependencies() -> None:
    engine = WorldEngine()
    initial = engine.initialize(dependency_scenario(), seed=7)

    blocked = engine.apply(initial, (attributed("executive", "release"),))
    prerequisite_done = engine.apply(
        initial,
        (attributed("contributor", "work", item_id="foundation", amount=1),),
    ).state
    released = engine.apply(prerequisite_done, (attributed("executive", "release"),))

    assert blocked.state == initial
    assert blocked.rejections[0].code == "dependency_blocked"
    assert released.rejections == ()
    assert all(item.released for item in released.state.items)
