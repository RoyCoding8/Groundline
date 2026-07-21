from groundline.observation.engine import ObservationEngine
from groundline.organization.models import AgentConfig
from groundline.world.engine import WorldEngine
from groundline.world.models import ActorAction, ScenarioConfig, WorkItemConfig, WorldAction


def test_departments_receive_distinct_scoped_evidence_without_world_state() -> None:
    scenario = ScenarioConfig(
        max_ticks=2,
        shock_tick=1,
        shock_item_id="release",
        shock_severity=5,
        work_items=[
            WorkItemConfig(
                id="release", department="Engineering", business_value=1, effort=3, deadline_tick=2
            )
        ],
    )
    world_engine = WorldEngine()
    state = world_engine.advance(world_engine.initialize(scenario, 3), 1, scenario).state
    observer = ObservationEngine()

    product = observer.observe(
        state,
        AgentConfig(id="p", role="contributor", department="Product"),
        ("release",),
    )
    engineering = observer.observe(
        state,
        AgentConfig(id="e", role="contributor", department="Engineering"),
        ("release",),
    )
    qa = observer.observe(
        state,
        AgentConfig(id="q", role="contributor", department="QA", skills={"defect_detection": 1.0}),
        ("release",),
    )

    assert product.schedule_risk is not None and product.effort_remaining is None
    assert engineering.effort_remaining == 3 and engineering.detected_defect_severity is None
    assert qa.detected_defect_severity == state.items[0].latent_defect_severity
    assert "seed_tape" not in qa.model_dump()
    assert "latent_defect_severity" not in qa.model_dump()


def test_engineering_observes_only_scoped_unresolved_dependencies() -> None:
    scenario = ScenarioConfig(
        max_ticks=2,
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
                effort=1,
                deadline_tick=2,
                dependencies=("foundation",),
            ),
        ),
    )
    state = WorldEngine().initialize(scenario, seed=3)
    observer = ObservationEngine()

    engineering = observer.observe(
        state,
        AgentConfig(id="e", role="contributor", department="Engineering"),
        ("service",),
    )
    product = observer.observe(
        state,
        AgentConfig(id="p", role="contributor", department="Product"),
        ("service",),
    )

    assert engineering.unresolved_dependencies == {"service": ("foundation",)}
    assert product.unresolved_dependencies is None


def test_dependency_structure_changes_product_schedule_risk() -> None:
    def scenario(*, linked: bool) -> ScenarioConfig:
        return ScenarioConfig(
            max_ticks=2,
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

    engine = WorldEngine()
    linked_scenario = scenario(linked=True)
    independent_scenario = scenario(linked=False)
    work = (
        ActorAction(
            actor_id="engineer",
            actor_role="contributor",
            action=WorldAction(kind="work", item_id="service", amount=1),
        ),
    )
    linked = engine.apply(engine.initialize(linked_scenario, seed=3), work).state
    independent = engine.apply(engine.initialize(independent_scenario, seed=3), work).state
    linked = engine.advance(linked, 1, linked_scenario).state
    independent = engine.advance(independent, 1, independent_scenario).state
    product = AgentConfig(id="p", role="contributor", department="Product")
    observer = ObservationEngine()

    linked_observation = observer.observe(linked, product, ("service",))
    independent_observation = observer.observe(independent, product, ("service",))

    assert linked_observation.schedule_risk == 1
    assert independent_observation.schedule_risk == 0


def test_unknown_department_receives_no_qa_only_defect_evidence() -> None:
    scenario = ScenarioConfig(
        max_ticks=1,
        shock_tick=1,
        shock_item_id="release",
        shock_severity=5,
        work_items=(
            WorkItemConfig(
                id="release",
                department="Engineering",
                business_value=1,
                effort=1,
                deadline_tick=1,
            ),
        ),
    )
    engine = WorldEngine()
    state = engine.advance(engine.initialize(scenario, seed=3), 1, scenario).state

    finance = ObservationEngine().observe(
        state,
        AgentConfig(
            id="finance",
            role="contributor",
            department="Finance",
            skills={"defect_detection": 1},
        ),
        ("release",),
    )

    assert state.items[0].latent_defect_severity > 0
    assert finance.detected_defect_severity is None
    assert finance.detected_defects is None
    assert finance.discovered_defect_severity is None
    assert finance.effort_remaining is None
    assert finance.schedule_risk is None
    assert finance.incident_severity == 0
