import pytest
from pydantic import ValidationError

from distortion_engine.organization import (
    AgentConfig,
    OrganizationConfig,
    transform_reporting_span,
)


def test_general_rooted_tree_derives_depth_span_and_reporting_order() -> None:
    agents = [AgentConfig(id="exec", role="executive", department="Executive")]
    for department in ("Product", "Engineering", "QA"):
        manager = f"{department.lower()}-manager"
        agents.append(
            AgentConfig(id=manager, manager_id="exec", role="manager", department=department)
        )
        agents.extend(
            AgentConfig(
                id=f"{department.lower()}-{index}",
                manager_id=manager,
                role="contributor",
                department=department,
            )
            for index in range(3)
        )

    organization = OrganizationConfig(agents=agents)

    assert organization.root_id == "exec"
    assert organization.depths["exec"] == 0
    assert organization.depths["qa-manager"] == 1
    assert organization.depths["qa-2"] == 2
    assert organization.spans["exec"] == 3
    assert organization.spans["qa-manager"] == 3
    assert organization.reporting_order[0] == tuple(
        sorted(a.id for a in agents if a.role == "contributor")
    )
    assert organization.reporting_order[-1] == ("exec",)


def test_invalid_cycle_fails_configuration() -> None:
    with pytest.raises(ValidationError, match="rooted tree"):
        OrganizationConfig(
            agents=[
                AgentConfig(id="a", manager_id="b", role="manager", department="QA"),
                AgentConfig(id="b", manager_id="a", role="manager", department="QA"),
            ]
        )


def span_organization(*, reverse: bool = False) -> OrganizationConfig:
    agents = [
        AgentConfig(id="exec", role="executive", department="Executive"),
        AgentConfig(id="manager-b", manager_id="exec", role="manager", department="QA"),
        AgentConfig(
            id="contributor-3", manager_id="manager-b", role="contributor", department="QA"
        ),
        AgentConfig(id="manager-a", manager_id="exec", role="manager", department="Engineering"),
        AgentConfig(
            id="contributor-1",
            manager_id="manager-b",
            role="contributor",
            department="Engineering",
            skills={"delivery": 0.9},
            traits={"honesty": 0.8},
            utility_weights={"quality": 0.7},
        ),
        AgentConfig(
            id="contributor-4", manager_id="manager-a", role="contributor", department="QA"
        ),
        AgentConfig(
            id="contributor-2", manager_id="manager-a", role="contributor", department="QA"
        ),
    ]
    return OrganizationConfig(agents=tuple(reversed(agents)) if reverse else tuple(agents))


def manager_map(organization: OrganizationConfig) -> dict[str, str | None]:
    return {agent.id: agent.manager_id for agent in organization.agents}


def test_reporting_span_transform_is_deterministic_balanced_and_role_preserving() -> None:
    base = span_organization()

    narrow = transform_reporting_span(base, "narrow")
    wide = transform_reporting_span(base, "wide")
    reordered_narrow = transform_reporting_span(span_organization(reverse=True), "narrow")

    assert manager_map(narrow.organization) == {
        "exec": None,
        "manager-a": "exec",
        "manager-b": "exec",
        "contributor-1": "manager-a",
        "contributor-2": "manager-b",
        "contributor-3": "manager-a",
        "contributor-4": "manager-b",
    }
    assert manager_map(wide.organization) == {
        "exec": None,
        "manager-a": "exec",
        "manager-b": "exec",
        "contributor-1": "manager-a",
        "contributor-2": "manager-a",
        "contributor-3": "manager-a",
        "contributor-4": "manager-a",
    }
    assert manager_map(reordered_narrow.organization) == manager_map(narrow.organization)
    assert tuple(agent.id for agent in narrow.organization.agents) == tuple(
        agent.id for agent in base.agents
    )
    assert tuple(agent.id for agent in reordered_narrow.organization.agents) == tuple(
        agent.id for agent in span_organization(reverse=True).agents
    )
    for result in (narrow, wide):
        for effective in result.organization.agents:
            requested = base.by_id[effective.id]
            assert effective.model_dump(exclude={"manager_id"}) == requested.model_dump(
                exclude={"manager_id"}
            )
    assert narrow.organization.spans["manager-a"] == 2
    assert narrow.organization.spans["manager-b"] == 2
    assert wide.organization.spans["manager-a"] == 4
    assert wide.organization.spans["manager-b"] == 0
    assert narrow.effective_fingerprint != wide.effective_fingerprint


def test_reporting_span_transform_is_idempotent_and_records_exact_edge_diff() -> None:
    base = span_organization()
    narrow = transform_reporting_span(base, "narrow")

    repeated = transform_reporting_span(narrow.organization, "narrow")

    assert repeated.organization == narrow.organization
    assert repeated.edge_changes == ()
    assert {
        (edge.agent_id, edge.previous_manager_id, edge.effective_manager_id)
        for edge in narrow.edge_changes
    } == {
        ("contributor-1", "manager-b", "manager-a"),
        ("contributor-2", "manager-a", "manager-b"),
        ("contributor-3", "manager-b", "manager-a"),
        ("contributor-4", "manager-a", "manager-b"),
    }


def test_reporting_span_transform_rejects_an_unidentifiable_contrast() -> None:
    organization = OrganizationConfig(
        agents=(
            AgentConfig(id="exec", role="executive", department="Executive"),
            AgentConfig(id="manager", manager_id="exec", role="manager", department="QA"),
            AgentConfig(id="one", manager_id="manager", role="contributor", department="QA"),
            AgentConfig(id="two", manager_id="manager", role="contributor", department="QA"),
        )
    )

    with pytest.raises(ValueError, match="at least two root-level managers"):
        transform_reporting_span(organization, "narrow")
