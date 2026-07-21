import pytest
from pydantic import ValidationError

from distortion_engine.events.models import Event
from distortion_engine.metrics.engine import DistortionMetric
from distortion_engine.metrics.outcomes import (
    NormalizedOperationalHarm,
    OperationalHarm,
    OperationalHarmComponents,
    OutcomeSpecification,
    calculate_run_outcomes,
    compare_operational_harm,
)
from distortion_engine.organization.models import AgentConfig, OrganizationConfig
from distortion_engine.world.models import (
    OperationalHarmMaxima,
    ScenarioConfig,
    WorkItemConfig,
)


def organization() -> OrganizationConfig:
    return OrganizationConfig(
        agents=(
            AgentConfig(id="exec", role="executive", department="Executive"),
            AgentConfig(
                id="worker",
                role="contributor",
                manager_id="exec",
                department="Engineering",
            ),
        )
    )


def scenario(*, max_ticks: int = 1) -> ScenarioConfig:
    return ScenarioConfig(
        max_ticks=max_ticks,
        shock_tick=1,
        shock_item_id="small",
        shock_severity=1,
        work_items=(
            WorkItemConfig(
                id="small",
                department="Engineering",
                business_value=1,
                effort=1,
                deadline_tick=1,
            ),
            WorkItemConfig(
                id="large",
                department="Engineering",
                business_value=3,
                effort=1,
                deadline_tick=1,
            ),
        ),
        harm_maxima=OperationalHarmMaxima(
            release_delay=max_ticks,
            escaped_defects=10,
            incident=10,
            remediation=10,
            scope_loss=1,
        ),
    )


def metric(
    *,
    agent_id: str,
    depth: int,
    tick: int,
    truth: float,
    report: float,
    confidence: float,
    value: float,
    edge: float | None = None,
) -> DistortionMetric:
    bias = report - truth
    return DistortionMetric(
        agent_id=agent_id,
        department="Executive" if depth == 0 else "Engineering",
        depth=depth,
        tick=tick,
        scope=("small", "large") if depth == 0 else ("small",),
        scope_business_value=value,
        truth_score=truth,
        report_score=report,
        confidence=confidence,
        escalated=depth == 1 and tick == 1,
        optimism_bias=bias,
        absolute_error=abs(bias),
        vector_loss=abs(bias),
        equal_weight_vector_loss=abs(bias),
        progress_error=abs(bias),
        quality_error=abs(bias),
        schedule_error=abs(bias),
        reliability_error=abs(bias),
        edge_transformation=edge,
    )


def test_run_outcomes_match_independently_worked_estimands() -> None:
    metrics = (
        metric(
            agent_id="worker",
            depth=1,
            tick=1,
            truth=0.4,
            report=0.5,
            confidence=0.8,
            value=1,
        ),
        metric(
            agent_id="exec",
            depth=0,
            tick=1,
            truth=0.4,
            report=0.8,
            confidence=0.9,
            value=3,
            edge=0.3,
        ),
        metric(
            agent_id="worker",
            depth=1,
            tick=2,
            truth=1,
            report=1,
            confidence=0.8,
            value=1,
        ),
        metric(
            agent_id="exec",
            depth=0,
            tick=2,
            truth=1,
            report=1,
            confidence=0.9,
            value=3,
            edge=0,
        ),
    )
    events = (
        Event(
            sequence=0,
            kind="observation",
            tick=1,
            actor_id="worker",
            payload={"detected_defect_severity": 2},
        ),
        Event(
            sequence=1,
            kind="report",
            tick=1,
            actor_id="worker",
            payload={"escalate": True},
        ),
        Event(sequence=2, kind="verification", tick=1, payload={}),
        Event(
            sequence=3,
            kind="consequence",
            tick=1,
            payload={"events": ["company_release"], "rejections": []},
        ),
        Event(
            sequence=4,
            kind="truth_snapshot",
            tick=1,
            payload={
                "state": {
                    "items": [
                        {
                            "id": "small",
                            "latent_defect_severity": 2,
                            "discovered_defect_severity": 0,
                            "released": True,
                            "removed_scope": False,
                        },
                        {
                            "id": "large",
                            "latent_defect_severity": 0,
                            "discovered_defect_severity": 0,
                            "released": False,
                            "removed_scope": True,
                        },
                    ],
                    "incident_severity": 2,
                    "remediation_cost": 4,
                    "staffing_cost": 5,
                }
            },
        ),
    )

    outcomes = calculate_run_outcomes(
        scenario=scenario(),
        organization=organization(),
        events=events,
        metrics=metrics,
        specification=OutcomeSpecification(escalation_sensitivity_thresholds=(0.5, 3.0)),
    )

    by_depth = {row.depth: row for row in outcomes.depth}
    pre_release_by_depth = {row.depth: row for row in outcomes.pre_release_depth}
    assert by_depth[1].adverse_ticks is not None
    assert by_depth[0].adverse_ticks is not None
    assert by_depth[1].adverse_ticks.optimism_bias_mean == pytest.approx(0.1)
    assert by_depth[0].adverse_ticks.optimism_bias_mean == pytest.approx(0.4)
    assert by_depth[1].all_ticks.optimism_bias_mean == pytest.approx(0.05)
    assert by_depth[0].all_ticks.optimism_bias_mean == pytest.approx(0.2)
    assert pre_release_by_depth[1].all_ticks.optimism_bias_mean == pytest.approx(0.1)
    assert pre_release_by_depth[0].all_ticks.optimism_bias_mean == pytest.approx(0.4)
    assert outcomes.upward_amplification.value == pytest.approx(0.3)
    assert outcomes.edge_transformation.value == pytest.approx(0.3)
    assert outcomes.calibration.brier_score == pytest.approx(0.025)
    assert outcomes.escalation_delays[0].agent_id == "worker"
    assert outcomes.escalation_delays[0].delay_ticks == 0
    assert outcomes.escalation_delays[0].censored is False
    escalation_sensitivities = {
        result.threshold: result.delays for result in outcomes.escalation_sensitivities
    }
    assert escalation_sensitivities[0.5][0].delay_ticks == 0
    assert escalation_sensitivities[3.0] == ()

    harm = outcomes.operational_harm
    assert harm.release_tick == 1
    assert harm.release_censored is False
    assert harm.raw.release_delay == 0
    assert harm.raw.escaped_defects == pytest.approx(0.5)
    assert harm.raw.incident == 2
    assert harm.raw.remediation == 4
    assert harm.raw.scope_loss == pytest.approx(0.75)
    assert harm.raw.verification_cost == 1
    assert harm.raw.staffing_cost == 5
    assert harm.index == pytest.approx(0.19)


def test_post_release_reports_are_excluded_from_scalar_sensitivities() -> None:
    metrics = (
        metric(
            agent_id="worker",
            depth=1,
            tick=1,
            truth=0.4,
            report=0.5,
            confidence=0.8,
            value=1,
        ),
        metric(
            agent_id="exec",
            depth=0,
            tick=1,
            truth=0.4,
            report=0.8,
            confidence=0.9,
            value=3,
            edge=0.3,
        ),
        metric(
            agent_id="worker",
            depth=1,
            tick=2,
            truth=0.4,
            report=0.9,
            confidence=0.8,
            value=1,
        ),
        metric(
            agent_id="exec",
            depth=0,
            tick=2,
            truth=0.4,
            report=0.5,
            confidence=0.9,
            value=3,
            edge=-0.4,
        ),
    )
    events = (
        Event(
            sequence=0,
            kind="consequence",
            tick=1,
            payload={"events": ["company_release"], "rejections": []},
        ),
        Event(
            sequence=1,
            kind="truth_snapshot",
            tick=1,
            payload={
                "state": {
                    "items": [
                        {
                            "id": item_id,
                            "latent_defect_severity": 0,
                            "discovered_defect_severity": 0,
                            "released": True,
                            "removed_scope": False,
                        }
                        for item_id in ("small", "large")
                    ],
                    "incident_severity": 0,
                    "remediation_cost": 0,
                    "staffing_cost": 0,
                }
            },
        ),
    )

    outcomes = calculate_run_outcomes(
        scenario=scenario(),
        organization=organization(),
        events=events,
        metrics=metrics,
        specification=OutcomeSpecification(),
    )

    assert outcomes.upward_amplification.value == pytest.approx(-0.05)
    assert outcomes.pre_release_upward_amplification.value == pytest.approx(0.3)
    assert outcomes.edge_transformation.value == pytest.approx(-0.05)
    assert outcomes.pre_release_edge_transformation.value == pytest.approx(0.3)


def test_unreleased_run_uses_declared_censoring_maximum() -> None:
    events = tuple(
        Event(
            sequence=tick,
            kind="truth_snapshot",
            tick=tick,
            payload={
                "state": {
                    "items": [
                        {
                            "id": item_id,
                            "latent_defect_severity": 0,
                            "discovered_defect_severity": 0,
                            "released": False,
                            "removed_scope": False,
                        }
                        for item_id in ("small", "large")
                    ],
                    "incident_severity": 0,
                    "remediation_cost": 0,
                    "staffing_cost": 0,
                }
            },
        )
        for tick in range(1, 4)
    )
    metrics = (
        metric(
            agent_id="worker",
            depth=1,
            tick=1,
            truth=1,
            report=1,
            confidence=1,
            value=1,
        ),
        metric(
            agent_id="exec",
            depth=0,
            tick=1,
            truth=1,
            report=1,
            confidence=1,
            value=3,
            edge=0,
        ),
    )

    outcomes = calculate_run_outcomes(
        scenario=scenario(max_ticks=3),
        organization=organization(),
        events=events,
        metrics=metrics,
        specification=OutcomeSpecification(),
    )

    assert outcomes.operational_harm.release_censored is True
    assert outcomes.operational_harm.raw.release_delay == 3
    assert outcomes.operational_harm.normalized.release_delay == 1
    assert outcomes.upward_amplification.status == "unavailable"
    assert outcomes.upward_amplification.reason == "no_adverse_reports"


def test_operational_harm_regret_preserves_signed_component_differences() -> None:
    maxima = OperationalHarmMaxima()
    observed = OperationalHarm(
        raw=OperationalHarmComponents(
            release_delay=2,
            escaped_defects=1,
            incident=1,
            remediation=4,
            scope_loss=0.2,
            verification_cost=1,
            staffing_cost=3,
        ),
        normalized=NormalizedOperationalHarm(
            release_delay=0.2,
            escaped_defects=0.1,
            incident=0.1,
            remediation=0.4,
            scope_loss=0.2,
        ),
        index=0.2,
        maxima=maxima,
        release_tick=2,
        release_censored=False,
    )
    oracle = OperationalHarm(
        raw=OperationalHarmComponents(
            release_delay=1,
            escaped_defects=0,
            incident=2,
            remediation=1,
            scope_loss=0.3,
            verification_cost=3,
            staffing_cost=1,
        ),
        normalized=NormalizedOperationalHarm(
            release_delay=0.1,
            escaped_defects=0,
            incident=0.2,
            remediation=0.1,
            scope_loss=0.3,
        ),
        index=0.18,
        maxima=maxima,
        release_tick=1,
        release_censored=False,
    )

    regret = compare_operational_harm(observed, oracle)

    assert regret.oracle == oracle
    assert regret.raw.release_delay == 1
    assert regret.raw.incident == -1
    assert regret.raw.verification_cost == -2
    assert regret.normalized.remediation == pytest.approx(0.3)
    assert regret.normalized.scope_loss == pytest.approx(-0.1)
    assert regret.index == pytest.approx(0.02)

    with pytest.raises(ValueError, match="identical normalization maxima"):
        compare_operational_harm(
            observed,
            oracle.model_copy(update={"maxima": OperationalHarmMaxima(release_delay=50)}),
        )


def test_outcome_specification_rejects_invalid_escalation_sensitivities() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        OutcomeSpecification(escalation_sensitivity_thresholds=(2, 2))
    with pytest.raises(ValidationError, match="must differ from the primary"):
        OutcomeSpecification(
            escalation_severity_threshold=2,
            escalation_sensitivity_thresholds=(2,),
        )


def test_harm_maxima_must_be_strictly_positive() -> None:
    with pytest.raises(ValidationError):
        OperationalHarmMaxima(release_delay=0)
