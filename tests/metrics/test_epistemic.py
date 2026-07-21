import pytest

from groundline.domain.reports import Report
from groundline.metrics.engine import MetricsEngine
from groundline.world.models import OperationalHealth


def health(score: float) -> OperationalHealth:
    return OperationalHealth(progress=score, quality=score, schedule=score, reliability=score)


def test_metrics_use_structured_claims_and_worked_truth_values() -> None:
    engine = MetricsEngine()
    contributor = Report(
        agent_id="qa-1",
        department="QA",
        depth=2,
        tick=3,
        scope=("release",),
        health=health(0.5),
        confidence=0.8,
        escalate=True,
        explanation="Risk remains.",
    )
    executive = contributor.model_copy(
        update={"agent_id": "exec", "department": "Executive", "depth": 0, "health": health(0.8)}
    )

    low = engine.measure(contributor, health(0.4))
    high = engine.measure(executive, health(0.4))

    assert low.optimism_bias == pytest.approx(0.1)
    assert low.absolute_error == pytest.approx(0.1)
    assert low.vector_loss == pytest.approx(0.1)
    assert low.scope == ("release",)
    assert low.scope_business_value == 1
    assert low.truth_score == pytest.approx(0.4)
    assert low.report_score == pytest.approx(0.5)
    assert low.confidence == 0.8
    assert low.escalated is True
    assert low.edge_transformation is None
    assert engine.upward_amplification(lower=(low,), upper=(high,)) == pytest.approx(0.3)


def test_vector_loss_exposes_equal_weight_and_each_health_dimension() -> None:
    report = Report(
        agent_id="qa-1",
        department="QA",
        depth=1,
        tick=2,
        scope=("release",),
        health=OperationalHealth(
            progress=0.5,
            quality=0.4,
            schedule=0.9,
            reliability=0.2,
        ),
        confidence=0.8,
        escalate=True,
        explanation="Dimension-specific evidence.",
    )
    truth = OperationalHealth(
        progress=0.1,
        quality=0.2,
        schedule=0.3,
        reliability=0.4,
    )

    metric = MetricsEngine().measure(report, truth)

    assert metric.progress_error == pytest.approx(0.4)
    assert metric.quality_error == pytest.approx(0.2)
    assert metric.schedule_error == pytest.approx(0.6)
    assert metric.reliability_error == pytest.approx(0.2)
    assert metric.vector_loss == pytest.approx(0.32)
    assert metric.equal_weight_vector_loss == pytest.approx(0.35)


def test_edge_transformation_uses_subordinate_scope_value_weights() -> None:
    engine = MetricsEngine()
    first_report = Report(
        agent_id="first",
        department="Engineering",
        depth=2,
        tick=1,
        scope=("small",),
        health=health(0.5),
        confidence=0.8,
        escalate=False,
        explanation="Small scope.",
    )
    second_report = first_report.model_copy(
        update={"agent_id": "second", "scope": ("large",), "health": health(0.7)}
    )
    manager_report = first_report.model_copy(
        update={
            "agent_id": "manager",
            "department": "Executive",
            "depth": 1,
            "scope": ("small", "large"),
            "health": health(0.8),
        }
    )
    first = engine.measure(first_report, health(0.4), scope_business_value=1)
    second = engine.measure(second_report, health(0.4), scope_business_value=3)

    manager = engine.measure(
        manager_report,
        health(0.4),
        scope_business_value=4,
        subordinate_metrics=(first, second),
    )

    assert manager.optimism_bias == pytest.approx(0.4)
    assert manager.edge_transformation == pytest.approx(0.15)
