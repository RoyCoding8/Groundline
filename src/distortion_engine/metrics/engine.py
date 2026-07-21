from pydantic import BaseModel, ConfigDict, Field

from distortion_engine.domain.reports import Report
from distortion_engine.world.models import OperationalHealth


class DistortionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    department: str
    depth: int
    tick: int
    scope: tuple[str, ...]
    scope_business_value: float = Field(gt=0)
    truth_score: float = Field(ge=0, le=1)
    report_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    escalated: bool
    optimism_bias: float
    absolute_error: float
    vector_loss: float
    equal_weight_vector_loss: float
    progress_error: float
    quality_error: float
    schedule_error: float
    reliability_error: float
    edge_transformation: float | None = None


class MetricsEngine:
    weights = (0.20, 0.30, 0.20, 0.30)

    def measure(
        self,
        report: Report,
        truth: OperationalHealth,
        *,
        scope_business_value: float = 1,
        subordinate_metrics: tuple[DistortionMetric, ...] = (),
    ) -> DistortionMetric:
        bias = report.health.score - truth.score
        report_values = (
            report.health.progress,
            report.health.quality,
            report.health.schedule,
            report.health.reliability,
        )
        truth_values = (truth.progress, truth.quality, truth.schedule, truth.reliability)
        dimension_errors = tuple(
            abs(claim - actual) for claim, actual in zip(report_values, truth_values, strict=True)
        )
        loss = sum(
            weight * error for weight, error in zip(self.weights, dimension_errors, strict=True)
        )
        edge_transformation = None
        if subordinate_metrics:
            subordinate_value = sum(metric.scope_business_value for metric in subordinate_metrics)
            implicit_bias = (
                sum(
                    metric.optimism_bias * metric.scope_business_value
                    for metric in subordinate_metrics
                )
                / subordinate_value
            )
            edge_transformation = bias - implicit_bias
        return DistortionMetric(
            agent_id=report.agent_id,
            department=report.department,
            depth=report.depth,
            tick=report.tick,
            scope=report.scope,
            scope_business_value=scope_business_value,
            truth_score=truth.score,
            report_score=report.health.score,
            confidence=report.confidence,
            escalated=report.escalate,
            optimism_bias=bias,
            absolute_error=abs(bias),
            vector_loss=loss,
            equal_weight_vector_loss=sum(dimension_errors) / len(dimension_errors),
            progress_error=dimension_errors[0],
            quality_error=dimension_errors[1],
            schedule_error=dimension_errors[2],
            reliability_error=dimension_errors[3],
            edge_transformation=edge_transformation,
        )

    def upward_amplification(
        self,
        *,
        lower: tuple[DistortionMetric, ...],
        upper: tuple[DistortionMetric, ...],
    ) -> float:
        if not lower or not upper:
            raise ValueError("amplification requires metrics at both levels")
        lower_mean = sum(metric.optimism_bias for metric in lower) / len(lower)
        upper_mean = sum(metric.optimism_bias for metric in upper) / len(upper)
        return upper_mean - lower_mean
