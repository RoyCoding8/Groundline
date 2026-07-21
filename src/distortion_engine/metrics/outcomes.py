import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distortion_engine.events.models import Event
from distortion_engine.metrics.engine import DistortionMetric
from distortion_engine.organization.models import OrganizationConfig
from distortion_engine.world.models import OperationalHarmMaxima, ScenarioConfig

type OutcomeStatus = Literal["available", "unavailable"]
type OutcomeUnavailableReason = Literal[
    "no_adverse_reports",
    "no_contributors",
    "no_edge_reports",
]


class OutcomeSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adverse_health_threshold: float = Field(default=0.95, ge=0, le=1)
    release_health_threshold: float = Field(default=0.95, ge=0, le=1)
    escalation_severity_threshold: float = Field(default=1, ge=0)
    escalation_sensitivity_thresholds: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_escalation_thresholds(self) -> "OutcomeSpecification":
        thresholds = self.escalation_sensitivity_thresholds
        if any(not math.isfinite(threshold) or threshold < 0 for threshold in thresholds):
            raise ValueError("escalation sensitivity thresholds must be finite and non-negative")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("escalation sensitivity thresholds must be unique")
        if self.escalation_severity_threshold in thresholds:
            raise ValueError(
                "escalation sensitivity thresholds must differ from the primary threshold"
            )
        return self


class ScalarOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: OutcomeStatus
    value: float | None
    reason: OutcomeUnavailableReason | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "ScalarOutcome":
        if self.status == "available":
            if self.value is None or self.reason is not None:
                raise ValueError("available outcome requires a value only")
        elif self.value is not None or self.reason is None:
            raise ValueError("unavailable outcome requires a reason only")
        return self


class DistortionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_count: int = Field(ge=1)
    total_business_value: float = Field(gt=0)
    optimism_bias_mean: float
    optimism_bias_median: float
    absolute_error_mean: float
    vector_loss_mean: float
    equal_weight_vector_loss_mean: float
    progress_error_mean: float
    quality_error_mean: float
    schedule_error_mean: float
    reliability_error_mean: float
    edge_transformation_mean: float | None


class DepthOutcomes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: int = Field(ge=0)
    all_ticks: DistortionSummary
    adverse_ticks: DistortionSummary | None


class EscalationDelay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    depth: int = Field(ge=0)
    evidence_tick: int = Field(ge=1)
    escalation_tick: int | None
    delay_ticks: int = Field(ge=0)
    censored: bool


class EscalationThresholdOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0)
    delays: tuple[EscalationDelay, ...]


class CalibrationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_count: int = Field(ge=1)
    brier_score: float = Field(ge=0, le=1)


class OperationalHarmComponents(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_delay: float = Field(ge=0)
    escaped_defects: float = Field(ge=0)
    incident: float = Field(ge=0)
    remediation: float = Field(ge=0)
    scope_loss: float = Field(ge=0)
    verification_cost: float = Field(ge=0)
    staffing_cost: float = Field(ge=0)


class NormalizedOperationalHarm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_delay: float = Field(ge=0, le=1)
    escaped_defects: float = Field(ge=0, le=1)
    incident: float = Field(ge=0, le=1)
    remediation: float = Field(ge=0, le=1)
    scope_loss: float = Field(ge=0, le=1)


class OperationalHarm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: OperationalHarmComponents
    normalized: NormalizedOperationalHarm
    index: float = Field(ge=0, le=1)
    maxima: OperationalHarmMaxima
    release_tick: int | None
    release_censored: bool


class OperationalHarmComponentRegret(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_delay: float
    escaped_defects: float
    incident: float
    remediation: float
    scope_loss: float
    verification_cost: float
    staffing_cost: float


class NormalizedOperationalHarmComponentRegret(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_delay: float
    escaped_defects: float
    incident: float
    remediation: float
    scope_loss: float


class OperationalHarmRegret(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    oracle: OperationalHarm
    raw: OperationalHarmComponentRegret
    normalized: NormalizedOperationalHarmComponentRegret
    index: float


class RunOutcomes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: tuple[DepthOutcomes, ...]
    pre_release_depth: tuple[DepthOutcomes, ...]
    upward_amplification: ScalarOutcome
    pre_release_upward_amplification: ScalarOutcome
    edge_transformation: ScalarOutcome
    pre_release_edge_transformation: ScalarOutcome
    escalation_delays: tuple[EscalationDelay, ...]
    escalation_sensitivities: tuple[EscalationThresholdOutcome, ...]
    calibration: CalibrationOutcome
    operational_harm: OperationalHarm


def compare_operational_harm(
    observed: OperationalHarm,
    oracle: OperationalHarm,
) -> OperationalHarmRegret:
    if observed.maxima != oracle.maxima:
        raise ValueError("observed and Oracle harm must use identical normalization maxima")
    return OperationalHarmRegret(
        oracle=oracle,
        raw=OperationalHarmComponentRegret(
            release_delay=observed.raw.release_delay - oracle.raw.release_delay,
            escaped_defects=observed.raw.escaped_defects - oracle.raw.escaped_defects,
            incident=observed.raw.incident - oracle.raw.incident,
            remediation=observed.raw.remediation - oracle.raw.remediation,
            scope_loss=observed.raw.scope_loss - oracle.raw.scope_loss,
            verification_cost=observed.raw.verification_cost - oracle.raw.verification_cost,
            staffing_cost=observed.raw.staffing_cost - oracle.raw.staffing_cost,
        ),
        normalized=NormalizedOperationalHarmComponentRegret(
            release_delay=(observed.normalized.release_delay - oracle.normalized.release_delay),
            escaped_defects=(
                observed.normalized.escaped_defects - oracle.normalized.escaped_defects
            ),
            incident=observed.normalized.incident - oracle.normalized.incident,
            remediation=observed.normalized.remediation - oracle.normalized.remediation,
            scope_loss=observed.normalized.scope_loss - oracle.normalized.scope_loss,
        ),
        index=observed.index - oracle.index,
    )


def _weighted_mean(rows: tuple[DistortionMetric, ...], field: str) -> float:
    total = sum(row.scope_business_value for row in rows)
    return sum(float(getattr(row, field)) * row.scope_business_value for row in rows) / total


def _weighted_median(rows: tuple[DistortionMetric, ...], field: str) -> float:
    ordered = sorted((float(getattr(row, field)), row.scope_business_value) for row in rows)
    midpoint = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    raise AssertionError("weighted median requires at least one row")


def _summary(rows: tuple[DistortionMetric, ...]) -> DistortionSummary:
    edge_rows = tuple(row for row in rows if row.edge_transformation is not None)
    return DistortionSummary(
        report_count=len(rows),
        total_business_value=sum(row.scope_business_value for row in rows),
        optimism_bias_mean=_weighted_mean(rows, "optimism_bias"),
        optimism_bias_median=_weighted_median(rows, "optimism_bias"),
        absolute_error_mean=_weighted_mean(rows, "absolute_error"),
        vector_loss_mean=_weighted_mean(rows, "vector_loss"),
        equal_weight_vector_loss_mean=_weighted_mean(rows, "equal_weight_vector_loss"),
        progress_error_mean=_weighted_mean(rows, "progress_error"),
        quality_error_mean=_weighted_mean(rows, "quality_error"),
        schedule_error_mean=_weighted_mean(rows, "schedule_error"),
        reliability_error_mean=_weighted_mean(rows, "reliability_error"),
        edge_transformation_mean=(
            _weighted_mean(edge_rows, "edge_transformation") if edge_rows else None
        ),
    )


def _scalar(
    rows: tuple[DistortionMetric, ...],
    field: str,
    reason: OutcomeUnavailableReason,
) -> ScalarOutcome:
    if not rows:
        return ScalarOutcome(status="unavailable", value=None, reason=reason)
    return ScalarOutcome(status="available", value=_weighted_mean(rows, field))


def _observation_severity(event: Event) -> float:
    values = (
        event.payload.get("schedule_risk"),
        event.payload.get("discovered_defect_severity"),
        event.payload.get("detected_defect_severity"),
        event.payload.get("incident_severity"),
    )
    return max((float(value) for value in values if value is not None), default=0.0)


def _escalation_delays(
    events: tuple[Event, ...],
    organization: OrganizationConfig,
    max_ticks: int,
    threshold: float,
) -> tuple[EscalationDelay, ...]:
    evidence_ticks: dict[str, int] = {}
    escalation_ticks: dict[str, int] = {}
    for event in events:
        if event.actor_id is None:
            continue
        if (
            event.kind == "observation"
            and event.actor_id not in evidence_ticks
            and _observation_severity(event) >= threshold
        ):
            evidence_ticks[event.actor_id] = event.tick
        if (
            event.kind == "report"
            and event.payload.get("escalate") is True
            and event.actor_id not in escalation_ticks
        ):
            escalation_ticks[event.actor_id] = event.tick
    rows = []
    for agent_id, evidence_tick in sorted(evidence_ticks.items()):
        escalation_tick = escalation_ticks.get(agent_id)
        qualifies = escalation_tick is not None and escalation_tick >= evidence_tick
        rows.append(
            EscalationDelay(
                agent_id=agent_id,
                depth=organization.depths[agent_id],
                evidence_tick=evidence_tick,
                escalation_tick=escalation_tick if qualifies else None,
                delay_ticks=(
                    escalation_tick - evidence_tick
                    if qualifies and escalation_tick is not None
                    else max_ticks - evidence_tick
                ),
                censored=not qualifies,
            )
        )
    return tuple(rows)


def _operational_harm(
    scenario: ScenarioConfig,
    events: tuple[Event, ...],
) -> OperationalHarm:
    snapshots = tuple(event for event in events if event.kind == "truth_snapshot")
    if not snapshots:
        raise ValueError("run outcomes require post-action truth snapshots")
    release_tick = next(
        (
            event.tick
            for event in events
            if event.kind == "consequence" and "company_release" in event.payload["events"]
        ),
        None,
    )
    release_or_censor_tick = release_tick if release_tick is not None else scenario.max_ticks + 1
    values = {item.id: item.business_value for item in scenario.work_items}
    deadlines = {item.id: item.deadline_tick for item in scenario.work_items}
    total_value = sum(values.values())
    release_delay = (
        sum(
            values[item_id] * max(0, release_or_censor_tick - deadline)
            for item_id, deadline in deadlines.items()
        )
        / total_value
    )
    release_state = next(
        (event.payload["state"] for event in snapshots if event.tick == release_tick),
        None,
    )
    escaped_defects = 0.0
    if release_state is not None:
        escaped_defects = (
            sum(
                values[item["id"]]
                * (
                    float(item["latent_defect_severity"])
                    + float(item["discovered_defect_severity"])
                )
                for item in release_state["items"]
                if item["released"] and not item["removed_scope"]
            )
            / total_value
        )
    final_state = max(snapshots, key=lambda event: event.tick).payload["state"]
    incident = sum(float(event.payload["state"]["incident_severity"]) for event in snapshots)
    scope_loss = (
        sum(values[item["id"]] for item in final_state["items"] if item["removed_scope"])
        / total_value
    )
    raw = OperationalHarmComponents(
        release_delay=release_delay,
        escaped_defects=escaped_defects,
        incident=incident,
        remediation=float(final_state["remediation_cost"]),
        scope_loss=scope_loss,
        verification_cost=float(sum(event.kind == "verification" for event in events)),
        staffing_cost=float(final_state["staffing_cost"]),
    )
    maxima = scenario.harm_maxima

    def normalized(value: float, maximum: float) -> float:
        return min(1.0, value / maximum)

    normalized_components = NormalizedOperationalHarm(
        release_delay=normalized(raw.release_delay, maxima.release_delay),
        escaped_defects=normalized(raw.escaped_defects, maxima.escaped_defects),
        incident=normalized(raw.incident, maxima.incident),
        remediation=normalized(raw.remediation, maxima.remediation),
        scope_loss=normalized(raw.scope_loss, maxima.scope_loss),
    )
    index = (
        0.20 * normalized_components.release_delay
        + 0.30 * normalized_components.escaped_defects
        + 0.30 * normalized_components.incident
        + 0.10 * normalized_components.remediation
        + 0.10 * normalized_components.scope_loss
    )
    return OperationalHarm(
        raw=raw,
        normalized=normalized_components,
        index=index,
        maxima=maxima,
        release_tick=release_tick,
        release_censored=release_tick is None,
    )


def _depth_outcomes(
    metrics: tuple[DistortionMetric, ...],
    adverse_health_threshold: float,
) -> tuple[DepthOutcomes, ...]:
    by_depth: dict[int, list[DistortionMetric]] = defaultdict(list)
    for metric in metrics:
        by_depth[metric.depth].append(metric)
    return tuple(
        DepthOutcomes(
            depth=level,
            all_ticks=_summary(tuple(rows)),
            adverse_ticks=(
                _summary(adverse_rows)
                if (
                    adverse_rows := tuple(
                        row for row in rows if row.truth_score < adverse_health_threshold
                    )
                )
                else None
            ),
        )
        for level, rows in sorted(by_depth.items())
    )


def _amplification(
    metrics: tuple[DistortionMetric, ...],
    organization: OrganizationConfig,
    adverse_health_threshold: float,
) -> ScalarOutcome:
    contributor_ids = {agent.id for agent in organization.agents if agent.role == "contributor"}
    if not contributor_ids:
        return ScalarOutcome(status="unavailable", value=None, reason="no_contributors")
    adverse = tuple(metric for metric in metrics if metric.truth_score < adverse_health_threshold)
    executive_rows = tuple(metric for metric in adverse if metric.agent_id == organization.root_id)
    contributor_rows = tuple(metric for metric in adverse if metric.agent_id in contributor_ids)
    if not executive_rows or not contributor_rows:
        return ScalarOutcome(status="unavailable", value=None, reason="no_adverse_reports")
    return ScalarOutcome(
        status="available",
        value=_weighted_mean(executive_rows, "optimism_bias")
        - _weighted_mean(contributor_rows, "optimism_bias"),
    )


def _edge_transformation(
    metrics: tuple[DistortionMetric, ...],
    adverse_health_threshold: float,
) -> ScalarOutcome:
    edge_rows = tuple(
        metric
        for metric in metrics
        if metric.truth_score < adverse_health_threshold and metric.edge_transformation is not None
    )
    return _scalar(edge_rows, "edge_transformation", "no_edge_reports")


def calculate_run_outcomes(
    *,
    scenario: ScenarioConfig,
    organization: OrganizationConfig,
    events: tuple[Event, ...],
    metrics: tuple[DistortionMetric, ...],
    specification: OutcomeSpecification,
) -> RunOutcomes:
    if not metrics:
        raise ValueError("run outcomes require distortion metrics")
    depth = _depth_outcomes(metrics, specification.adverse_health_threshold)
    operational_harm = _operational_harm(scenario, events)
    pre_release_metrics = tuple(
        metric
        for metric in metrics
        if operational_harm.release_tick is None or metric.tick <= operational_harm.release_tick
    )
    pre_release_depth = _depth_outcomes(
        pre_release_metrics,
        specification.adverse_health_threshold,
    )
    amplification = _amplification(
        metrics,
        organization,
        specification.adverse_health_threshold,
    )
    pre_release_amplification = _amplification(
        pre_release_metrics,
        organization,
        specification.adverse_health_threshold,
    )
    edge = _edge_transformation(metrics, specification.adverse_health_threshold)
    pre_release_edge = _edge_transformation(
        pre_release_metrics,
        specification.adverse_health_threshold,
    )
    brier_values = []
    for metric in metrics:
        predicted_healthy = (
            metric.confidence
            if metric.report_score >= specification.release_health_threshold
            else 1 - metric.confidence
        )
        actual_healthy = float(metric.truth_score >= specification.release_health_threshold)
        brier_values.append((predicted_healthy - actual_healthy) ** 2)
    return RunOutcomes(
        depth=depth,
        pre_release_depth=pre_release_depth,
        upward_amplification=amplification,
        pre_release_upward_amplification=pre_release_amplification,
        edge_transformation=edge,
        pre_release_edge_transformation=pre_release_edge,
        escalation_delays=_escalation_delays(
            events,
            organization,
            scenario.max_ticks,
            specification.escalation_severity_threshold,
        ),
        escalation_sensitivities=tuple(
            EscalationThresholdOutcome(
                threshold=threshold,
                delays=_escalation_delays(
                    events,
                    organization,
                    scenario.max_ticks,
                    threshold,
                ),
            )
            for threshold in specification.escalation_sensitivity_thresholds
        ),
        calibration=CalibrationOutcome(
            report_count=len(brier_values),
            brier_score=sum(brier_values) / len(brier_values),
        ),
        operational_harm=operational_harm,
    )
