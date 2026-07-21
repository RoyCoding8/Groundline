import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distortion_engine import ENGINE_FINGERPRINT
from distortion_engine.events.models import DecisionLedgerEntry, Event, RunManifest
from distortion_engine.events.store import FileEventStore, canonical_json
from distortion_engine.metrics.engine import DistortionMetric, MetricsEngine
from distortion_engine.metrics.outcomes import (
    OutcomeSpecification,
    calculate_run_outcomes,
)
from distortion_engine.observation.engine import ObservationEngine
from distortion_engine.organization.models import OrganizationConfig
from distortion_engine.organization.topology import (
    ReportingSpan,
    ReportingTopology,
    transform_reporting_span,
)
from distortion_engine.policy.models import (
    AgentContext,
    AgentMemory,
    AgentPolicy,
    TruthAwarePolicy,
    VerificationEvidence,
)
from distortion_engine.world.engine import WorldEngine
from distortion_engine.world.models import ActorAction, ScenarioConfig


class TreatmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incentive_pressure: float = Field(ge=0, le=1)
    attention_budget: int = Field(ge=0)
    reporting_span: ReportingSpan | None = None


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: ScenarioConfig
    organization: OrganizationConfig
    treatment: TreatmentConfig
    seed: int
    outcome_specification: OutcomeSpecification = Field(default_factory=OutcomeSpecification)
    requested_organization: OrganizationConfig | None = None
    topology: ReportingTopology | None = None

    @model_validator(mode="after")
    def validate_effective_topology(self) -> "RunRequest":
        if self.topology is None:
            if self.requested_organization is not None:
                raise ValueError("requested organization and topology must be provided together")
            return self
        if self.requested_organization is None:
            raise ValueError("requested organization and topology must be provided together")
        if self.treatment.reporting_span is None:
            raise ValueError("effective topology requires a reporting span treatment")
        transformed = transform_reporting_span(
            self.requested_organization,
            self.treatment.reporting_span,
        )
        if transformed.organization != self.organization or transformed.topology != self.topology:
            raise ValueError("effective topology does not match the requested organization")
        return self

    def effective(self) -> "RunRequest":
        if self.topology is not None or self.treatment.reporting_span is None:
            return self
        transformed = transform_reporting_span(
            self.organization,
            self.treatment.reporting_span,
        )
        return RunRequest(
            scenario=self.scenario,
            organization=transformed.organization,
            treatment=self.treatment,
            seed=self.seed,
            outcome_specification=self.outcome_specification,
            requested_organization=self.organization,
            topology=transformed.topology,
        )


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    manifest: RunManifest
    events: tuple[Event, ...]
    run_directory: Path
    agent_turns: dict[str, int]


class SimulationRunner:
    async def run(
        self, request: RunRequest, policy: AgentPolicy, store: FileEventStore
    ) -> RunResult:
        request = request.effective()
        request_data = request.model_dump(mode="json")
        request_hash = hashlib.sha256(canonical_json(request_data).encode()).hexdigest()
        policy_fingerprint = getattr(policy, "fingerprint", policy.name)
        policy_tag = hashlib.sha256(
            f"{ENGINE_FINGERPRINT}:{policy_fingerprint}".encode()
        ).hexdigest()[:8]
        run_id = f"run-{request_hash[:12]}-{policy.name}-{policy_tag}"
        world_engine = WorldEngine()
        observation_engine = ObservationEngine()
        metrics_engine = MetricsEngine()
        business_value_by_item = {
            item.id: item.business_value for item in request.scenario.work_items
        }
        world = world_engine.initialize(request.scenario, request.seed)
        events: list[Event] = []
        decisions: list[DecisionLedgerEntry] = []
        memories = {agent.id: AgentMemory() for agent in request.organization.agents}
        latest_reports: dict[str, Any] = {}
        latest_report_events: dict[str, int] = {}
        latest_metrics: dict[str, DistortionMetric] = {}
        turn_counts = {agent.id: 0 for agent in request.organization.agents}
        metrics: list[DistortionMetric] = []
        previous_consequences: tuple[int, ...] = ()

        def emit(
            kind: str,
            tick: int,
            payload: dict[str, Any],
            actor: str | None = None,
            causes: tuple[int, ...] = (),
        ) -> int:
            sequence = len(events)
            events.append(
                Event(
                    sequence=sequence,
                    kind=kind,
                    tick=tick,
                    actor_id=actor,
                    causes=causes,
                    payload=payload,
                )
            )
            return sequence

        def truth_payload() -> dict[str, Any]:
            company_scope = tuple(item.id for item in world.items if not item.removed_scope)
            company_health = world_engine.truth(world, company_scope) if company_scope else None
            return {
                "state": world.model_dump(mode="json"),
                "health": (
                    company_health.model_dump(mode="json") if company_health is not None else None
                ),
                "health_score": company_health.score if company_health is not None else None,
            }

        for tick in range(1, request.scenario.max_ticks + 1):
            advanced = world_engine.advance(world, tick, request.scenario)
            world = advanced.state
            decision_truth_sequence = emit(
                "decision_truth_snapshot",
                tick,
                truth_payload(),
                causes=previous_consequences,
            )
            tick_consequences: list[int] = []
            for level in request.organization.reporting_order:
                pending_actions: list[ActorAction] = []
                level_decisions: list[int] = []
                for agent_id in level:
                    agent = request.organization.by_id[agent_id]
                    scope = tuple(
                        item.id
                        for item in world.items
                        if agent.department in {"Executive", "QA"}
                        or item.department == agent.department
                    ) or tuple(item.id for item in world.items)
                    observation = observation_engine.observe(world, agent, scope)
                    observation_sequence = emit(
                        "observation",
                        tick,
                        observation.model_dump(mode="json"),
                        agent_id,
                        (decision_truth_sequence,),
                    )
                    incoming = tuple(
                        latest_reports[child]
                        for child in request.organization.children.get(agent_id, ())
                        if child in latest_reports
                    )
                    verification_evidence: tuple[VerificationEvidence, ...] = ()
                    verification_sequences: list[int] = []
                    if agent.role == "manager" and request.treatment.attention_budget:
                        candidates = sorted(
                            incoming,
                            key=lambda report: (
                                report.confidence * report.health.score,
                                report.agent_id,
                            ),
                            reverse=True,
                        )[: request.treatment.attention_budget]
                        verification_evidence = tuple(
                            VerificationEvidence(
                                reported_by=report.agent_id,
                                scope=report.scope,
                                observed_health=world_engine.truth(world, report.scope),
                            )
                            for report in candidates
                        )
                        for evidence in verification_evidence:
                            source_report = latest_report_events[evidence.reported_by]
                            verification_sequences.append(
                                emit(
                                    "verification",
                                    tick,
                                    evidence.model_dump(mode="json"),
                                    agent_id,
                                    (decision_truth_sequence, source_report),
                                )
                            )
                    context = AgentContext(
                        agent=agent,
                        depth=request.organization.depths[agent_id],
                        tick=tick,
                        scope=scope,
                        observation=observation,
                        incoming_reports=incoming,
                        verification_evidence=verification_evidence,
                        memory=memories[agent_id],
                        incentive_pressure=request.treatment.incentive_pressure,
                        attention_budget=(
                            request.treatment.attention_budget if agent.role == "manager" else 0
                        ),
                        release_window_open=tick
                        >= min(
                            item.deadline_tick for item in world.items if not item.removed_scope
                        ),
                    )
                    context_data = context.model_dump(mode="json")
                    context_hash = hashlib.sha256(canonical_json(context_data).encode()).hexdigest()
                    if isinstance(policy, TruthAwarePolicy):
                        decision = await policy.decide_with_truth(
                            context, world_engine.truth(world, scope)
                        )
                    else:
                        decision = await policy.decide(context)
                    decisions.append(
                        DecisionLedgerEntry(
                            sequence=len(decisions),
                            agent_id=agent_id,
                            tick=tick,
                            context_hash=context_hash,
                            policy=policy.name,
                            decision=decision.model_dump(mode="json"),
                        )
                    )
                    incoming_sequences = tuple(
                        latest_report_events[report.agent_id] for report in incoming
                    )
                    decision_sequence = emit(
                        "decision",
                        tick,
                        decision.model_dump(mode="json"),
                        agent_id,
                        (
                            observation_sequence,
                            *incoming_sequences,
                            *verification_sequences,
                        ),
                    )
                    level_decisions.append(decision_sequence)
                    report_payload = decision.report.model_dump(mode="json")
                    report_payload["health_score"] = decision.report.health.score
                    report_sequence = emit(
                        "report", tick, report_payload, agent_id, (decision_sequence,)
                    )
                    metric = metrics_engine.measure(
                        decision.report,
                        world_engine.truth(world, scope),
                        scope_business_value=sum(
                            business_value_by_item[item_id] for item_id in scope
                        ),
                        subordinate_metrics=tuple(
                            latest_metrics[report.agent_id] for report in incoming
                        ),
                    )
                    metrics.append(metric)
                    emit(
                        "metric",
                        tick,
                        metric.model_dump(mode="json"),
                        agent_id,
                        (decision_truth_sequence, report_sequence),
                    )
                    latest_reports[agent_id] = decision.report
                    latest_report_events[agent_id] = report_sequence
                    latest_metrics[agent_id] = metric
                    memories[agent_id] = decision.memory
                    turn_counts[agent_id] += 1
                    pending_actions.extend(
                        ActorAction(
                            actor_id=agent.id,
                            actor_role=agent.role,
                            action=action,
                        )
                        for action in decision.actions
                    )
                applied = world_engine.apply(world, tuple(pending_actions))
                world = applied.state
                tick_consequences.append(
                    emit(
                        "consequence",
                        tick,
                        {
                            "events": list(applied.events),
                            "rejections": [
                                rejection.model_dump(mode="json")
                                for rejection in applied.rejections
                            ],
                        },
                        causes=tuple(level_decisions),
                    )
                )
            emit(
                "truth_snapshot",
                tick,
                truth_payload(),
                causes=tuple(tick_consequences),
            )
            previous_consequences = tuple(tick_consequences)
        outcomes = calculate_run_outcomes(
            scenario=request.scenario,
            organization=request.organization,
            events=tuple(events),
            metrics=tuple(metrics),
            specification=request.outcome_specification,
        )
        run_directory, manifest = store.finalize(
            run_id=run_id,
            seed=request.seed,
            request_hash=request_hash,
            policy=policy.name,
            policy_fingerprint=policy_fingerprint,
            engine_fingerprint=ENGINE_FINGERPRINT,
            events=tuple(events),
            decisions=tuple(decisions),
            metrics={
                "distortion": [metric.model_dump(mode="json") for metric in metrics],
                "outcomes": outcomes.model_dump(mode="json"),
            },
            request=request_data,
        )
        return RunResult(
            manifest=manifest,
            events=tuple(events),
            run_directory=run_directory,
            agent_turns=turn_counts,
        )
