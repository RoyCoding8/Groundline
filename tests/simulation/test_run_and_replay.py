import hashlib
import json
from pathlib import Path

import pytest

from groundline.events.artifacts import (
    ArtifactCorruptError,
    canonical_hash,
    verify_run_artifacts,
)
from groundline.events.store import FileEventStore, canonical_json
from groundline.organization.models import AgentConfig, OrganizationConfig
from groundline.organization.topology import ReportingSpan
from groundline.policy.fixture import FixturePolicy
from groundline.policy.models import AgentContext, PolicyDecision
from groundline.replay.engine import ReplayEngine
from groundline.simulation.runner import RunRequest, SimulationRunner, TreatmentConfig
from groundline.world.models import ScenarioConfig, WorkItemConfig, WorldAction


def request(*, max_ticks: int = 3) -> RunRequest:
    return RunRequest(
        scenario=ScenarioConfig(
            max_ticks=max_ticks,
            shock_tick=1,
            shock_item_id="release",
            shock_severity=6,
            work_items=[
                WorkItemConfig(
                    id="release",
                    department="Engineering",
                    business_value=1,
                    effort=3,
                    deadline_tick=3,
                )
            ],
        ),
        organization=OrganizationConfig(
            agents=[
                AgentConfig(id="exec", role="executive", department="Executive"),
                AgentConfig(id="manager", manager_id="exec", role="manager", department="QA"),
                AgentConfig(
                    id="qa",
                    manager_id="manager",
                    role="contributor",
                    department="QA",
                    skills={"defect_detection": 1},
                    traits={"blame_sensitivity": 0.8, "honesty": 0.5},
                ),
            ]
        ),
        treatment=TreatmentConfig(incentive_pressure=1, attention_budget=0),
        seed=42,
    )


def span_request(span: ReportingSpan) -> RunRequest:
    base = request(max_ticks=1)
    organization = OrganizationConfig(
        agents=(
            AgentConfig(id="exec", role="executive", department="Executive"),
            AgentConfig(
                id="manager-b", manager_id="exec", role="manager", department="Engineering"
            ),
            AgentConfig(
                id="manager-a", manager_id="exec", role="manager", department="Engineering"
            ),
            *(
                AgentConfig(
                    id=f"contributor-{index}",
                    manager_id="manager-b",
                    role="contributor",
                    department="Engineering",
                    skills={"delivery": 0.5},
                )
                for index in range(1, 5)
            ),
        )
    )
    return base.model_copy(
        update={
            "organization": organization,
            "treatment": TreatmentConfig(
                incentive_pressure=0.5,
                attention_budget=1,
                reporting_span=span,
            ),
        }
    )


class ContributorReleasePolicy(FixturePolicy):
    name = "contributor-release"
    fingerprint = "contributor-release-v1"

    async def decide(self, context: AgentContext) -> PolicyDecision:
        decision = await super().decide(context)
        if context.agent.role == "contributor":
            return decision.model_copy(update={"actions": (WorldAction(kind="release"),)})
        return decision


class FinalTickActionsPolicy(FixturePolicy):
    name = "final-tick-actions"
    fingerprint = "final-tick-actions-v1"

    async def decide(self, context: AgentContext) -> PolicyDecision:
        decision = await super().decide(context)
        item_id = context.scope[0]
        if context.agent.role == "contributor":
            return decision.model_copy(
                update={
                    "actions": (
                        WorldAction(kind="test", item_id=item_id, amount=100),
                        WorldAction(kind="work", item_id=item_id, amount=0.5),
                    )
                }
            )
        if context.agent.role == "executive":
            return decision.model_copy(
                update={
                    "actions": (
                        WorldAction(kind="remediate", item_id=item_id, amount=100),
                        WorldAction(kind="release"),
                    )
                }
            )
        return decision


@pytest.mark.asyncio
async def test_complete_run_finalizes_and_replays_without_policy_calls(tmp_path: Path) -> None:
    store = FileEventStore(tmp_path)
    result = await SimulationRunner().run(request(), FixturePolicy(), store)

    assert result.manifest.finalized
    assert result.manifest.event_count > 0
    assert {event.kind for event in result.events} >= {
        "decision_truth_snapshot",
        "truth_snapshot",
        "observation",
        "decision",
        "report",
        "consequence",
    }
    assert result.agent_turns == {"exec": 3, "manager": 3, "qa": 3}
    assert (result.run_directory / "events.jsonl").exists()
    assert (result.run_directory / "decisions.jsonl").exists()
    metrics_payload = json.loads((result.run_directory / "metrics.json").read_text())
    metrics = metrics_payload["distortion"]
    outcomes = metrics_payload["outcomes"]
    assert len(metrics) == 9
    assert {metric["depth"] for metric in metrics} == {0, 1, 2}
    assert all(metric["scope_business_value"] == 1 for metric in metrics)
    assert all(metric["scope"] == ["release"] for metric in metrics)
    assert all(0 <= metric["truth_score"] <= 1 for metric in metrics)
    assert all(0 <= metric["report_score"] <= 1 for metric in metrics)
    assert all(0 <= metric["confidence"] <= 1 for metric in metrics)
    assert all(metric["edge_transformation"] is None for metric in metrics if metric["depth"] == 2)
    assert all(
        metric["edge_transformation"] is not None for metric in metrics if metric["depth"] < 2
    )
    assert outcomes["upward_amplification"]["status"] == "available"
    assert outcomes["calibration"]["report_count"] == 9
    assert outcomes["operational_harm"]["maxima"] == request().scenario.harm_maxima.model_dump(
        mode="json"
    )

    replay = await ReplayEngine().replay(result.run_directory)

    assert replay.equivalent
    assert replay.network_calls == 0
    assert replay.event_hash == result.manifest.event_hash
    assert replay.reconstructed_event_count == result.manifest.event_count


@pytest.mark.asyncio
async def test_reporting_span_uses_and_persists_the_effective_topology(tmp_path: Path) -> None:
    narrow = await SimulationRunner().run(
        span_request("narrow"),
        FixturePolicy(),
        FileEventStore(tmp_path / "narrow"),
    )
    wide = await SimulationRunner().run(
        span_request("wide"),
        FixturePolicy(),
        FileEventStore(tmp_path / "wide"),
    )

    persisted: dict[str, RunRequest] = {}
    for span, result in (("narrow", narrow), ("wide", wide)):
        request_data = json.loads((result.run_directory / "request.json").read_text())
        effective = RunRequest.model_validate(request_data)
        persisted[span] = effective
        assert effective.requested_organization is not None
        assert effective.topology is not None
        assert effective.topology.requested_span == span
        assert effective.effective() == effective
        assert (
            result.manifest.request_hash
            == hashlib.sha256(canonical_json(request_data).encode()).hexdigest()
        )
        replay = await ReplayEngine().replay(result.run_directory)
        assert replay.equivalent

    assert narrow.manifest.run_id != wide.manifest.run_id
    assert persisted["narrow"].treatment.model_copy(update={"reporting_span": None}) == persisted[
        "wide"
    ].treatment.model_copy(update={"reporting_span": None})
    assert persisted["narrow"].scenario == persisted["wide"].scenario
    assert persisted["narrow"].seed == persisted["wide"].seed
    assert persisted["narrow"].organization.spans["manager-a"] == 2
    assert persisted["narrow"].organization.spans["manager-b"] == 2
    assert persisted["wide"].organization.spans["manager-a"] == 4
    assert persisted["wide"].organization.spans["manager-b"] == 0

    narrow_manager_causes = {
        event.actor_id: len(event.causes)
        for event in narrow.events
        if event.kind == "decision" and event.actor_id in {"manager-a", "manager-b"}
    }
    wide_manager_causes = {
        event.actor_id: len(event.causes)
        for event in wide.events
        if event.kind == "decision" and event.actor_id in {"manager-a", "manager-b"}
    }
    assert narrow_manager_causes == {"manager-a": 4, "manager-b": 4}
    assert wide_manager_causes == {"manager-a": 6, "manager-b": 1}


@pytest.mark.asyncio
async def test_runner_binds_policy_actions_to_the_real_actor_role(tmp_path: Path) -> None:
    one_tick = request(max_ticks=1)

    result = await SimulationRunner().run(
        one_tick,
        ContributorReleasePolicy(),
        FileEventStore(tmp_path),
    )

    consequences = [event for event in result.events if event.kind == "consequence"]
    rejections = [rejection for event in consequences for rejection in event.payload["rejections"]]
    assert {
        "code": "unauthorized_action",
        "reason": "contributor cannot perform release",
        "actor_id": "qa",
        "actor_role": "contributor",
        "action_kind": "release",
        "item_id": None,
    } in rejections
    assert all("company_release" not in event.payload["events"] for event in consequences)


@pytest.mark.asyncio
async def test_run_persists_post_action_final_truth_without_replacing_pre_action_truth(
    tmp_path: Path,
) -> None:
    one_tick = request(max_ticks=1)

    result = await SimulationRunner().run(
        one_tick,
        FinalTickActionsPolicy(),
        FileEventStore(tmp_path),
    )

    pre_action = [event for event in result.events if event.kind == "decision_truth_snapshot"]
    final = [event for event in result.events if event.kind == "truth_snapshot"]
    final_consequences = {
        event.sequence
        for event in result.events
        if event.kind == "consequence" and event.tick == one_tick.scenario.max_ticks
    }

    assert len(pre_action) == 1
    assert len(final) == 1
    pre_state = pre_action[0].payload["state"]
    final_state = final[0].payload["state"]
    assert pre_state["remediation_cost"] == 0
    assert final[0] == result.events[-1]
    assert set(final[0].causes) == final_consequences
    assert final_state["items"][0]["effort_remaining"] < pre_state["items"][0]["effort_remaining"]
    assert final_state["items"][0]["discovered_defect_severity"] == 0
    assert final_state["items"][0]["released"] is True
    assert final_state["remediation_cost"] > 0

    replay = await ReplayEngine().replay(result.run_directory)
    assert replay.equivalent
    assert replay.event_hash == result.manifest.event_hash


@pytest.mark.asyncio
async def test_manager_attention_changes_information_by_verification(tmp_path: Path) -> None:
    low = request()
    high = low.model_copy(
        update={"treatment": TreatmentConfig(incentive_pressure=1, attention_budget=1)}
    )

    low_result = await SimulationRunner().run(
        low, FixturePolicy(), FileEventStore(tmp_path / "low")
    )
    high_result = await SimulationRunner().run(
        high, FixturePolicy(), FileEventStore(tmp_path / "high")
    )

    assert not any(event.kind == "verification" for event in low_result.events)
    assert any(event.kind == "verification" for event in high_result.events)
    low_metrics = json.loads((low_result.run_directory / "metrics.json").read_text())["distortion"]
    high_metrics = json.loads((high_result.run_directory / "metrics.json").read_text())[
        "distortion"
    ]
    low_exec = [m for m in low_metrics if m["agent_id"] == "exec"][-1]
    high_exec = [m for m in high_metrics if m["agent_id"] == "exec"][-1]
    assert high_exec["optimism_bias"] < low_exec["optimism_bias"]


@pytest.mark.asyncio
async def test_finalized_run_cannot_be_silently_overwritten(tmp_path: Path) -> None:
    store = FileEventStore(tmp_path)
    result = await SimulationRunner().run(request(), FixturePolicy(), store)
    (result.run_directory / "events.jsonl").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="finalized run is immutable"):
        await SimulationRunner().run(request(), FixturePolicy(), store)


@pytest.mark.asyncio
async def test_verifier_rejects_tampered_request_metrics_and_manifest_counts(
    tmp_path: Path,
) -> None:
    result = await SimulationRunner().run(
        request(), FixturePolicy(), FileEventStore(tmp_path / "metrics")
    )
    metrics_path = result.run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["distortion"][0]["optimism_bias"] += 0.1
    metrics_path.write_text(canonical_json(metrics) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError) as metrics_error:
        verify_run_artifacts(result.run_directory)
    assert metrics_error.value.code == "hash_mismatch"
    assert metrics_error.value.filename == "metrics.json"

    result = await SimulationRunner().run(
        request(), FixturePolicy(), FileEventStore(tmp_path / "request")
    )
    request_path = result.run_directory / "request.json"
    request_payload = json.loads(request_path.read_text())
    request_payload["seed"] += 1
    request_path.write_text(canonical_json(request_payload) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError) as request_error:
        verify_run_artifacts(result.run_directory)
    assert request_error.value.code == "hash_mismatch"
    assert request_error.value.filename == "request.json"

    result = await SimulationRunner().run(
        request(), FixturePolicy(), FileEventStore(tmp_path / "count")
    )
    manifest_path = result.run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["event_count"] += 1
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError) as count_error:
        verify_run_artifacts(result.run_directory)
    assert count_error.value.code == "count_mismatch"
    assert count_error.value.filename == "events.jsonl"


@pytest.mark.asyncio
async def test_replay_detects_self_consistent_metric_ledger_tampering(tmp_path: Path) -> None:
    result = await SimulationRunner().run(request(), FixturePolicy(), FileEventStore(tmp_path))
    events_path = result.run_directory / "events.jsonl"
    metrics_path = result.run_directory / "metrics.json"
    manifest_path = result.run_directory / "manifest.json"

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    metric_event = next(event for event in events if event["kind"] == "metric")
    metric_event["payload"]["department"] = "Tampered"
    events_path.write_text(
        "".join(f"{canonical_json(event)}\n" for event in events), encoding="utf-8"
    )

    metrics = json.loads(metrics_path.read_text())
    metrics["distortion"][0]["department"] = "Tampered"
    metrics_path.write_text(canonical_json(metrics) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text())
    manifest["event_hash"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    manifest["metrics_hash"] = canonical_hash(metrics)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    verify_run_artifacts(result.run_directory)
    replay = await ReplayEngine().replay(result.run_directory)

    assert not replay.equivalent


@pytest.mark.asyncio
async def test_event_ledger_records_an_explicit_evidence_chain(tmp_path: Path) -> None:
    result = await SimulationRunner().run(request(), FixturePolicy(), FileEventStore(tmp_path))
    by_sequence = {event.sequence: event for event in result.events}

    for event in result.events:
        assert all(parent < event.sequence for parent in event.causes)
        assert all(parent in by_sequence for parent in event.causes)

    observations = [event for event in result.events if event.kind == "observation"]
    decisions = [event for event in result.events if event.kind == "decision"]
    reports = [event for event in result.events if event.kind == "report"]
    metrics = [event for event in result.events if event.kind == "metric"]
    consequences = [event for event in result.events if event.kind == "consequence"]
    truth_snapshots = [event for event in result.events if event.kind == "truth_snapshot"]
    assert observations and all(event.causes for event in observations)
    assert decisions and all(event.causes for event in decisions)
    assert reports and all(by_sequence[event.causes[0]].kind == "decision" for event in reports)
    assert metrics and all(
        {by_sequence[parent].kind for parent in event.causes}
        == {"decision_truth_snapshot", "report"}
        for event in metrics
    )
    assert consequences and all(
        {by_sequence[parent].kind for parent in event.causes} == {"decision"}
        for event in consequences
    )
    assert truth_snapshots and all(
        {by_sequence[parent].kind for parent in event.causes} == {"consequence"}
        for event in truth_snapshots
    )
