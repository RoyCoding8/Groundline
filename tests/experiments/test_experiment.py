import asyncio
import json
from pathlib import Path

import pytest

from distortion_engine.events.store import FileEventStore
from distortion_engine.experiments.analysis import (
    AnalysisSpecification,
    ContrastSpecification,
    SensitivitySpecification,
)
from distortion_engine.experiments.runner import ExperimentRequest, ExperimentRunner
from distortion_engine.metrics.outcomes import OutcomeSpecification
from distortion_engine.organization.models import AgentConfig, OrganizationConfig
from distortion_engine.policy.fixture import FixturePolicy
from distortion_engine.policy.models import AgentContext, PolicyDecision
from distortion_engine.simulation.runner import RunRequest, SimulationRunner, TreatmentConfig
from distortion_engine.world.models import ScenarioConfig, WorkItemConfig, WorldAction


def _analysis(baseline: str, intervention: str) -> AnalysisSpecification:
    return AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id=f"{intervention}-minus-{baseline}",
                baseline=baseline,
                intervention=intervention,
                outcome="executive_optimism_bias_adverse_mean",
                direction="two_sided",
                family="primary",
                status="confirmatory",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_paired_experiment_holds_seed_and_writes_run_index(tmp_path: Path) -> None:
    organization = OrganizationConfig(
        agents=(
            AgentConfig(
                id="worker", role="contributor", manager_id="manager", department="Engineering"
            ),
            AgentConfig(id="manager", role="manager", manager_id="exec", department="Engineering"),
            AgentConfig(id="exec", role="executive", manager_id=None, department="Executive"),
        )
    )
    scenario = ScenarioConfig(
        max_ticks=2,
        shock_tick=1,
        shock_severity=0.7,
        shock_item_id="api",
        work_items=(
            WorkItemConfig(
                id="api",
                department="Engineering",
                business_value=1.0,
                effort=3.0,
                deadline_tick=2,
            ),
        ),
    )
    request = ExperimentRequest(
        name="paired-smoke",
        seeds=(11, 29),
        scenario=scenario,
        organization=organization,
        treatments={
            "pressure": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
            "ablation": TreatmentConfig(incentive_pressure=0.0, attention_budget=0),
        },
        analysis=_analysis("ablation", "pressure"),
    )

    result = await ExperimentRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))

    assert len(result.runs) == 4
    assert {(run.seed, run.treatment) for run in result.runs} == {
        (11, "pressure"),
        (11, "ablation"),
        (29, "pressure"),
        (29, "ablation"),
    }
    index_rows = [json.loads(line) for line in result.index_path.read_text().splitlines()]
    assert len(index_rows) == 4
    assert all(Path(row["run_directory"]).exists() for row in index_rows)
    assert all(row["incident_regret"] >= 0 for row in index_rows)
    assert all("oracle_incident_duration" in row for row in index_rows)
    assert all(row["oracle_operational_harm"] is not None for row in index_rows)
    assert all(row["oracle_regret"] is not None for row in index_rows)
    assert all(row["operational_harm_regret"] is not None for row in index_rows)
    assert all(
        row["operational_harm_regret"]["index"] == pytest.approx(row["oracle_regret"])
        for row in index_rows
    )
    assert result.analysis_path.exists()
    assert (result.analysis_path.parent / "outcomes.csv").exists()
    assert (result.analysis_path.parent / "outcomes.parquet").exists()
    assert (result.analysis_path.parent / "report.md").exists()
    analysis = json.loads(result.analysis_path.read_text())
    assert analysis["unit_of_analysis"] == "seed"
    assert analysis["multiplicity"] == "Holm within declared confirmatory family"
    comparison = analysis["comparisons"]["pressure-minus-ablation"]
    assert comparison["baseline"] == "ablation"
    assert comparison["intervention"] == "pressure"
    assert comparison["outcome"] == "executive_optimism_bias_adverse_mean"
    assert comparison["p_value_method"] == "exact_sign_flip"
    assert analysis["design_diagnostics"]["complete_pairs"] == 2


def test_experiment_rejects_analysis_threshold_not_declared_for_outcomes() -> None:
    base = _request(name="missing-threshold", seeds=(11,))
    analysis = AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id="escalation-delay",
                baseline="ablation",
                intervention="pressure",
                outcome="escalation_delay_mean",
                direction="two_sided",
                family="secondary",
                status="exploratory",
            ),
        ),
        sensitivities=(
            SensitivitySpecification(
                id="threshold-2",
                contrast_id="escalation-delay",
                kind="alternative_escalation_threshold",
                threshold=2,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="analysis escalation thresholds are not computed.*2.0",
    ):
        ExperimentRequest(
            name=base.name,
            seeds=base.seeds,
            scenario=base.scenario,
            organization=base.organization,
            treatments=base.treatments,
            outcome_specification=OutcomeSpecification(escalation_sensitivity_thresholds=(0.5,)),
            analysis=analysis,
        )


def test_markdown_report_labels_monte_carlo_randomization_honestly() -> None:
    report = ExperimentRunner._markdown_report(
        "large-design",
        {
            "comparisons": {
                "pressure_minus_control": {
                    "status": "complete",
                    "mean_difference": 0.25,
                    "ci_low": 0.1,
                    "ci_high": 0.4,
                    "interval_method": "bca",
                    "p_value": 1 / 100_001,
                    "p_value_method": "monte_carlo_sign_flip",
                    "randomization_draws": 100_000,
                    "holm_adjusted_p_value": 0.04,
                },
                "exploratory": {
                    "status": "complete",
                    "mean_difference": 0.1,
                    "ci_low": -0.1,
                    "ci_high": 0.3,
                    "interval_method": "bca",
                    "p_value": 0.5,
                    "p_value_method": "exact_sign_flip",
                    "randomization_draws": 8,
                    "holm_adjusted_p_value": None,
                },
            },
            "sensitivities": {
                "all-ticks": {
                    "status": "complete",
                    "kind": "adverse_vs_all_ticks",
                    "outcome": "executive_optimism_bias_all_mean",
                    "mean_difference": 0.2,
                    "ci_low": 0.05,
                    "ci_high": 0.35,
                }
            },
            "design_diagnostics": {
                "complete_pairs": 32,
                "adequate_for_holm_resolution": True,
            },
            "secondary_repeated_measures": {"status": "fit"},
        },
    )

    assert "exact sign-flip tests" not in report
    assert "BCa" in report
    assert "Monte Carlo sign-flip; 100000 draws" in report
    assert "| 0.0000 (Monte Carlo" not in report
    assert "| exploratory | 0.1000 | [-0.1000, 0.3000] (BCa) | " in report
    assert "| not adjusted |" in report
    assert "## Preregistered sensitivities" in report
    assert "| all-ticks | adverse_vs_all_ticks | executive_optimism_bias_all_mean |" in report


@pytest.mark.asyncio
async def test_contributor_less_experiment_records_unavailable_amplification(
    tmp_path: Path,
) -> None:
    request = ExperimentRequest(
        name="executive-only",
        seeds=(11,),
        scenario=ScenarioConfig(
            max_ticks=1,
            shock_tick=1,
            shock_severity=0.1,
            shock_item_id="api",
            work_items=(
                WorkItemConfig(
                    id="api",
                    department="Executive",
                    business_value=1,
                    effort=1,
                    deadline_tick=1,
                ),
            ),
        ),
        organization=OrganizationConfig(
            agents=(AgentConfig(id="exec", role="executive", department="Executive"),)
        ),
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=0),
            "pressure": TreatmentConfig(incentive_pressure=1, attention_budget=0),
        },
        analysis=_analysis("control", "pressure"),
    )

    result = await ExperimentRunner().run(
        request,
        FixturePolicy(),
        FileEventStore(tmp_path),
    )

    assert len(result.runs) == 2
    assert all(run.upward_amplification is None for run in result.runs)
    assert all(run.upward_amplification_status == "unavailable" for run in result.runs)
    assert all(run.upward_amplification_reason == "no_contributors" for run in result.runs)
    index_rows = [json.loads(line) for line in result.index_path.read_text().splitlines()]
    assert all(row["upward_amplification"] is None for row in index_rows)
    assert all(row["upward_amplification_status"] == "unavailable" for row in index_rows)
    assert all(row["upward_amplification_reason"] == "no_contributors" for row in index_rows)


@pytest.mark.asyncio
async def test_experiment_runs_and_resumes_paired_reporting_span_topologies(
    tmp_path: Path,
) -> None:
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
                )
                for index in range(1, 5)
            ),
        )
    )
    request = ExperimentRequest(
        name="paired-span",
        seeds=(11,),
        scenario=ScenarioConfig(
            max_ticks=1,
            shock_tick=1,
            shock_severity=0.1,
            shock_item_id="api",
            work_items=(
                WorkItemConfig(
                    id="api",
                    department="Engineering",
                    business_value=1,
                    effort=4,
                    deadline_tick=1,
                ),
            ),
        ),
        organization=organization,
        treatments={
            "narrow": TreatmentConfig(
                incentive_pressure=0.5,
                attention_budget=1,
                reporting_span="narrow",
            ),
            "wide": TreatmentConfig(
                incentive_pressure=0.5,
                attention_budget=1,
                reporting_span="wide",
            ),
        },
        analysis=_analysis("narrow", "wide"),
    )

    first = await ExperimentRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))

    assert {run.reporting_span for run in first.runs} == {"narrow", "wide"}
    effective_spans: dict[str, tuple[int, int]] = {}
    for run in first.runs:
        persisted = RunRequest.model_validate_json(
            (Path(run.run_directory) / "request.json").read_text()
        )
        assert persisted.topology is not None
        effective_spans[run.reporting_span or ""] = (
            persisted.organization.spans["manager-a"],
            persisted.organization.spans["manager-b"],
        )
    assert effective_spans == {"narrow": (2, 2), "wide": (4, 0)}
    state = json.loads(first.state_path.read_text())
    oracle_records = {
        key: record for key, record in state["runs"].items() if record["kind"] == "oracle"
    }
    assert set(oracle_records) == {"oracle:narrow:seed:11", "oracle:wide:seed:11"}
    for span in ("narrow", "wide"):
        oracle_request = RunRequest.model_validate_json(
            (
                Path(oracle_records[f"oracle:{span}:seed:11"]["run_directory"]) / "request.json"
            ).read_text()
        )
        assert oracle_request.treatment.reporting_span == span

    resumed = await ExperimentRunner(simulation_runner=NeverRunSimulation()).run(
        request,
        FixturePolicy(),
        FileEventStore(tmp_path),
    )
    assert resumed.executed_runs == 0
    assert resumed.resumed_runs == 4


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
async def test_experiment_indexes_post_action_final_outcomes(tmp_path: Path) -> None:
    request = ExperimentRequest(
        name="post-action-outcomes",
        seeds=(11,),
        scenario=ScenarioConfig(
            max_ticks=1,
            shock_tick=1,
            shock_severity=1,
            shock_item_id="api",
            work_items=(
                WorkItemConfig(
                    id="api",
                    department="Engineering",
                    business_value=1,
                    effort=1,
                    deadline_tick=1,
                ),
            ),
        ),
        organization=OrganizationConfig(
            agents=(
                AgentConfig(id="exec", role="executive", department="Executive"),
                AgentConfig(id="manager", role="manager", manager_id="exec", department="QA"),
                AgentConfig(
                    id="qa",
                    role="contributor",
                    manager_id="manager",
                    department="QA",
                    skills={"defect_detection": 1},
                ),
            )
        ),
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=0),
            "pressure": TreatmentConfig(incentive_pressure=1, attention_budget=0),
        },
        analysis=_analysis("control", "pressure"),
    )

    result = await ExperimentRunner().run(
        request,
        FinalTickActionsPolicy(),
        FileEventStore(tmp_path),
    )

    assert len(result.runs) == 2
    for indexed in result.runs:
        events = [
            json.loads(line)
            for line in (Path(indexed.run_directory) / "events.jsonl").read_text().splitlines()
        ]
        pre_action = next(event for event in events if event["kind"] == "decision_truth_snapshot")
        final = next(event for event in events if event["kind"] == "truth_snapshot")
        assert pre_action["payload"]["state"]["remediation_cost"] == 0
        assert indexed.remediation_cost == final["payload"]["state"]["remediation_cost"]
        assert indexed.remediation_cost > 0


class NeverRunSimulation:
    async def run(self, *_: object, **__: object) -> object:
        raise AssertionError("a verified finalized run must be resumed, not executed")


@pytest.mark.asyncio
async def test_experiment_resume_skips_verified_finalized_runs(tmp_path: Path) -> None:
    request = _request(name="resume", seeds=(11, 29))
    first = await ExperimentRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))

    resumed = await ExperimentRunner(simulation_runner=NeverRunSimulation()).run(
        request, FixturePolicy(), FileEventStore(tmp_path)
    )

    assert resumed.executed_runs == 0
    assert resumed.resumed_runs == len(first.runs) + len(request.seeds)
    assert resumed.failed_runs == 0


class ConcurrencyProbe(SimulationRunner):
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def run(self, request: RunRequest, policy: object, store: FileEventStore):  # type: ignore[override]
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        try:
            return await super().run(request, policy, store)  # type: ignore[arg-type]
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_experiment_honors_bounded_concurrency(tmp_path: Path) -> None:
    probe = ConcurrencyProbe()
    request = _request(name="bounded", seeds=(1, 2, 3), max_concurrency=2)

    await ExperimentRunner(simulation_runner=probe).run(
        request, FixturePolicy(), FileEventStore(tmp_path)
    )

    assert probe.maximum == 2


class FailOnceSimulation(SimulationRunner):
    def __init__(self) -> None:
        self.failed = False

    async def run(self, request: RunRequest, policy: object, store: FileEventStore):  # type: ignore[override]
        if request.seed == 2 and request.treatment.incentive_pressure > 0 and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic interruption")
        return await super().run(request, policy, store)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_failed_run_is_indexed_and_recovered_on_resume(tmp_path: Path) -> None:
    request = _request(name="recover", seeds=(1, 2), max_concurrency=1)
    first = await ExperimentRunner(simulation_runner=FailOnceSimulation()).run(
        request, FixturePolicy(), FileEventStore(tmp_path)
    )
    state = json.loads((tmp_path / "experiments" / "recover" / "experiment-state.json").read_text())

    assert first.failed_runs == 1
    assert any(run["status"] == "failed" and run["error"] for run in state["runs"].values())

    recovered = await ExperimentRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))

    assert recovered.failed_runs == 0
    assert recovered.executed_runs == 1
    assert recovered.resumed_runs == 5


def _request(*, name: str, seeds: tuple[int, ...], max_concurrency: int = 4) -> ExperimentRequest:
    organization = OrganizationConfig(
        agents=(
            AgentConfig(
                id="worker", role="contributor", manager_id="manager", department="Engineering"
            ),
            AgentConfig(id="manager", role="manager", manager_id="exec", department="Engineering"),
            AgentConfig(id="exec", role="executive", manager_id=None, department="Executive"),
        )
    )
    scenario = ScenarioConfig(
        max_ticks=2,
        shock_tick=1,
        shock_severity=0.7,
        shock_item_id="api",
        work_items=(
            WorkItemConfig(
                id="api",
                department="Engineering",
                business_value=1.0,
                effort=3.0,
                deadline_tick=2,
            ),
        ),
    )
    return ExperimentRequest(
        name=name,
        seeds=seeds,
        scenario=scenario,
        organization=organization,
        treatments={
            "pressure": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
            "ablation": TreatmentConfig(incentive_pressure=0.0, attention_budget=0),
        },
        analysis=_analysis("ablation", "pressure"),
        max_concurrency=max_concurrency,
    )
