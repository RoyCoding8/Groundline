import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from groundline import __version__
from groundline.api.app import ArtifactRepository, create_app
from groundline.api.jobs import ExperimentJobManager, ExperimentLaunchConflict
from groundline.api.models import LaunchExperiment
from groundline.events.store import FileEventStore
from groundline.experiments.analysis import AnalysisSpecification, ContrastSpecification
from groundline.experiments.runner import ExperimentRequest
from groundline.organization.models import AgentConfig, OrganizationConfig
from groundline.policy.fixture import FixturePolicy
from groundline.simulation.runner import RunRequest, SimulationRunner, TreatmentConfig
from groundline.world.models import ScenarioConfig, WorkItemConfig


@pytest.mark.asyncio
async def test_read_only_api_serves_run_truth_reports_and_metrics(tmp_path: Path) -> None:
    request = RunRequest(
        scenario=ScenarioConfig(
            max_ticks=1,
            shock_tick=1,
            shock_item_id="api",
            shock_severity=6,
            work_items=(
                WorkItemConfig(
                    id="api",
                    department="Engineering",
                    business_value=1,
                    effort=2,
                    deadline_tick=1,
                ),
            ),
        ),
        organization=OrganizationConfig(
            agents=(
                AgentConfig(
                    id="worker", role="contributor", manager_id="exec", department="Engineering"
                ),
                AgentConfig(id="exec", role="executive", department="Executive"),
            )
        ),
        treatment=TreatmentConfig(incentive_pressure=0.8, attention_budget=0),
        seed=5,
    )
    result = await SimulationRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        runs = (await client.get("/api/runs")).json()
        detail = (await client.get(f"/api/runs/{result.manifest.run_id}")).json()
        timeline = (await client.get(f"/api/runs/{result.manifest.run_id}/timeline")).json()
        shell = await client.get("/")

    assert runs[0]["run_id"] == result.manifest.run_id
    assert detail["manifest"]["seed"] == 5
    assert "distortion" in detail["metrics"]
    assert {event["kind"] for event in timeline} >= {"truth_snapshot", "report", "metric"}
    assert shell.status_code == 200
    assert "The Groundline" in shell.text


@pytest.mark.asyncio
async def test_evidence_endpoint_filters_department_depth_and_exposes_causes(
    tmp_path: Path,
) -> None:
    request = _run_request(seed=5)
    result = await SimulationRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/runs/{result.manifest.run_id}/evidence",
            params={"department": "Engineering", "depth": 1},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"]
    assert {node["department"] for node in body["nodes"]} == {"Engineering"}
    assert {node["depth"] for node in body["nodes"]} == {1}
    assert any(node["causes"] for node in body["nodes"])


@pytest.mark.asyncio
async def test_api_reports_missing_corrupt_and_invalid_artifacts(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/api/runs/does-not-exist")
        invalid = await client.post(
            "/api/experiments",
            json={"experiment": {"name": "../escape"}, "policy": "fixture"},
        )

    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_experiment_request_rejects_path_traversal(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path)

    with pytest.raises(KeyError):
        repository.experiment_request("../escape")


@pytest.mark.asyncio
async def test_corrupt_run_is_skipped_in_listing_but_loud_on_direct_lookup(
    tmp_path: Path,
) -> None:
    good = await SimulationRunner().run(
        _run_request(seed=3), FixturePolicy(), FileEventStore(tmp_path)
    )
    bad = await SimulationRunner().run(
        _run_request(seed=9), FixturePolicy(), FileEventStore(tmp_path)
    )
    # Corrupt the second run's manifest (missing required fields).
    (bad.run_directory / "manifest.json").write_text('{"run_id":"broken"}\n', encoding="utf-8")

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/api/runs")
        # The listing is resilient: the corrupt run is skipped, the good one survives.
        assert listing.status_code == 200
        listed_ids = {run["run_id"] for run in listing.json()}
        assert good.manifest.run_id in listed_ids
        assert bad.manifest.run_id not in listed_ids

        # A direct lookup of the corrupt run still verifies it and surfaces the
        # corruption as a typed 500 rather than a silent 404.
        direct = await client.get(f"/api/runs/{bad.manifest.run_id}")
        assert direct.status_code == 500
        assert direct.json()["detail"] == {
            "code": "invalid_structure",
            "artifact": "manifest.json",
            "message": "artifact manifest.json has an invalid structure",
        }


@pytest.mark.asyncio
async def test_decisions_endpoint_exposes_agent_reasoning(tmp_path: Path) -> None:
    result = await SimulationRunner().run(
        _run_request(seed=5), FixturePolicy(), FileEventStore(tmp_path)
    )
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/runs/{result.manifest.run_id}/decisions")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == result.manifest.run_id
    assert body["nodes"], "fixture run should produce at least one decision"

    departments = {"Engineering", "Executive"}
    for node in body["nodes"]:
        assert isinstance(node["sequence"], int)
        assert node["agent_id"]
        assert node["policy"] == "fixture"
        assert node["context_hash"]
        report = node["report"]
        assert report["agent_id"] == node["agent_id"]
        assert report["department"] in departments
        assert isinstance(report["depth"], int)
        assert isinstance(report["tick"], int)
        assert isinstance(report["confidence"], (int, float))
        assert isinstance(report["explanation"], str) and report["explanation"]
        assert isinstance(node["actions"], list)
        assert isinstance(node["provider_metadata"], dict)


@pytest.mark.asyncio
async def test_decisions_endpoint_404_for_missing_run(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/runs/does-not-exist/decisions")
    assert response.status_code == 404



@pytest.mark.asyncio
async def test_api_returns_structured_errors_for_corruption_filters_and_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _run_request(seed=9)
    result = await SimulationRunner().run(request, FixturePolicy(), FileEventStore(tmp_path))
    (result.run_directory / "metrics.json").write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("GROUNDLINE_MODEL", "")
    experiment = ExperimentRequest(
        name="missing-model-check",
        seeds=(1,),
        scenario=request.scenario,
        organization=request.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=0),
            "pressure": TreatmentConfig(incentive_pressure=1, attention_budget=0),
        },
        analysis=_analysis("control", "pressure"),
    )
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        corrupt = await client.get(f"/api/runs/{result.manifest.run_id}")
        bad_filter = await client.get(
            f"/api/runs/{result.manifest.run_id}/evidence", params={"depth": -1}
        )
        missing_job = await client.get("/api/jobs/missing")
        missing_model = await client.post(
            "/api/experiments",
            json={
                "experiment": experiment.model_dump(mode="json"),
                "policy": "record",
            },
        )

    assert corrupt.status_code == 500
    assert corrupt.json()["detail"] == {
        "code": "invalid_json",
        "artifact": "metrics.json",
        "message": "artifact metrics.json is not valid JSON",
    }
    assert bad_filter.status_code == 422
    assert missing_job.status_code == 404
    assert missing_model.status_code == 503
    assert "model" in missing_model.json()["detail"].lower()


@pytest.mark.asyncio
async def test_api_returns_typed_errors_for_structurally_malformed_artifacts(
    tmp_path: Path,
) -> None:
    result = await SimulationRunner().run(
        _run_request(seed=11), FixturePolicy(), FileEventStore(tmp_path)
    )
    metrics_path = result.run_directory / "metrics.json"
    original_metrics = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text("{}\n", encoding="utf-8")
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        malformed_metrics = await client.get(f"/api/runs/{result.manifest.run_id}")

    metrics_path.write_text(original_metrics, encoding="utf-8")
    manifest_path = result.run_directory / "manifest.json"
    manifest_path.write_text('{"run_id":"broken"}\n', encoding="utf-8")
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        malformed_manifest = await client.get("/api/runs")

    assert malformed_metrics.status_code == 500
    assert malformed_metrics.json()["detail"] == {
        "code": "invalid_structure",
        "artifact": "metrics.json",
        "message": "artifact metrics.json has an invalid structure",
    }
    # A corrupt manifest must not poison the run listing: the index skips the
    # bad run and returns the survivors (here: none) instead of a 500.
    assert malformed_manifest.status_code == 200
    assert malformed_manifest.json() == []


@pytest.mark.asyncio
async def test_api_launches_and_reports_a_new_intervention_experiment(tmp_path: Path) -> None:
    run = _run_request(seed=3)
    experiment = ExperimentRequest(
        name="launched-intervention",
        seeds=(3, 7),
        scenario=run.scenario,
        organization=run.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=1),
            "intervention": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
        },
        analysis=_analysis("control", "intervention"),
        max_concurrency=2,
    )
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        launch = await client.post(
            "/api/experiments",
            json={"experiment": experiment.model_dump(mode="json"), "policy": "fixture"},
        )
        assert launch.status_code == 202
        job = launch.json()
        for _ in range(100):
            status = (await client.get(f"/api/jobs/{job['job_id']}")).json()
            if status["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        detail = await client.get("/api/experiments/launched-intervention")

    assert status["status"] == "completed"
    assert status["failed_runs"] == 0
    assert detail.status_code == 200
    assert len(detail.json()["runs"]) == 4


@pytest.mark.asyncio
async def test_job_status_survives_app_recreation_and_recovers_expired_claim(
    tmp_path: Path,
) -> None:
    run = _run_request(seed=13)
    experiment = ExperimentRequest(
        name="durable-restart",
        seeds=(13,),
        scenario=run.scenario,
        organization=run.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=1),
            "intervention": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
        },
        analysis=_analysis("control", "intervention"),
    )
    first_transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=first_transport, base_url="http://first") as client:
        launch = await client.post(
            "/api/experiments",
            json={"experiment": experiment.model_dump(mode="json"), "policy": "fixture"},
        )
        job_id = launch.json()["job_id"]
        for _ in range(200):
            first_status = (await client.get(f"/api/jobs/{job_id}")).json()
            if first_status["status"] == "completed":
                break
            await asyncio.sleep(0.01)

    assert first_status["status"] == "completed"
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'running', completed_runs = 0, lease_owner = 'dead-process',
                lease_expires_at = 0
            WHERE job_id = ?
            """,
            (job_id,),
        )

    second_transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=second_transport, base_url="http://second") as client:
        for _ in range(200):
            recovered = (await client.get(f"/api/jobs/{job_id}")).json()
            if recovered["status"] == "completed":
                break
            await asyncio.sleep(0.01)

    assert recovered["status"] == "completed"
    state = json.loads(
        (tmp_path / "experiments" / experiment.name / "experiment-state.json").read_text()
    )
    assert all(record["attempts"] == 1 for record in state["runs"].values())


@pytest.mark.asyncio
async def test_two_app_instances_claim_a_job_once(tmp_path: Path) -> None:
    run = _run_request(seed=17)
    experiment = ExperimentRequest(
        name="two-app-claim",
        seeds=(17,),
        scenario=run.scenario,
        organization=run.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=1),
            "intervention": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
        },
        analysis=_analysis("control", "intervention"),
    )
    first = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url="http://first"
    )
    second = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url="http://second"
    )
    payload = {"experiment": experiment.model_dump(mode="json"), "policy": "fixture"}
    async with first, second:
        first_launch, second_launch = await asyncio.gather(
            first.post("/api/experiments", json=payload),
            second.post("/api/experiments", json=payload),
        )
        assert first_launch.json()["job_id"] == second_launch.json()["job_id"]
        job_id = first_launch.json()["job_id"]
        for _ in range(200):
            status_body = (await second.get(f"/api/jobs/{job_id}")).json()
            if status_body["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)

    assert status_body["status"] == "completed"
    state = json.loads(
        (tmp_path / "experiments" / experiment.name / "experiment-state.json").read_text()
    )
    assert all(record["attempts"] == 1 for record in state["runs"].values())


@pytest.mark.asyncio
async def test_experiment_name_conflicts_return_409_before_background_execution(
    tmp_path: Path,
) -> None:
    run = _run_request(seed=19)
    experiment = ExperimentRequest(
        name="conflicting-launch",
        seeds=(19,),
        scenario=run.scenario,
        organization=run.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=1),
            "intervention": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
        },
        analysis=_analysis("control", "intervention"),
    )
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/api/experiments",
            json={"experiment": experiment.model_dump(mode="json"), "policy": "fixture"},
        )
        conflict = await client.post(
            "/api/experiments",
            json={
                "experiment": experiment.model_dump(mode="json"),
                "policy": "locked",
            },
        )

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == ("experiment name already belongs to a different launch")


@pytest.mark.asyncio
async def test_launch_rechecks_experiment_name_conflict_inside_transaction(tmp_path: Path) -> None:
    run = _run_request(seed=23)
    experiment = ExperimentRequest(
        name="transactional-conflict",
        seeds=(23,),
        scenario=run.scenario,
        organization=run.organization,
        treatments={
            "control": TreatmentConfig(incentive_pressure=0, attention_budget=1),
            "intervention": TreatmentConfig(incentive_pressure=0.9, attention_budget=0),
        },
        analysis=_analysis("control", "intervention"),
    )
    first = LaunchExperiment(experiment=experiment, policy="fixture")
    second = LaunchExperiment(experiment=experiment, policy="locked")
    manager = ExperimentJobManager(tmp_path)

    await manager.launch(first)

    with pytest.raises(ExperimentLaunchConflict):
        await manager.launch(second)


@pytest.mark.asyncio
async def test_openapi_declares_job_and_artifact_error_contracts(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        schema = (await client.get("/openapi.json")).json()

    assert schema["info"]["version"] == __version__

    def response_schema(path: str, method: str, status_code: int) -> dict[str, object]:
        return schema["paths"][path][method]["responses"][str(status_code)]["content"][
            "application/json"
        ]["schema"]

    def response_model(path: str, method: str, status_code: int) -> str:
        return str(response_schema(path, method, status_code)["$ref"])

    assert response_model("/api/jobs/{job_id}", "get", 404).endswith("MessageErrorResponse")
    assert response_model("/api/runs/{run_id}", "get", 200).endswith("RunDetailResponse")
    assert response_model("/api/runs/{run_id}", "get", 500).endswith("ArtifactErrorResponse")
    assert response_model("/api/runs/{run_id}/evidence", "get", 200).endswith("EvidenceResponse")
    assert response_model("/api/runs/{run_id}/evidence", "get", 422).endswith(
        "ValidationErrorResponse"
    )
    assert response_model("/api/experiments/{name}", "get", 200).endswith(
        "ExperimentDetailResponse"
    )
    experiment_list = response_schema("/api/experiments", "get", 200)
    assert str(experiment_list["items"]["$ref"]).endswith("ExperimentSummaryResponse")
    for status_code, model in (
        (202, "JobStatus"),
        (409, "MessageErrorResponse"),
        (422, "ValidationErrorResponse"),
        (500, "ArtifactErrorResponse"),
        (503, "MessageErrorResponse"),
    ):
        assert response_model("/api/experiments", "post", status_code).endswith(model)


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


def _run_request(*, seed: int) -> RunRequest:
    return RunRequest(
        scenario=ScenarioConfig(
            max_ticks=1,
            shock_tick=1,
            shock_item_id="api",
            shock_severity=0.6,
            work_items=(
                WorkItemConfig(
                    id="api",
                    department="Engineering",
                    business_value=1,
                    effort=2,
                    deadline_tick=1,
                ),
            ),
        ),
        organization=OrganizationConfig(
            agents=(
                AgentConfig(
                    id="worker", role="contributor", manager_id="exec", department="Engineering"
                ),
                AgentConfig(id="exec", role="executive", department="Executive"),
            )
        ),
        treatment=TreatmentConfig(incentive_pressure=0.8, attention_budget=0),
        seed=seed,
    )
