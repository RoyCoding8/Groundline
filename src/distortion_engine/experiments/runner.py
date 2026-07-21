from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from distortion_engine import ENGINE_FINGERPRINT
from distortion_engine.events.artifacts import ArtifactCorruptError, verify_run_artifacts
from distortion_engine.events.store import FileEventStore, canonical_json
from distortion_engine.experiments.analysis import (
    AnalysisRun,
    AnalysisSpecification,
    ExperimentAnalysis,
    analyze_experiment,
)
from distortion_engine.metrics.outcomes import (
    OperationalHarm,
    OperationalHarmRegret,
    OutcomeSpecification,
    OutcomeUnavailableReason,
    RunOutcomes,
    compare_operational_harm,
)
from distortion_engine.organization.models import OrganizationConfig
from distortion_engine.organization.topology import ReportingSpan
from distortion_engine.policy.models import AgentPolicy
from distortion_engine.policy.oracle import OraclePolicy
from distortion_engine.replay.engine import ReplayEngine
from distortion_engine.simulation.runner import RunRequest, SimulationRunner, TreatmentConfig
from distortion_engine.statistics.inference import (
    PairedAnalyzer,
    adjust_holm,
    factorial_contrasts,
    minimum_sign_flip_pairs,
)
from distortion_engine.world.models import ScenarioConfig


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    seeds: tuple[int, ...] = Field(min_length=1)
    scenario: ScenarioConfig
    organization: OrganizationConfig
    treatments: dict[str, TreatmentConfig]
    outcome_specification: OutcomeSpecification = Field(default_factory=OutcomeSpecification)
    analysis: AnalysisSpecification
    max_concurrency: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def validate_design(self) -> ExperimentRequest:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment seeds must be unique")
        if len(self.treatments) < 2:
            raise ValueError("an experiment requires at least two treatments")
        if any(
            not name or "/" in name or "\\" in name or name in {".", ".."}
            for name in self.treatments
        ):
            raise ValueError("treatment names must be non-empty path-safe labels")
        declared_treatments = {
            treatment
            for contrast in self.analysis.contrasts
            for treatment in (contrast.baseline, contrast.intervention)
        }
        unknown_treatments = declared_treatments.difference(self.treatments)
        if unknown_treatments:
            raise ValueError(
                f"analysis references unknown treatments: {sorted(unknown_treatments)}"
            )
        declared_escalation_thresholds = {
            sensitivity.threshold
            for sensitivity in self.analysis.sensitivities
            if sensitivity.kind == "alternative_escalation_threshold"
            and sensitivity.threshold is not None
        }
        missing_escalation_thresholds = declared_escalation_thresholds.difference(
            self.outcome_specification.escalation_sensitivity_thresholds
        )
        if missing_escalation_thresholds:
            raise ValueError(
                "analysis escalation thresholds are not computed by the outcome specification: "
                f"{sorted(missing_escalation_thresholds)}"
            )
        return self


type OutcomeAvailability = Literal["available", "unavailable"]


class IndexedRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    treatment: str
    run_id: str
    run_directory: str
    executive_optimism_bias: float | None
    executive_absolute_error: float | None
    executive_outcome_status: OutcomeAvailability
    executive_outcome_reason: OutcomeUnavailableReason | None = None
    upward_amplification: float | None
    upward_amplification_status: OutcomeAvailability
    upward_amplification_reason: OutcomeUnavailableReason | None = None
    incentive_pressure: float
    attention_budget: int
    reporting_span: ReportingSpan | None
    outcomes: RunOutcomes
    incident_duration: int
    remediation_cost: float
    oracle_incident_duration: int | None = None
    incident_regret: int | None = None
    oracle_operational_harm: float | None = None
    oracle_regret: float | None = None
    operational_harm_regret: OperationalHarmRegret | None = None

    @model_validator(mode="after")
    def validate_outcome_availability(self) -> IndexedRun:
        if self.executive_outcome_status == "available":
            if (
                self.executive_optimism_bias is None
                or self.executive_absolute_error is None
                or self.executive_outcome_reason is not None
            ):
                raise ValueError("available executive outcome requires values only")
        elif (
            self.executive_optimism_bias is not None
            or self.executive_absolute_error is not None
            or self.executive_outcome_reason is None
        ):
            raise ValueError("unavailable executive outcome requires a reason only")
        if self.upward_amplification_status == "available":
            if self.upward_amplification is None or self.upward_amplification_reason is not None:
                raise ValueError("available upward amplification requires a value only")
        elif self.upward_amplification is not None or self.upward_amplification_reason is None:
            raise ValueError("unavailable upward amplification requires a reason")
        return self


class ExperimentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    runs: tuple[IndexedRun, ...]
    index_path: Path
    analysis_path: Path
    state_path: Path
    executed_runs: int
    resumed_runs: int
    failed_runs: int


@dataclass(frozen=True)
class _Task:
    key: str
    kind: Literal["treatment", "oracle"]
    seed: int
    treatment: str | None
    request: RunRequest
    policy: AgentPolicy

    @property
    def policy_fingerprint(self) -> str:
        return str(getattr(self.policy, "fingerprint", self.policy.name))


class ExperimentRunner:
    """Resumable, bounded experiment orchestration over immutable run artifacts."""

    def __init__(self, *, simulation_runner: Any | None = None) -> None:
        self.simulation_runner = simulation_runner or SimulationRunner()

    async def run(
        self,
        request: ExperimentRequest,
        policy: AgentPolicy,
        store: FileEventStore,
    ) -> ExperimentResult:
        experiment_dir = store.root / "experiments" / request.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        request_payload = request.model_dump(mode="json")
        request_hash = hashlib.sha256(canonical_json(request_payload).encode()).hexdigest()
        policy_fingerprint = str(getattr(policy, "fingerprint", policy.name))
        state_path = experiment_dir / "experiment-state.json"
        state = self._load_state(
            state_path,
            request_hash=request_hash,
            policy_fingerprint=policy_fingerprint,
            max_concurrency=request.max_concurrency,
        )
        (experiment_dir / "experiment-request.json").write_text(
            canonical_json(request_payload) + "\n", encoding="utf-8"
        )
        tasks = self._tasks(request, policy)
        for task in tasks:
            state["runs"].setdefault(
                task.key,
                {
                    "kind": task.kind,
                    "seed": task.seed,
                    "treatment": task.treatment,
                    "status": "pending",
                    "attempts": 0,
                    "run_id": None,
                    "run_directory": None,
                    "error": None,
                    "policy_fingerprint": task.policy_fingerprint,
                },
            )
        self._write_json_atomic(state_path, state)

        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(request.max_concurrency)
        counters = {"executed": 0, "resumed": 0}

        async def execute(task: _Task) -> None:
            record = state["runs"][task.key]
            valid, reason = await self._validate_record(record, task, store)
            async with lock:
                if record["status"] == "completed" and valid:
                    counters["resumed"] += 1
                    return
                if record["status"] == "completed" and not valid:
                    self._quarantine_invalid(record, task, experiment_dir, store)
                    record["error"] = f"invalid finalized artifact: {reason}"
                record["status"] = "running"
                record["attempts"] += 1
                self._write_json_atomic(state_path, state)
            try:
                async with semaphore:
                    result = await self.simulation_runner.run(task.request, task.policy, store)
                async with lock:
                    record.update(
                        {
                            "status": "completed",
                            "run_id": result.manifest.run_id,
                            "run_directory": str(result.run_directory.resolve()),
                            "error": None,
                        }
                    )
                    counters["executed"] += 1
                    self._write_json_atomic(state_path, state)
            except Exception as error:
                async with lock:
                    record.update(
                        {
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    counters["executed"] += 1
                    self._write_json_atomic(state_path, state)

        await asyncio.gather(*(execute(task) for task in tasks))

        oracle_results = {
            (int(record["seed"]), str(record["treatment"])): self._oracle_result(
                Path(record["run_directory"])
            )
            for record in state["runs"].values()
            if record["kind"] == "oracle" and record["status"] == "completed"
        }
        indexed: list[IndexedRun] = []
        failure_rows: list[dict[str, Any]] = []
        for task in tasks:
            record = state["runs"][task.key]
            if task.kind == "oracle":
                if record["status"] == "failed":
                    failure_rows.append(self._failure_row(task, record))
                continue
            if record["status"] != "completed":
                failure_rows.append(self._failure_row(task, record))
                continue
            indexed.append(
                self._summarize_run(
                    task,
                    Path(record["run_directory"]),
                    oracle_results.get(
                        (task.seed, task.request.treatment.reporting_span or "base")
                    ),
                )
            )

        index_path = experiment_dir / "run-index.jsonl"
        index_rows = [
            {"status": "completed", **run.model_dump(mode="json")} for run in indexed
        ] + failure_rows
        index_path.write_bytes("".join(f"{canonical_json(row)}\n" for row in index_rows).encode())
        analysis = self._analyze(request, indexed)
        analysis["execution"] = {
            "executed_runs": counters["executed"],
            "resumed_runs": counters["resumed"],
            "failed_runs": len(failure_rows),
            "max_concurrency": request.max_concurrency,
        }
        analysis_path = experiment_dir / "analysis.json"
        analysis_path.write_text(canonical_json(analysis) + "\n", encoding="utf-8")
        self._write_exports(experiment_dir, indexed, request.name, analysis)
        state["status"] = "completed" if not failure_rows else "completed_with_failures"
        self._write_json_atomic(state_path, state)
        return ExperimentResult(
            runs=tuple(indexed),
            index_path=index_path,
            analysis_path=analysis_path,
            state_path=state_path,
            executed_runs=counters["executed"],
            resumed_runs=counters["resumed"],
            failed_runs=len(failure_rows),
        )

    @staticmethod
    def _tasks(request: ExperimentRequest, policy: AgentPolicy) -> list[_Task]:
        tasks = [
            _Task(
                key=f"treatment:{name}:seed:{seed}",
                kind="treatment",
                seed=seed,
                treatment=name,
                request=RunRequest(
                    scenario=request.scenario,
                    organization=request.organization,
                    treatment=treatment,
                    seed=seed,
                    outcome_specification=request.outcome_specification,
                ).effective(),
                policy=policy,
            )
            for seed in request.seeds
            for name, treatment in request.treatments.items()
        ]
        reporting_spans = sorted(
            {treatment.reporting_span for treatment in request.treatments.values()},
            key=lambda span: span or "",
        )
        tasks.extend(
            _Task(
                key=f"oracle:{reporting_span or 'base'}:seed:{seed}",
                kind="oracle",
                seed=seed,
                treatment=reporting_span or "base",
                request=RunRequest(
                    scenario=request.scenario,
                    organization=request.organization,
                    treatment=TreatmentConfig(
                        incentive_pressure=0,
                        attention_budget=max(request.organization.spans.values()),
                        reporting_span=reporting_span,
                    ),
                    seed=seed,
                    outcome_specification=request.outcome_specification,
                ).effective(),
                policy=OraclePolicy(),
            )
            for seed in request.seeds
            for reporting_span in reporting_spans
        )
        return tasks

    @staticmethod
    def _load_state(
        path: Path,
        *,
        request_hash: str,
        policy_fingerprint: str,
        max_concurrency: int,
    ) -> dict[str, Any]:
        if not path.exists():
            return {
                "schema_version": 1,
                "request_hash": request_hash,
                "policy_fingerprint": policy_fingerprint,
                "engine_fingerprint": ENGINE_FINGERPRINT,
                "max_concurrency": max_concurrency,
                "status": "running",
                "runs": {},
            }
        state = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if state.get("request_hash") != request_hash:
            raise ValueError("experiment name already belongs to a different request")
        if state.get("policy_fingerprint") != policy_fingerprint:
            raise ValueError("experiment name already belongs to a different policy")
        state["max_concurrency"] = max_concurrency
        state["engine_fingerprint"] = ENGINE_FINGERPRINT
        state["status"] = "running"
        return state

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    async def _validate_record(
        record: dict[str, Any], task: _Task, store: FileEventStore
    ) -> tuple[bool, str]:
        directory_text = record.get("run_directory")
        if not directory_text:
            return False, "run directory is absent"
        directory = Path(directory_text)
        try:
            if not directory.resolve().is_relative_to(store.root.resolve()):
                return False, "run directory escapes artifact root"
        except OSError:
            return False, "run directory cannot be resolved"
        try:
            verify_run_artifacts(
                directory,
                expected_request=task.request,
                expected_policy_fingerprint=task.policy_fingerprint,
                expected_engine_fingerprint=ENGINE_FINGERPRINT,
            )
            replay = await ReplayEngine().replay(directory)
        except ArtifactCorruptError as error:
            return False, f"{error.code}: {error.filename}"
        except (OSError, ValueError) as error:
            return False, f"replay failed: {type(error).__name__}"
        if not replay.equivalent:
            return False, "reconstructed artifacts differ"
        return True, "verified"

    @staticmethod
    def _quarantine_invalid(
        record: dict[str, Any], task: _Task, experiment_dir: Path, store: FileEventStore
    ) -> None:
        directory_text = record.get("run_directory")
        if not directory_text:
            return
        directory = Path(directory_text)
        if not directory.exists() or not directory.resolve().is_relative_to(store.root.resolve()):
            return
        quarantine = experiment_dir / "quarantine"
        quarantine.mkdir(exist_ok=True)
        safe_key = task.key.replace(":", "-")
        target = quarantine / f"{safe_key}-attempt-{record['attempts']}"
        if not target.exists():
            shutil.move(str(directory), str(target))

    @staticmethod
    def _failure_row(task: _Task, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "failed",
            "kind": task.kind,
            "seed": task.seed,
            "treatment": task.treatment,
            "attempts": record["attempts"],
            "error": record.get("error") or "run did not finalize",
        }

    @staticmethod
    def _events(directory: Path) -> list[dict[str, Any]]:
        artifacts = verify_run_artifacts(directory)
        return [event.model_dump(mode="json") for event in artifacts.events]

    @classmethod
    def _final_state(cls, directory: Path) -> dict[str, Any]:
        snapshots = [event for event in cls._events(directory) if event["kind"] == "truth_snapshot"]
        if not snapshots:
            raise ValueError("run must contain post-action truth snapshots")
        final_tick = max(int(snapshot["tick"]) for snapshot in snapshots)
        final_snapshots = [
            snapshot for snapshot in snapshots if int(snapshot["tick"]) == final_tick
        ]
        if len(final_snapshots) != 1:
            raise ValueError("run must contain exactly one final truth snapshot")
        return cast(dict[str, Any], final_snapshots[0]["payload"]["state"])

    @classmethod
    def _final_incident(cls, directory: Path) -> int:
        return int(cls._final_state(directory)["incident_duration"])

    @classmethod
    def _oracle_result(cls, directory: Path) -> tuple[int, OperationalHarm]:
        artifacts = verify_run_artifacts(directory)
        return cls._final_incident(directory), artifacts.metrics.outcomes.operational_harm

    @classmethod
    def _summarize_run(
        cls,
        task: _Task,
        directory: Path,
        oracle_result: tuple[int, OperationalHarm] | None,
    ) -> IndexedRun:
        artifacts = verify_run_artifacts(
            directory,
            expected_request=task.request,
            expected_policy_fingerprint=task.policy_fingerprint,
            expected_engine_fingerprint=ENGINE_FINGERPRINT,
        )
        root_id = task.request.organization.root_id
        outcomes = artifacts.metrics.outcomes
        root_outcomes = next(
            depth
            for depth in outcomes.depth
            if depth.depth == task.request.organization.depths[root_id]
        )
        executive_summary = root_outcomes.adverse_ticks
        executive_optimism_bias = (
            executive_summary.optimism_bias_mean if executive_summary is not None else None
        )
        executive_absolute_error = (
            executive_summary.absolute_error_mean if executive_summary is not None else None
        )
        executive_outcome_status: OutcomeAvailability = (
            "available" if executive_summary is not None else "unavailable"
        )
        executive_outcome_reason: OutcomeUnavailableReason | None = (
            None if executive_summary is not None else "no_adverse_reports"
        )
        state = cls._final_state(directory)
        manifest = artifacts.manifest
        incident_duration = int(state["incident_duration"])
        oracle_incident = oracle_result[0] if oracle_result is not None else None
        oracle_harm = oracle_result[1] if oracle_result is not None else None
        return IndexedRun(
            seed=task.seed,
            treatment=task.treatment or "",
            run_id=manifest.run_id,
            run_directory=str(directory.resolve()),
            executive_optimism_bias=executive_optimism_bias,
            executive_absolute_error=executive_absolute_error,
            executive_outcome_status=executive_outcome_status,
            executive_outcome_reason=executive_outcome_reason,
            upward_amplification=outcomes.upward_amplification.value,
            upward_amplification_status=outcomes.upward_amplification.status,
            upward_amplification_reason=outcomes.upward_amplification.reason,
            incentive_pressure=task.request.treatment.incentive_pressure,
            attention_budget=task.request.treatment.attention_budget,
            reporting_span=task.request.treatment.reporting_span,
            outcomes=outcomes,
            incident_duration=incident_duration,
            remediation_cost=float(state["remediation_cost"]),
            oracle_incident_duration=oracle_incident,
            incident_regret=(
                incident_duration - oracle_incident if oracle_incident is not None else None
            ),
            oracle_operational_harm=oracle_harm.index if oracle_harm is not None else None,
            oracle_regret=(
                outcomes.operational_harm.index - oracle_harm.index
                if oracle_harm is not None
                else None
            ),
            operational_harm_regret=(
                compare_operational_harm(outcomes.operational_harm, oracle_harm)
                if oracle_harm is not None
                else None
            ),
        )

    @classmethod
    def _analyze(cls, request: ExperimentRequest, indexed: list[IndexedRun]) -> dict[str, Any]:
        declared: ExperimentAnalysis = analyze_experiment(
            request.analysis,
            tuple(
                AnalysisRun(
                    seed=run.seed,
                    treatment=run.treatment,
                    outcomes=run.outcomes,
                    oracle_regret=run.oracle_regret,
                )
                for run in indexed
            ),
            requested_seeds=request.seeds,
        )
        comparisons: dict[str, dict[str, Any]] = {}
        for result in declared.contrasts:
            comparison = {
                "status": result.status,
                "baseline": result.baseline,
                "intervention": result.intervention,
                "outcome": result.outcome,
                "direction": result.direction,
                "family": result.family,
                "declaration_status": result.declaration_status,
                "requested_pairs": result.requested_pairs,
                "complete_pairs": result.complete_pairs,
                "missing_by_reason": result.missing_by_reason,
                "holm_adjusted_p_value": result.holm_adjusted_p_value,
            }
            if result.effect is not None:
                comparison.update(result.effect.model_dump(mode="json"))
            comparisons[result.id] = comparison

        sensitivities: dict[str, dict[str, Any]] = {}
        for sensitivity_result in declared.sensitivities:
            sensitivity = {
                "status": sensitivity_result.status,
                "contrast_id": sensitivity_result.contrast_id,
                "kind": sensitivity_result.kind,
                "baseline": sensitivity_result.baseline,
                "intervention": sensitivity_result.intervention,
                "outcome": sensitivity_result.outcome,
                "direction": sensitivity_result.direction,
                "missingness": sensitivity_result.missingness,
                "threshold": sensitivity_result.threshold,
                "requested_pairs": sensitivity_result.requested_pairs,
                "complete_pairs": sensitivity_result.complete_pairs,
                "missing_by_reason": sensitivity_result.missing_by_reason,
            }
            if sensitivity_result.effect is not None:
                sensitivity.update(sensitivity_result.effect.model_dump(mode="json"))
            sensitivities[sensitivity_result.id] = sensitivity

        complete_pairs = min(
            (result.complete_pairs for result in declared.contrasts),
            default=0,
        )
        family_sizes: dict[str, int] = {}
        for result in declared.contrasts:
            if result.declaration_status == "confirmatory" and result.effect is not None:
                family_sizes[result.family] = family_sizes.get(result.family, 0) + 1
        family_size = max(family_sizes.values(), default=1)
        minimum_pairs = minimum_sign_flip_pairs(alpha=0.05, family_size=family_size)
        diagnostics = {
            "requested_pairs": len(request.seeds),
            "complete_pairs": complete_pairs,
            "family_size": family_size,
            "minimum_pairs_for_unanimous_holm_005": minimum_pairs,
            "recommended_pairs_for_sensitivity": 12,
            "randomization_method": (
                "exact_sign_flip" if complete_pairs <= 20 else "monte_carlo_sign_flip"
            ),
            "minimum_reportable_randomization_p": (
                (2.0 ** (1 - complete_pairs) if complete_pairs <= 20 else 1 / 100_001)
                if complete_pairs
                else None
            ),
            "adequate_for_holm_resolution": complete_pairs >= minimum_pairs,
            "adequate_for_leave_one_out_sensitivity": complete_pairs >= 12,
        }
        analyzer = PairedAnalyzer(analysis_seed=request.analysis.seed)
        frame = pd.DataFrame([run.model_dump(mode="json") for run in indexed])
        factorial = cls._factorial_analysis(frame, analyzer)
        secondary = cls._secondary_repeated_measures(cls._report_frame(indexed))
        return {
            "preregistration": request.analysis.model_dump(mode="json"),
            "baseline": request.analysis.contrasts[0].baseline,
            "unit_of_analysis": declared.unit_of_analysis,
            "multiplicity": declared.multiplicity,
            "comparisons": comparisons,
            "sensitivities": sensitivities,
            "factorial_effects": factorial,
            "secondary_repeated_measures": secondary,
            "design_diagnostics": diagnostics,
        }

    @staticmethod
    def _factorial_analysis(frame: pd.DataFrame, analyzer: PairedAnalyzer) -> dict[str, Any]:
        if frame.empty:
            return {"status": "unavailable", "reason": "no completed runs"}
        try:
            effects = factorial_contrasts(frame, "executive_optimism_bias")
        except ValueError as error:
            return {"status": "not_applicable", "reason": str(error)}
        results: dict[str, dict[str, Any]] = {}
        for name, values in effects.items():
            intervention = {cast(int, seed): float(value) for seed, value in values.items()}
            if not intervention:
                continue
            results[name] = analyzer.compare(
                baseline={seed: 0.0 for seed in intervention},
                intervention=intervention,
                contrast_id=f"factorial:{name}",
            ).model_dump(mode="json")
        adjusted = adjust_holm(result["p_value"] for result in results.values())
        for result, adjusted_value in zip(results.values(), adjusted, strict=True):
            result["holm_adjusted_p_value"] = adjusted_value
        return {
            "status": "complete" if results else "unavailable",
            "estimands": {
                "incentive": "within-seed marginal high minus low incentive",
                "attention": "within-seed marginal high minus low manager attention",
                "interaction": "within-seed difference in differences",
            },
            "results": results,
        }

    @classmethod
    def _report_frame(cls, indexed: list[IndexedRun]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run in indexed:
            for event in cls._events(Path(run.run_directory)):
                if event["kind"] != "metric":
                    continue
                metric = cast(dict[str, Any], event["payload"])
                rows.append(
                    {
                        "seed": run.seed,
                        "treatment": run.treatment,
                        "agent_id": metric["agent_id"],
                        "department": metric["department"],
                        "depth": metric["depth"],
                        "tick": metric["tick"],
                        "optimism_bias": metric["optimism_bias"],
                        "incentive_pressure": run.incentive_pressure,
                        "attention_budget": run.attention_budget,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _secondary_repeated_measures(frame: pd.DataFrame) -> dict[str, Any]:
        """Fit the preregistered report-level mixed-effects model."""
        if frame.empty or frame["seed"].nunique() < 3:
            return {
                "status": "not_identified",
                "reason": "at least three complete seeds are required",
            }
        if frame["incentive_pressure"].nunique() != 2 or frame["attention_budget"].nunique() != 2:
            return {
                "status": "not_applicable",
                "reason": "report-level mixed model requires a complete 2x2 design",
            }
        import warnings

        import statsmodels.formula.api as smf  # type: ignore[import-untyped]
        from numpy.linalg import LinAlgError

        modeled = frame.copy()
        max_attention = max(float(modeled["attention_budget"].max()), 1.0)
        modeled["attention_scaled"] = modeled["attention_budget"] / max_attention
        formula = (
            "optimism_bias ~ incentive_pressure * attention_scaled + "
            "incentive_pressure * depth + C(department) + C(tick)"
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = smf.mixedlm(
                    formula,
                    modeled,
                    groups=modeled["seed"],
                    re_formula="1",
                    vc_formula={"agent": "0 + C(agent_id)"},
                ).fit(
                    reml=False,
                    method=["lbfgs", "powell", "cg"],
                    maxiter=500,
                    disp=False,
                )
        except (ValueError, ArithmeticError, RuntimeError, LinAlgError) as error:
            return {
                "status": "failed",
                "model": "report-level mixed effects with seed and agent random intercepts",
                "reason": f"{type(error).__name__}: {error}",
                "n_observations": len(modeled),
                "n_seeds": int(modeled["seed"].nunique()),
                "n_agents": int(modeled["agent_id"].nunique()),
            }
        return {
            "status": "fit" if fitted.converged else "not_converged",
            "model": "report-level mixed effects with seed and agent random intercepts",
            "formula": formula,
            "coefficients": {key: float(value) for key, value in fitted.fe_params.items()},
            "standard_errors": {key: float(fitted.bse[key]) for key in fitted.fe_params.index},
            "converged": bool(fitted.converged),
            "log_likelihood": float(fitted.llf),
            "seed_random_intercept_variance": float(fitted.cov_re.iloc[0, 0]),
            "agent_random_intercept_variance": (
                float(fitted.vcomp[0]) if len(fitted.vcomp) else None
            ),
            "warnings": tuple(str(warning.message) for warning in caught),
            "n_observations": len(modeled),
            "n_seeds": int(modeled["seed"].nunique()),
            "n_agents": int(modeled["agent_id"].nunique()),
        }

    @classmethod
    def _write_exports(
        cls,
        experiment_dir: Path,
        indexed: list[IndexedRun],
        name: str,
        analysis: dict[str, Any],
    ) -> None:
        frame = pd.DataFrame([run.model_dump(mode="json") for run in indexed])
        frame.to_csv(experiment_dir / "outcomes.csv", index=False)
        frame.to_parquet(experiment_dir / "outcomes.parquet", index=False)
        (experiment_dir / "report.md").write_text(
            cls._markdown_report(name, analysis), encoding="utf-8"
        )

    @staticmethod
    def _markdown_report(name: str, analysis: dict[str, Any]) -> str:
        lines = [
            f"# Experiment report: {name}",
            "",
            "Primary inference uses paired seeds and two-sided sign-flip tests. "
            "Each row identifies exact enumeration or seeded Monte Carlo inference; "
            "ticks and agent messages are not treated as independent replicates.",
            "",
            "| Contrast | Paired mean | 95% CI (method) | Randomization p (method) | Holm p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for contrast, result in analysis["comparisons"].items():
            if result["status"] != "complete":
                lines.append(
                    f"| {contrast} | unavailable | unavailable | unavailable | unavailable |"
                )
                continue
            interval_method = "BCa" if result["interval_method"] == "bca" else "percentile"
            randomization_method = (
                "exact sign-flip"
                if result["p_value_method"] == "exact_sign_flip"
                else "Monte Carlo sign-flip"
            )
            draw_label = (
                "permutations" if result["p_value_method"] == "exact_sign_flip" else "draws"
            )
            holm_value = result["holm_adjusted_p_value"]
            holm_text = "not adjusted" if holm_value is None else f"{holm_value:.6g}"
            lines.append(
                f"| {contrast} | {result['mean_difference']:.4f} | "
                f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] ({interval_method}) | "
                f"{result['p_value']:.6g} "
                f"({randomization_method}; {result['randomization_draws']} {draw_label}) | "
                f"{holm_text} |"
            )
        sensitivities = analysis.get("sensitivities", {})
        if sensitivities:
            lines.extend(
                [
                    "",
                    "## Preregistered sensitivities",
                    "",
                    "| Sensitivity | Kind | Outcome | Paired mean | 95% CI |",
                    "| --- | --- | --- | ---: | ---: |",
                ]
            )
            for sensitivity, result in sensitivities.items():
                if result["status"] != "complete":
                    lines.append(
                        f"| {sensitivity} | {result['kind']} | {result['outcome']} | "
                        "unavailable | unavailable |"
                    )
                    continue
                lines.append(
                    f"| {sensitivity} | {result['kind']} | {result['outcome']} | "
                    f"{result['mean_difference']:.4f} | "
                    f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] |"
                )
        diagnostics = analysis["design_diagnostics"]
        lines.extend(
            [
                "",
                "## Design diagnostics",
                "",
                f"Complete paired seeds: {diagnostics['complete_pairs']}",
                f"Randomization-test Holm resolution adequate: "
                f"{diagnostics['adequate_for_holm_resolution']}",
                "",
                "## Secondary repeated-measures model",
                "",
                f"Status: {analysis['secondary_repeated_measures']['status']}",
                "",
                "The full seed-level outcome table is available as CSV and Parquet.",
            ]
        )
        return "\n".join(lines) + "\n"
