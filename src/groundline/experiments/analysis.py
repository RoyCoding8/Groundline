from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundline.metrics.outcomes import RunOutcomes
from groundline.statistics.inference import PairedAnalyzer, PairedEffect, adjust_holm

type AnalysisOutcome = Literal[
    "upward_amplification",
    "pre_release_upward_amplification",
    "executive_optimism_bias_adverse_mean",
    "executive_optimism_bias_adverse_median",
    "executive_optimism_bias_all_mean",
    "executive_optimism_bias_all_median",
    "executive_absolute_error_adverse_mean",
    "executive_absolute_error_all_mean",
    "executive_vector_loss_adverse_mean",
    "executive_vector_loss_all_mean",
    "executive_equal_weight_vector_loss_adverse_mean",
    "executive_equal_weight_vector_loss_all_mean",
    "executive_progress_error_adverse_mean",
    "executive_progress_error_all_mean",
    "executive_quality_error_adverse_mean",
    "executive_quality_error_all_mean",
    "executive_schedule_error_adverse_mean",
    "executive_schedule_error_all_mean",
    "executive_reliability_error_adverse_mean",
    "executive_reliability_error_all_mean",
    "executive_optimism_bias_pre_release_adverse_mean",
    "executive_optimism_bias_pre_release_adverse_median",
    "executive_optimism_bias_pre_release_all_mean",
    "executive_optimism_bias_pre_release_all_median",
    "executive_absolute_error_pre_release_adverse_mean",
    "executive_absolute_error_pre_release_all_mean",
    "executive_vector_loss_pre_release_adverse_mean",
    "executive_vector_loss_pre_release_all_mean",
    "edge_transformation",
    "pre_release_edge_transformation",
    "calibration_brier_score",
    "operational_harm",
    "oracle_regret",
    "escalation_delay_mean",
]
type ExpectedDirection = Literal["increase", "decrease", "two_sided"]
type AnalysisStatus = Literal["confirmatory", "exploratory"]
type MissingnessPolicy = Literal["complete_case", "fail_if_missing"]
type SensitivityKind = Literal[
    "adverse_vs_all_ticks",
    "mean_vs_median",
    "equal_health_dimension_weights",
    "progress_dimension_only",
    "quality_dimension_only",
    "schedule_dimension_only",
    "reliability_dimension_only",
    "exclude_post_release_ticks",
    "exclude_invalid_or_refused",
    "alternative_escalation_threshold",
]
type ContrastResultStatus = Literal["complete", "insufficient_pairs"]


class ContrastSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    baseline: str = Field(min_length=1)
    intervention: str = Field(min_length=1)
    outcome: AnalysisOutcome
    direction: ExpectedDirection
    family: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    status: AnalysisStatus

    @model_validator(mode="after")
    def validate_treatments(self) -> ContrastSpecification:
        if self.baseline == self.intervention:
            raise ValueError("contrast baseline and intervention must differ")
        return self


class SensitivitySpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    contrast_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    kind: SensitivityKind
    missingness: MissingnessPolicy | None = None
    threshold: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_parameters(self) -> SensitivitySpecification:
        if self.kind == "alternative_escalation_threshold":
            if self.threshold is None:
                raise ValueError(
                    "alternative escalation threshold sensitivity requires a threshold"
                )
        elif self.threshold is not None:
            raise ValueError(f"sensitivity kind {self.kind} does not accept a threshold")
        if self.kind == "exclude_invalid_or_refused" and self.missingness != "complete_case":
            raise ValueError(
                "invalid or refused response exclusion requires complete_case missingness"
            )
        return self


_SENSITIVITY_OUTCOMES: dict[tuple[SensitivityKind, AnalysisOutcome], AnalysisOutcome] = {
    (
        "adverse_vs_all_ticks",
        "executive_optimism_bias_adverse_mean",
    ): "executive_optimism_bias_all_mean",
    (
        "adverse_vs_all_ticks",
        "executive_optimism_bias_adverse_median",
    ): "executive_optimism_bias_all_median",
    (
        "adverse_vs_all_ticks",
        "executive_absolute_error_adverse_mean",
    ): "executive_absolute_error_all_mean",
    (
        "mean_vs_median",
        "executive_optimism_bias_adverse_mean",
    ): "executive_optimism_bias_adverse_median",
    (
        "mean_vs_median",
        "executive_optimism_bias_all_mean",
    ): "executive_optimism_bias_all_median",
    (
        "equal_health_dimension_weights",
        "executive_vector_loss_adverse_mean",
    ): "executive_equal_weight_vector_loss_adverse_mean",
    (
        "equal_health_dimension_weights",
        "executive_vector_loss_all_mean",
    ): "executive_equal_weight_vector_loss_all_mean",
    (
        "progress_dimension_only",
        "executive_vector_loss_adverse_mean",
    ): "executive_progress_error_adverse_mean",
    (
        "progress_dimension_only",
        "executive_vector_loss_all_mean",
    ): "executive_progress_error_all_mean",
    (
        "quality_dimension_only",
        "executive_vector_loss_adverse_mean",
    ): "executive_quality_error_adverse_mean",
    (
        "quality_dimension_only",
        "executive_vector_loss_all_mean",
    ): "executive_quality_error_all_mean",
    (
        "schedule_dimension_only",
        "executive_vector_loss_adverse_mean",
    ): "executive_schedule_error_adverse_mean",
    (
        "schedule_dimension_only",
        "executive_vector_loss_all_mean",
    ): "executive_schedule_error_all_mean",
    (
        "reliability_dimension_only",
        "executive_vector_loss_adverse_mean",
    ): "executive_reliability_error_adverse_mean",
    (
        "reliability_dimension_only",
        "executive_vector_loss_all_mean",
    ): "executive_reliability_error_all_mean",
    (
        "exclude_post_release_ticks",
        "upward_amplification",
    ): "pre_release_upward_amplification",
    (
        "exclude_post_release_ticks",
        "edge_transformation",
    ): "pre_release_edge_transformation",
    (
        "exclude_post_release_ticks",
        "executive_optimism_bias_adverse_mean",
    ): "executive_optimism_bias_pre_release_adverse_mean",
    (
        "exclude_post_release_ticks",
        "executive_optimism_bias_adverse_median",
    ): "executive_optimism_bias_pre_release_adverse_median",
    (
        "exclude_post_release_ticks",
        "executive_optimism_bias_all_mean",
    ): "executive_optimism_bias_pre_release_all_mean",
    (
        "exclude_post_release_ticks",
        "executive_optimism_bias_all_median",
    ): "executive_optimism_bias_pre_release_all_median",
    (
        "exclude_post_release_ticks",
        "executive_absolute_error_adverse_mean",
    ): "executive_absolute_error_pre_release_adverse_mean",
    (
        "exclude_post_release_ticks",
        "executive_absolute_error_all_mean",
    ): "executive_absolute_error_pre_release_all_mean",
    (
        "exclude_post_release_ticks",
        "executive_vector_loss_adverse_mean",
    ): "executive_vector_loss_pre_release_adverse_mean",
    (
        "exclude_post_release_ticks",
        "executive_vector_loss_all_mean",
    ): "executive_vector_loss_pre_release_all_mean",
    (
        "alternative_escalation_threshold",
        "escalation_delay_mean",
    ): "escalation_delay_mean",
}
for registered_outcome in cast(
    tuple[AnalysisOutcome, ...],
    get_args(AnalysisOutcome.__value__),
):
    _SENSITIVITY_OUTCOMES[("exclude_invalid_or_refused", registered_outcome)] = registered_outcome


class AnalysisSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    missingness: MissingnessPolicy
    contrasts: tuple[ContrastSpecification, ...] = Field(min_length=1)
    sensitivities: tuple[SensitivitySpecification, ...] = ()

    @model_validator(mode="after")
    def validate_declarations(self) -> AnalysisSpecification:
        contrasts_by_id = {contrast.id: contrast for contrast in self.contrasts}
        if len(contrasts_by_id) != len(self.contrasts):
            raise ValueError("analysis contrast ids must be unique")
        sensitivity_ids = [sensitivity.id for sensitivity in self.sensitivities]
        if len(set(sensitivity_ids)) != len(sensitivity_ids):
            raise ValueError("analysis sensitivity ids must be unique")
        duplicate_ids = set(contrasts_by_id).intersection(sensitivity_ids)
        if duplicate_ids:
            raise ValueError(f"analysis declaration ids overlap: {sorted(duplicate_ids)}")
        for sensitivity in self.sensitivities:
            contrast = contrasts_by_id.get(sensitivity.contrast_id)
            if contrast is None:
                raise ValueError(
                    f"sensitivity {sensitivity.id} references unknown contrast "
                    f"{sensitivity.contrast_id}"
                )
            if (sensitivity.kind, contrast.outcome) not in _SENSITIVITY_OUTCOMES:
                raise ValueError(
                    f"sensitivity {sensitivity.id} kind {sensitivity.kind} is not "
                    f"defined for outcome {contrast.outcome}"
                )
        return self


class AnalysisRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    treatment: str
    outcomes: RunOutcomes
    oracle_regret: float | None


class ContrastAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    baseline: str
    intervention: str
    outcome: AnalysisOutcome
    direction: ExpectedDirection
    family: str
    declaration_status: AnalysisStatus
    status: ContrastResultStatus
    requested_pairs: int = Field(ge=0)
    complete_pairs: int = Field(ge=0)
    missing_by_reason: dict[str, int]
    effect: PairedEffect | None
    holm_adjusted_p_value: float | None = Field(default=None, ge=0, le=1)


class SensitivityAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    contrast_id: str
    kind: SensitivityKind
    baseline: str
    intervention: str
    outcome: AnalysisOutcome
    direction: ExpectedDirection
    missingness: MissingnessPolicy
    threshold: float | None = Field(default=None, ge=0)
    status: ContrastResultStatus
    requested_pairs: int = Field(ge=0)
    complete_pairs: int = Field(ge=0)
    missing_by_reason: dict[str, int]
    effect: PairedEffect | None


class ExperimentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_seed: int
    unit_of_analysis: Literal["seed"] = "seed"
    multiplicity: Literal["Holm within declared confirmatory family"] = (
        "Holm within declared confirmatory family"
    )
    contrasts: tuple[ContrastAnalysis, ...]
    sensitivities: tuple[SensitivityAnalysis, ...]


_SUMMARY_OUTCOME_FIELDS: dict[
    AnalysisOutcome,
    tuple[Literal["all", "pre_release"], Literal["all_ticks", "adverse_ticks"], str],
] = {
    "executive_optimism_bias_adverse_mean": ("all", "adverse_ticks", "optimism_bias_mean"),
    "executive_optimism_bias_adverse_median": (
        "all",
        "adverse_ticks",
        "optimism_bias_median",
    ),
    "executive_optimism_bias_all_mean": ("all", "all_ticks", "optimism_bias_mean"),
    "executive_optimism_bias_all_median": ("all", "all_ticks", "optimism_bias_median"),
    "executive_absolute_error_adverse_mean": ("all", "adverse_ticks", "absolute_error_mean"),
    "executive_absolute_error_all_mean": ("all", "all_ticks", "absolute_error_mean"),
    "executive_vector_loss_adverse_mean": ("all", "adverse_ticks", "vector_loss_mean"),
    "executive_vector_loss_all_mean": ("all", "all_ticks", "vector_loss_mean"),
    "executive_equal_weight_vector_loss_adverse_mean": (
        "all",
        "adverse_ticks",
        "equal_weight_vector_loss_mean",
    ),
    "executive_equal_weight_vector_loss_all_mean": (
        "all",
        "all_ticks",
        "equal_weight_vector_loss_mean",
    ),
    "executive_progress_error_adverse_mean": ("all", "adverse_ticks", "progress_error_mean"),
    "executive_progress_error_all_mean": ("all", "all_ticks", "progress_error_mean"),
    "executive_quality_error_adverse_mean": ("all", "adverse_ticks", "quality_error_mean"),
    "executive_quality_error_all_mean": ("all", "all_ticks", "quality_error_mean"),
    "executive_schedule_error_adverse_mean": ("all", "adverse_ticks", "schedule_error_mean"),
    "executive_schedule_error_all_mean": ("all", "all_ticks", "schedule_error_mean"),
    "executive_reliability_error_adverse_mean": (
        "all",
        "adverse_ticks",
        "reliability_error_mean",
    ),
    "executive_reliability_error_all_mean": (
        "all",
        "all_ticks",
        "reliability_error_mean",
    ),
    "executive_optimism_bias_pre_release_adverse_mean": (
        "pre_release",
        "adverse_ticks",
        "optimism_bias_mean",
    ),
    "executive_optimism_bias_pre_release_adverse_median": (
        "pre_release",
        "adverse_ticks",
        "optimism_bias_median",
    ),
    "executive_optimism_bias_pre_release_all_mean": (
        "pre_release",
        "all_ticks",
        "optimism_bias_mean",
    ),
    "executive_optimism_bias_pre_release_all_median": (
        "pre_release",
        "all_ticks",
        "optimism_bias_median",
    ),
    "executive_absolute_error_pre_release_adverse_mean": (
        "pre_release",
        "adverse_ticks",
        "absolute_error_mean",
    ),
    "executive_absolute_error_pre_release_all_mean": (
        "pre_release",
        "all_ticks",
        "absolute_error_mean",
    ),
    "executive_vector_loss_pre_release_adverse_mean": (
        "pre_release",
        "adverse_ticks",
        "vector_loss_mean",
    ),
    "executive_vector_loss_pre_release_all_mean": (
        "pre_release",
        "all_ticks",
        "vector_loss_mean",
    ),
}


def _extract_outcome(
    run: AnalysisRun,
    outcome: AnalysisOutcome,
    *,
    escalation_threshold: float | None = None,
) -> tuple[float | None, str | None]:
    outcomes = run.outcomes
    if outcome == "upward_amplification":
        scalar = outcomes.upward_amplification
        return scalar.value, scalar.reason
    if outcome == "pre_release_upward_amplification":
        scalar = outcomes.pre_release_upward_amplification
        return scalar.value, scalar.reason
    summary_field = _SUMMARY_OUTCOME_FIELDS.get(outcome)
    if summary_field is not None:
        period, tick_filter, field = summary_field
        depth_rows = outcomes.depth if period == "all" else outcomes.pre_release_depth
        executive = min(depth_rows, key=lambda row: row.depth)
        summary = executive.all_ticks if tick_filter == "all_ticks" else executive.adverse_ticks
        if summary is None:
            return None, "no_adverse_reports"
        return float(getattr(summary, field)), None
    if outcome == "edge_transformation":
        scalar = outcomes.edge_transformation
        return scalar.value, scalar.reason
    if outcome == "pre_release_edge_transformation":
        scalar = outcomes.pre_release_edge_transformation
        return scalar.value, scalar.reason
    if outcome == "calibration_brier_score":
        return outcomes.calibration.brier_score, None
    if outcome == "operational_harm":
        return outcomes.operational_harm.index, None
    if outcome == "oracle_regret":
        return (
            (run.oracle_regret, None)
            if run.oracle_regret is not None
            else (None, "oracle_outcome_unavailable")
        )
    if escalation_threshold is None:
        delays = outcomes.escalation_delays
    else:
        threshold_outcome = next(
            (
                result
                for result in outcomes.escalation_sensitivities
                if result.threshold == escalation_threshold
            ),
            None,
        )
        if threshold_outcome is None:
            return None, "escalation_threshold_not_computed"
        delays = threshold_outcome.delays
    if not delays:
        return None, "no_escalation_conditions"
    return sum(delay.delay_ticks for delay in delays) / len(delays), None


def _paired_effect(
    *,
    analyzer: PairedAnalyzer,
    by_treatment: dict[str, dict[int, AnalysisRun]],
    requested_seeds: tuple[int, ...],
    baseline_treatment: str,
    intervention_treatment: str,
    outcome: AnalysisOutcome,
    analysis_id: str,
    missingness: MissingnessPolicy,
    escalation_threshold: float | None = None,
) -> tuple[PairedEffect | None, dict[str, int], int]:
    baseline_runs = by_treatment.get(baseline_treatment, {})
    intervention_runs = by_treatment.get(intervention_treatment, {})
    baseline: dict[int, float] = {}
    intervention: dict[int, float] = {}
    missing: Counter[str] = Counter()
    for seed in requested_seeds:
        baseline_run = baseline_runs.get(seed)
        intervention_run = intervention_runs.get(seed)
        if baseline_run is None and intervention_run is None:
            missing["missing_both_runs"] += 1
            continue
        if baseline_run is None:
            missing["missing_baseline_run"] += 1
            continue
        if intervention_run is None:
            missing["missing_intervention_run"] += 1
            continue
        baseline_value, baseline_reason = _extract_outcome(
            baseline_run,
            outcome,
            escalation_threshold=escalation_threshold,
        )
        intervention_value, intervention_reason = _extract_outcome(
            intervention_run,
            outcome,
            escalation_threshold=escalation_threshold,
        )
        if baseline_value is None or intervention_value is None:
            missing[baseline_reason or intervention_reason or "outcome_unavailable"] += 1
            continue
        baseline[seed] = baseline_value
        intervention[seed] = intervention_value
    if missing and missingness == "fail_if_missing":
        reasons = ", ".join(f"{name}={count}" for name, count in sorted(missing.items()))
        raise ValueError(f"analysis {analysis_id} has missing pairs: {reasons}")
    effect = (
        analyzer.compare(
            baseline=baseline,
            intervention=intervention,
            contrast_id=analysis_id,
        )
        if baseline
        else None
    )
    return effect, dict(sorted(missing.items())), len(baseline)


def analyze_experiment(
    specification: AnalysisSpecification,
    runs: tuple[AnalysisRun, ...],
    *,
    requested_seeds: tuple[int, ...],
) -> ExperimentAnalysis:
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("requested analysis seeds must be unique")
    unexpected_seeds = {run.seed for run in runs}.difference(requested_seeds)
    if unexpected_seeds:
        raise ValueError(f"analysis runs contain unexpected seeds: {sorted(unexpected_seeds)}")

    analyzer = PairedAnalyzer(analysis_seed=specification.seed)
    by_treatment: dict[str, dict[int, AnalysisRun]] = defaultdict(dict)
    for run in runs:
        treatment_runs = by_treatment[run.treatment]
        if run.seed in treatment_runs:
            raise ValueError(
                f"duplicate analysis run for treatment {run.treatment!r} and seed {run.seed}"
            )
        treatment_runs[run.seed] = run

    results: list[ContrastAnalysis] = []
    for contrast in specification.contrasts:
        effect, missing, complete_pairs = _paired_effect(
            analyzer=analyzer,
            by_treatment=by_treatment,
            requested_seeds=requested_seeds,
            baseline_treatment=contrast.baseline,
            intervention_treatment=contrast.intervention,
            outcome=contrast.outcome,
            analysis_id=contrast.id,
            missingness=specification.missingness,
        )
        results.append(
            ContrastAnalysis(
                id=contrast.id,
                baseline=contrast.baseline,
                intervention=contrast.intervention,
                outcome=contrast.outcome,
                direction=contrast.direction,
                family=contrast.family,
                declaration_status=contrast.status,
                status="complete" if effect is not None else "insufficient_pairs",
                requested_pairs=len(requested_seeds),
                complete_pairs=complete_pairs,
                missing_by_reason=missing,
                effect=effect,
            )
        )

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        if result.declaration_status == "confirmatory" and result.effect is not None:
            by_family[result.family].append(index)
    for indexes in by_family.values():
        p_values: list[float] = []
        for index in indexes:
            effect = results[index].effect
            if effect is None:
                raise AssertionError("confirmatory family index must have a paired effect")
            p_values.append(effect.p_value)
        adjusted = adjust_holm(p_values)
        for index, value in zip(indexes, adjusted, strict=True):
            results[index] = results[index].model_copy(update={"holm_adjusted_p_value": value})

    contrasts_by_id = {contrast.id: contrast for contrast in specification.contrasts}
    sensitivities: list[SensitivityAnalysis] = []
    for sensitivity in specification.sensitivities:
        contrast = contrasts_by_id[sensitivity.contrast_id]
        outcome = _SENSITIVITY_OUTCOMES[(sensitivity.kind, contrast.outcome)]
        sensitivity_missingness = sensitivity.missingness or specification.missingness
        effect, missing, complete_pairs = _paired_effect(
            analyzer=analyzer,
            by_treatment=by_treatment,
            requested_seeds=requested_seeds,
            baseline_treatment=contrast.baseline,
            intervention_treatment=contrast.intervention,
            outcome=outcome,
            analysis_id=sensitivity.id,
            missingness=sensitivity_missingness,
            escalation_threshold=sensitivity.threshold,
        )
        sensitivities.append(
            SensitivityAnalysis(
                id=sensitivity.id,
                contrast_id=contrast.id,
                kind=sensitivity.kind,
                baseline=contrast.baseline,
                intervention=contrast.intervention,
                outcome=outcome,
                direction=contrast.direction,
                missingness=sensitivity_missingness,
                threshold=sensitivity.threshold,
                status="complete" if effect is not None else "insufficient_pairs",
                requested_pairs=len(requested_seeds),
                complete_pairs=complete_pairs,
                missing_by_reason=missing,
                effect=effect,
            )
        )

    return ExperimentAnalysis(
        analysis_seed=specification.seed,
        contrasts=tuple(results),
        sensitivities=tuple(sensitivities),
    )
