import pytest
from pydantic import ValidationError

from distortion_engine.experiments.analysis import (
    AnalysisRun,
    AnalysisSpecification,
    ContrastSpecification,
    MissingnessPolicy,
    SensitivitySpecification,
    analyze_experiment,
)
from distortion_engine.metrics.outcomes import (
    CalibrationOutcome,
    DepthOutcomes,
    DistortionSummary,
    EscalationDelay,
    EscalationThresholdOutcome,
    NormalizedOperationalHarm,
    OperationalHarm,
    OperationalHarmComponents,
    RunOutcomes,
    ScalarOutcome,
)
from distortion_engine.world.models import OperationalHarmMaxima


def _summary(optimism: float, *, median: float | None = None) -> DistortionSummary:
    return DistortionSummary(
        report_count=1,
        total_business_value=1,
        optimism_bias_mean=optimism,
        optimism_bias_median=optimism if median is None else median,
        absolute_error_mean=abs(optimism),
        vector_loss_mean=abs(optimism),
        equal_weight_vector_loss_mean=abs(optimism),
        progress_error_mean=abs(optimism),
        quality_error_mean=abs(optimism),
        schedule_error_mean=abs(optimism),
        reliability_error_mean=abs(optimism),
        edge_transformation_mean=None,
    )


def _run(seed: int, treatment: str, *, executive_bias: float, amplification: float) -> AnalysisRun:
    outcomes = RunOutcomes(
        depth=(
            DepthOutcomes(
                depth=0,
                all_ticks=_summary(executive_bias),
                adverse_ticks=_summary(executive_bias),
            ),
        ),
        pre_release_depth=(
            DepthOutcomes(
                depth=0,
                all_ticks=_summary(executive_bias),
                adverse_ticks=_summary(executive_bias),
            ),
        ),
        upward_amplification=ScalarOutcome(status="available", value=amplification),
        pre_release_upward_amplification=ScalarOutcome(status="available", value=amplification),
        edge_transformation=ScalarOutcome(
            status="unavailable",
            value=None,
            reason="no_edge_reports",
        ),
        pre_release_edge_transformation=ScalarOutcome(
            status="unavailable",
            value=None,
            reason="no_edge_reports",
        ),
        escalation_delays=(),
        escalation_sensitivities=(),
        calibration=CalibrationOutcome(report_count=1, brier_score=0.25),
        operational_harm=OperationalHarm(
            raw=OperationalHarmComponents(
                release_delay=0,
                escaped_defects=0,
                incident=0,
                remediation=0,
                scope_loss=0,
                verification_cost=0,
                staffing_cost=0,
            ),
            normalized=NormalizedOperationalHarm(
                release_delay=0,
                escaped_defects=0,
                incident=0,
                remediation=0,
                scope_loss=0,
            ),
            index=0,
            maxima=OperationalHarmMaxima(),
            release_tick=1,
            release_censored=False,
        ),
    )
    return AnalysisRun(
        seed=seed,
        treatment=treatment,
        outcomes=outcomes,
        oracle_regret=0,
    )


def _run_with_summaries(
    seed: int,
    treatment: str,
    *,
    adverse_mean: float,
    adverse_median: float,
    all_mean: float,
    all_median: float,
) -> AnalysisRun:
    run = _run(seed, treatment, executive_bias=adverse_mean, amplification=0)
    depth = DepthOutcomes(
        depth=0,
        all_ticks=_summary(all_mean, median=all_median),
        adverse_ticks=_summary(adverse_mean, median=adverse_median),
    )
    return run.model_copy(update={"outcomes": run.outcomes.model_copy(update={"depth": (depth,)})})


def _run_with_vector_outcomes(
    seed: int,
    treatment: str,
    *,
    vector_loss: float,
    equal_weight: float,
    progress: float,
    quality: float,
    schedule: float,
    reliability: float,
    pre_release_vector_loss: float,
) -> AnalysisRun:
    run = _run(seed, treatment, executive_bias=0, amplification=0)
    summary = _summary(0).model_copy(
        update={
            "vector_loss_mean": vector_loss,
            "equal_weight_vector_loss_mean": equal_weight,
            "progress_error_mean": progress,
            "quality_error_mean": quality,
            "schedule_error_mean": schedule,
            "reliability_error_mean": reliability,
        }
    )
    pre_release_summary = summary.model_copy(update={"vector_loss_mean": pre_release_vector_loss})
    return run.model_copy(
        update={
            "outcomes": run.outcomes.model_copy(
                update={
                    "depth": (DepthOutcomes(depth=0, all_ticks=summary, adverse_ticks=summary),),
                    "pre_release_depth": (
                        DepthOutcomes(
                            depth=0,
                            all_ticks=pre_release_summary,
                            adverse_ticks=pre_release_summary,
                        ),
                    ),
                }
            )
        }
    )


def _specification(missingness: MissingnessPolicy = "complete_case") -> AnalysisSpecification:
    return AnalysisSpecification(
        seed=17,
        missingness=missingness,
        contrasts=(
            ContrastSpecification(
                id="h1-pressure-amplification",
                baseline="control",
                intervention="pressure",
                outcome="upward_amplification",
                direction="increase",
                family="co-primary",
                status="confirmatory",
            ),
        ),
    )


def test_declared_upward_amplification_contrast_ignores_mapping_and_run_order() -> None:
    specification = _specification()
    runs = (
        _run(2, "pressure", executive_bias=0.5, amplification=1.2),
        _run(1, "control", executive_bias=0.5, amplification=0.0),
        _run(1, "pressure", executive_bias=0.5, amplification=1.0),
        _run(2, "control", executive_bias=0.5, amplification=0.0),
    )

    analysis = analyze_experiment(specification, runs, requested_seeds=(1, 2))
    result = analysis.contrasts[0]

    assert result.id == "h1-pressure-amplification"
    assert result.baseline == "control"
    assert result.intervention == "pressure"
    assert result.outcome == "upward_amplification"
    assert result.effect is not None
    assert result.effect.mean_difference == pytest.approx(1.1)


def test_preregistered_sensitivities_emit_named_alternative_estimands() -> None:
    specification = AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id="h2-bias",
                baseline="control",
                intervention="pressure",
                outcome="executive_optimism_bias_adverse_mean",
                direction="increase",
                family="co-primary",
                status="confirmatory",
            ),
        ),
        sensitivities=(
            SensitivitySpecification(
                id="h2-all-ticks",
                contrast_id="h2-bias",
                kind="adverse_vs_all_ticks",
            ),
            SensitivitySpecification(
                id="h2-median",
                contrast_id="h2-bias",
                kind="mean_vs_median",
            ),
        ),
    )
    runs = (
        _run_with_summaries(
            1,
            "control",
            adverse_mean=0.1,
            adverse_median=0.05,
            all_mean=0.2,
            all_median=0.1,
        ),
        _run_with_summaries(
            1,
            "pressure",
            adverse_mean=0.4,
            adverse_median=0.2,
            all_mean=0.8,
            all_median=0.4,
        ),
    )

    analysis = analyze_experiment(specification, runs, requested_seeds=(1,))
    sensitivities = {result.id: result for result in analysis.sensitivities}

    assert sensitivities["h2-all-ticks"].outcome == "executive_optimism_bias_all_mean"
    assert sensitivities["h2-all-ticks"].effect is not None
    assert sensitivities["h2-all-ticks"].effect.mean_difference == pytest.approx(0.6)
    assert sensitivities["h2-median"].outcome == "executive_optimism_bias_adverse_median"
    assert sensitivities["h2-median"].effect is not None
    assert sensitivities["h2-median"].effect.mean_difference == pytest.approx(0.15)


def test_post_release_exclusion_uses_pre_release_scalar_estimands() -> None:
    specification = AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id="h1-amplification",
                baseline="control",
                intervention="pressure",
                outcome="upward_amplification",
                direction="increase",
                family="primary",
                status="confirmatory",
            ),
            ContrastSpecification(
                id="edge-transformation",
                baseline="control",
                intervention="pressure",
                outcome="edge_transformation",
                direction="increase",
                family="secondary",
                status="exploratory",
            ),
        ),
        sensitivities=(
            SensitivitySpecification(
                id="h1-pre-release",
                contrast_id="h1-amplification",
                kind="exclude_post_release_ticks",
            ),
            SensitivitySpecification(
                id="edge-pre-release",
                contrast_id="edge-transformation",
                kind="exclude_post_release_ticks",
            ),
        ),
    )

    def with_scalar_outcomes(
        treatment: str,
        *,
        amplification: float,
        pre_release_amplification: float,
        edge: float,
        pre_release_edge: float,
    ) -> AnalysisRun:
        run = _run(1, treatment, executive_bias=0, amplification=amplification)
        return run.model_copy(
            update={
                "outcomes": run.outcomes.model_copy(
                    update={
                        "pre_release_upward_amplification": ScalarOutcome(
                            status="available",
                            value=pre_release_amplification,
                        ),
                        "edge_transformation": ScalarOutcome(
                            status="available",
                            value=edge,
                        ),
                        "pre_release_edge_transformation": ScalarOutcome(
                            status="available",
                            value=pre_release_edge,
                        ),
                    }
                )
            }
        )

    analysis = analyze_experiment(
        specification,
        (
            with_scalar_outcomes(
                "control",
                amplification=0.4,
                pre_release_amplification=0.1,
                edge=0.3,
                pre_release_edge=0.1,
            ),
            with_scalar_outcomes(
                "pressure",
                amplification=1.2,
                pre_release_amplification=0.4,
                edge=0.9,
                pre_release_edge=0.3,
            ),
        ),
        requested_seeds=(1,),
    )
    sensitivities = {result.id: result for result in analysis.sensitivities}

    amplification = sensitivities["h1-pre-release"]
    assert amplification.outcome == "pre_release_upward_amplification"
    assert amplification.effect is not None
    assert amplification.effect.mean_difference == pytest.approx(0.3)
    edge = sensitivities["edge-pre-release"]
    assert edge.outcome == "pre_release_edge_transformation"
    assert edge.effect is not None
    assert edge.effect.mean_difference == pytest.approx(0.2)


def test_preregistered_vector_loss_sensitivities_use_named_alternative_estimands() -> None:
    specification = AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id="vector-loss",
                baseline="control",
                intervention="pressure",
                outcome="executive_vector_loss_adverse_mean",
                direction="increase",
                family="secondary",
                status="confirmatory",
            ),
        ),
        sensitivities=(
            SensitivitySpecification(
                id="vector-equal-weights",
                contrast_id="vector-loss",
                kind="equal_health_dimension_weights",
            ),
            SensitivitySpecification(
                id="vector-progress",
                contrast_id="vector-loss",
                kind="progress_dimension_only",
            ),
            SensitivitySpecification(
                id="vector-quality",
                contrast_id="vector-loss",
                kind="quality_dimension_only",
            ),
            SensitivitySpecification(
                id="vector-schedule",
                contrast_id="vector-loss",
                kind="schedule_dimension_only",
            ),
            SensitivitySpecification(
                id="vector-reliability",
                contrast_id="vector-loss",
                kind="reliability_dimension_only",
            ),
            SensitivitySpecification(
                id="vector-pre-release",
                contrast_id="vector-loss",
                kind="exclude_post_release_ticks",
            ),
        ),
    )
    runs = (
        _run_with_vector_outcomes(
            1,
            "control",
            vector_loss=0.2,
            equal_weight=0.3,
            progress=0.1,
            quality=0.2,
            schedule=0.3,
            reliability=0.6,
            pre_release_vector_loss=0.15,
        ),
        _run_with_vector_outcomes(
            1,
            "pressure",
            vector_loss=0.5,
            equal_weight=0.7,
            progress=0.3,
            quality=0.4,
            schedule=0.8,
            reliability=1.0,
            pre_release_vector_loss=0.35,
        ),
    )

    analysis = analyze_experiment(specification, runs, requested_seeds=(1,))
    sensitivities = {result.id: result for result in analysis.sensitivities}

    expected = {
        "vector-equal-weights": 0.4,
        "vector-progress": 0.2,
        "vector-quality": 0.2,
        "vector-schedule": 0.5,
        "vector-reliability": 0.4,
        "vector-pre-release": 0.2,
    }
    for sensitivity_id, difference in expected.items():
        effect = sensitivities[sensitivity_id].effect
        assert effect is not None
        assert effect.mean_difference == pytest.approx(difference)


def test_escalation_threshold_and_invalid_response_sensitivities_are_explicit() -> None:
    specification = AnalysisSpecification(
        seed=17,
        missingness="complete_case",
        contrasts=(
            ContrastSpecification(
                id="escalation-delay",
                baseline="control",
                intervention="pressure",
                outcome="escalation_delay_mean",
                direction="decrease",
                family="secondary",
                status="confirmatory",
            ),
        ),
        sensitivities=(
            SensitivitySpecification(
                id="escalation-threshold-2",
                contrast_id="escalation-delay",
                kind="alternative_escalation_threshold",
                threshold=2,
            ),
            SensitivitySpecification(
                id="valid-responses-only",
                contrast_id="escalation-delay",
                kind="exclude_invalid_or_refused",
                missingness="complete_case",
            ),
        ),
    )

    def with_delays(treatment: str, primary: int, alternative: int) -> AnalysisRun:
        run = _run(1, treatment, executive_bias=0, amplification=0)
        primary_delay = EscalationDelay(
            agent_id="worker",
            depth=1,
            evidence_tick=1,
            escalation_tick=1 + primary,
            delay_ticks=primary,
            censored=False,
        )
        alternative_delay = primary_delay.model_copy(
            update={"escalation_tick": 1 + alternative, "delay_ticks": alternative}
        )
        return run.model_copy(
            update={
                "outcomes": run.outcomes.model_copy(
                    update={
                        "escalation_delays": (primary_delay,),
                        "escalation_sensitivities": (
                            EscalationThresholdOutcome(
                                threshold=2,
                                delays=(alternative_delay,),
                            ),
                        ),
                    }
                )
            }
        )

    analysis = analyze_experiment(
        specification,
        (with_delays("control", 4, 3), with_delays("pressure", 2, 1)),
        requested_seeds=(1, 2),
    )
    sensitivities = {result.id: result for result in analysis.sensitivities}

    threshold = sensitivities["escalation-threshold-2"]
    assert threshold.threshold == 2
    assert threshold.effect is not None
    assert threshold.effect.mean_difference == pytest.approx(-2)
    valid_only = sensitivities["valid-responses-only"]
    assert valid_only.missingness == "complete_case"
    assert valid_only.complete_pairs == 1
    assert valid_only.missing_by_reason == {"missing_both_runs": 1}


def test_complete_case_counts_requested_seed_missing_from_both_treatments() -> None:
    runs = (
        _run(1, "control", executive_bias=0, amplification=0),
        _run(1, "pressure", executive_bias=0, amplification=1),
    )

    analysis = analyze_experiment(_specification(), runs, requested_seeds=(1, 2))
    result = analysis.contrasts[0]

    assert result.requested_pairs == 2
    assert result.complete_pairs == 1
    assert result.missing_by_reason == {"missing_both_runs": 1}


def test_fail_if_missing_rejects_requested_seed_missing_from_both_treatments() -> None:
    runs = (
        _run(1, "control", executive_bias=0, amplification=0),
        _run(1, "pressure", executive_bias=0, amplification=1),
    )

    with pytest.raises(
        ValueError,
        match="analysis h1-pressure-amplification has missing pairs: missing_both_runs=1",
    ):
        analyze_experiment(
            _specification("fail_if_missing"),
            runs,
            requested_seeds=(1, 2),
        )


def test_analysis_rejects_duplicate_treatment_seed_runs() -> None:
    duplicate = _run(1, "control", executive_bias=0, amplification=0)

    with pytest.raises(
        ValueError,
        match="duplicate analysis run for treatment 'control' and seed 1",
    ):
        analyze_experiment(
            _specification(),
            (duplicate, duplicate),
            requested_seeds=(1,),
        )


def test_analysis_specification_rejects_unknown_declarations() -> None:
    with pytest.raises(ValidationError):
        ContrastSpecification(
            id="unknown",
            baseline="control",
            intervention="pressure",
            outcome="not_a_registered_outcome",  # type: ignore[arg-type]
            direction="increase",
            family="co-primary",
            status="confirmatory",
        )

    base = _specification()
    with pytest.raises(ValidationError, match="references unknown contrast"):
        AnalysisSpecification(
            seed=base.seed,
            missingness=base.missingness,
            contrasts=base.contrasts,
            sensitivities=(
                SensitivitySpecification(
                    id="unknown-contrast",
                    contrast_id="missing",
                    kind="mean_vs_median",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="requires a threshold"):
        SensitivitySpecification(
            id="missing-threshold",
            contrast_id="h1-pressure-amplification",
            kind="alternative_escalation_threshold",
        )

    with pytest.raises(ValidationError, match="requires complete_case missingness"):
        SensitivitySpecification(
            id="invalid-response-rule",
            contrast_id="h1-pressure-amplification",
            kind="exclude_invalid_or_refused",
        )

    with pytest.raises(ValidationError, match="is not defined for outcome upward_amplification"):
        AnalysisSpecification(
            seed=base.seed,
            missingness=base.missingness,
            contrasts=base.contrasts,
            sensitivities=(
                SensitivitySpecification(
                    id="unsupported",
                    contrast_id="h1-pressure-amplification",
                    kind="adverse_vs_all_ticks",
                ),
            ),
        )
