import math

import pandas as pd
import pytest

from groundline.statistics.inference import (
    PairedAnalyzer,
    adjust_holm,
    aggregate_seed_outcomes,
    factorial_contrasts,
    minimum_sign_flip_pairs,
)


def test_exact_sign_flip_uses_seed_as_the_unit_of_analysis() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=500, analysis_seed=7)

    result = analyzer.compare(
        baseline={1: 0.1, 2: 0.2, 3: 0.3},
        intervention={1: 1.1, 2: 1.2, 3: 1.3},
    )

    assert result.n_pairs == 3
    assert result.mean_difference == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.25)
    assert result.p_value_method == "exact_sign_flip"
    assert result.ci_low <= result.mean_difference <= result.ci_high


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_paired_analysis_rejects_non_finite_inputs(invalid: float) -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)

    with pytest.raises(ValueError, match="finite"):
        analyzer.compare(
            baseline={1: 0.0, 2: 0.0},
            intervention={1: 1.0, 2: invalid},
        )


def test_paired_analysis_rejects_non_finite_subtraction_overflow() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)

    with pytest.raises(ValueError, match="differences must be finite"):
        analyzer.compare(
            baseline={1: -1e308, 2: 0.0},
            intervention={1: 1e308, 2: 1.0},
        )


def test_paired_analysis_rejects_non_finite_derived_statistics() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)

    with pytest.raises(ValueError, match="derived statistics must be finite"):
        analyzer.compare(
            baseline={1: 0.0, 2: 0.0},
            intervention={1: 1e308, 2: 1e308},
        )


def test_bca_acceleration_is_stable_for_large_finite_effects() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=1_000, analysis_seed=7)

    result = analyzer.compare(
        baseline={1: 0.0, 2: 0.0},
        intervention={1: 1e154, 2: -1e154},
    )

    assert result.interval_method == "bca"
    assert all(math.isfinite(value) for value in (result.ci_low, result.ci_high))


def test_exact_sign_flip_is_invariant_to_positive_rescaling() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)
    baseline = {seed: 0.0 for seed in range(7)}

    unit_scale = analyzer.compare(
        baseline=baseline,
        intervention={seed: 1.0 for seed in baseline},
    )
    tiny_scale = analyzer.compare(
        baseline=baseline,
        intervention={seed: 1e-14 for seed in baseline},
    )

    assert unit_scale.p_value == pytest.approx(0.015625)
    assert tiny_scale.p_value == unit_scale.p_value


def test_near_constant_paired_effect_has_typed_non_estimable_standardization() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)

    result = analyzer.compare(
        baseline={1: 0.1, 2: 0.2, 3: 0.3},
        intervention={1: 0.2, 2: 0.3, 3: 0.4},
    )

    assert result.standardized_effect is None
    assert result.standardized_effect_status == "not_estimable"
    assert result.standardized_effect_reason == "near_zero_variance"
    assert result.interval_method == "percentile"
    assert result.interval_reason == "insufficient_jackknife_variation"


def test_paired_analysis_reports_median_and_deterministic_bca_interval() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=20_000, analysis_seed=17)
    differences = [0.0, 0.0, 0.0, 0.0, 1.0, 8.0]
    baseline = dict.fromkeys(range(len(differences)), 0.0)
    intervention = dict(enumerate(differences))

    first = analyzer.compare(baseline=baseline, intervention=intervention)
    second = analyzer.compare(baseline=baseline, intervention=intervention)

    assert first.median_difference == 0.0
    assert first.interval_method == "bca"
    assert first.interval_draws == 20_000
    assert first.ci_low == pytest.approx(0.0)
    assert first.ci_high == pytest.approx(5.5, abs=0.25)
    assert second == first


def test_analysis_seed_derives_independent_method_seeds_per_contrast() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=91)
    baseline = {seed: 0.0 for seed in range(4)}
    intervention = {0: 0.0, 1: 1.0, 2: 2.0, 3: 8.0}

    first = analyzer.compare(
        baseline=baseline,
        intervention=intervention,
        contrast_id="attention",
    )
    repeated = analyzer.compare(
        baseline=baseline,
        intervention=intervention,
        contrast_id="attention",
    )
    second = analyzer.compare(
        baseline=baseline,
        intervention=intervention,
        contrast_id="incentive",
    )

    assert repeated == first
    assert first.bootstrap_seed != first.randomization_seed
    assert second.bootstrap_seed != first.bootstrap_seed
    assert second.randomization_seed != first.randomization_seed


def test_large_design_labels_seeded_monte_carlo_inference_honestly() -> None:
    analyzer = PairedAnalyzer(bootstrap_draws=100, analysis_seed=7)
    baseline = {seed: 0.0 for seed in range(21)}
    intervention = {seed: 1.0 for seed in range(21)}

    result = analyzer.compare(baseline=baseline, intervention=intervention)

    assert result.p_value_method == "monte_carlo_sign_flip"
    assert result.randomization_draws == 100_000
    assert result.p_value >= 1 / 100_001


def test_seed_aggregation_rejects_non_finite_outcomes() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 2],
            "treatment": ["control", "control"],
            "optimism_bias": [0.1, float("nan")],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        aggregate_seed_outcomes(rows, "optimism_bias")


def test_tick_duplication_does_not_create_fake_replicates() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1, 1, 2, 2, 2],
            "treatment": ["control"] * 6,
            "optimism_bias": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        }
    )

    aggregated = aggregate_seed_outcomes(rows, "optimism_bias")

    assert len(aggregated) == 2
    assert aggregated.loc[aggregated.seed == 1, "optimism_bias"].item() == pytest.approx(0.1)


def test_holm_adjustment_controls_a_family() -> None:
    adjusted = adjust_holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjustment_rejects_non_finite_p_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            adjust_holm([0.01, value, 0.03])


def test_minimum_pairs_accounts_for_two_sided_exact_test_and_holm_family() -> None:
    assert minimum_sign_flip_pairs(alpha=0.05, family_size=3) == 7


def test_factorial_contrasts_reject_non_finite_cells() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1, 1, 1],
            "incentive_pressure": [0.0, 0.0, 1.0, 1.0],
            "attention_budget": [0, 1, 0, 1],
            "executive_optimism_bias": [0.1, 0.2, 0.3, float("inf")],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        factorial_contrasts(rows, "executive_optimism_bias")


def test_factorial_contrasts_reject_non_finite_derived_effects() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1, 1, 1],
            "incentive_pressure": [0.0, 0.0, 1.0, 1.0],
            "attention_budget": [0, 1, 0, 1],
            "executive_optimism_bias": [-1e308, -1e308, 1e308, 1e308],
        }
    )

    with pytest.raises(ValueError, match="effects must be finite"):
        factorial_contrasts(rows, "executive_optimism_bias")


def test_factorial_contrasts_are_computed_within_seed() -> None:
    rows = []
    for seed, offset in ((1, 0.0), (2, 0.1)):
        for incentive, attention, outcome in (
            (0.0, 0, 0.10 + offset),
            (0.0, 2, 0.05 + offset),
            (1.0, 0, 0.40 + offset),
            (1.0, 2, 0.25 + offset),
        ):
            rows.append(
                {
                    "seed": seed,
                    "incentive_pressure": incentive,
                    "attention_budget": attention,
                    "executive_optimism_bias": outcome,
                }
            )

    contrasts = factorial_contrasts(pd.DataFrame(rows), "executive_optimism_bias")

    assert contrasts["incentive"].tolist() == pytest.approx([0.25, 0.25])
    assert contrasts["attention"].tolist() == pytest.approx([-0.10, -0.10])
    assert contrasts["interaction"].tolist() == pytest.approx([-0.10, -0.10])
