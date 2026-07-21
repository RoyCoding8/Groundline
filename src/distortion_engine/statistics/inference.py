from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterable, Mapping
from statistics import NormalDist
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class PairedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int
    mean_difference: float
    median_difference: float
    standardized_effect: float | None
    standardized_effect_status: Literal["available", "not_estimable"]
    standardized_effect_reason: Literal["insufficient_pairs", "near_zero_variance"] | None = None
    p_value: float
    p_value_method: Literal["exact_sign_flip", "monte_carlo_sign_flip"]
    randomization_draws: int
    ci_low: float
    ci_high: float
    interval_method: Literal["bca", "percentile"]
    interval_draws: int
    interval_reason: Literal["insufficient_jackknife_variation"] | None = None
    bootstrap_seed: int
    randomization_seed: int


def aggregate_seed_outcomes(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Collapse repeated ticks/agents before inference; seeds are the replicates."""
    required = {"seed", "treatment", outcome}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not np.isfinite(frame[outcome].to_numpy(dtype=float)).all():
        raise ValueError(f"{outcome} values must be finite")
    grouped = frame.groupby(["seed", "treatment"], as_index=False, observed=True)[[outcome]].mean()
    return grouped.sort_values(by=["seed", "treatment"], ignore_index=True)


def adjust_holm(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("p-values must be finite")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[original_index] = running
    return adjusted


def minimum_sign_flip_pairs(*, alpha: float, family_size: int) -> int:
    """Smallest n where a unanimous two-sided sign pattern can survive Holm."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if family_size < 1:
        raise ValueError("family_size must be positive")
    pairs = 1
    while family_size * (2.0 ** (1 - pairs)) > alpha:
        pairs += 1
    return pairs


def factorial_contrasts(frame: pd.DataFrame, outcome: str) -> dict[str, pd.Series]:
    """Calculate 2x2 treatment effects within each paired seed.

    The effects use marginal differences for both knobs and a difference-in-differences
    interaction. Seeds missing any factorial cell are excluded rather than imputed.
    """
    required = {"seed", "incentive_pressure", "attention_budget", outcome}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    numeric_columns = ["seed", "incentive_pressure", "attention_budget", outcome]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("factorial contrast cells must be finite")
    incentive_levels = sorted(frame["incentive_pressure"].unique())
    attention_levels = sorted(frame["attention_budget"].unique())
    if len(incentive_levels) != 2 or len(attention_levels) != 2:
        raise ValueError("factorial contrasts require exactly two levels for each knob")
    low_i, high_i = incentive_levels
    low_a, high_a = attention_levels
    effects: dict[str, dict[int, float]] = {
        "incentive": {},
        "attention": {},
        "interaction": {},
    }
    for seed, group in frame.groupby("seed", observed=True):
        cells = {
            (row.incentive_pressure, row.attention_budget): float(getattr(row, outcome))
            for row in group.itertuples(index=False)
        }
        required_cells = {(low_i, low_a), (low_i, high_a), (high_i, low_a), (high_i, high_a)}
        if not required_cells.issubset(cells):
            continue
        low_low = cells[(low_i, low_a)]
        low_high = cells[(low_i, high_a)]
        high_low = cells[(high_i, low_a)]
        high_high = cells[(high_i, high_a)]
        key = cast(int, seed)
        seed_effects = {
            "incentive": ((high_low + high_high) - (low_low + low_high)) / 2,
            "attention": ((low_high + high_high) - (low_low + high_low)) / 2,
            "interaction": (high_high - high_low) - (low_high - low_low),
        }
        if any(not math.isfinite(value) for value in seed_effects.values()):
            raise ValueError("factorial contrast effects must be finite")
        for name, value in seed_effects.items():
            effects[name][key] = value
    return {
        name: pd.Series(values, name=name, dtype=float).sort_index()
        for name, values in effects.items()
    }


class PairedAnalyzer:
    def __init__(self, *, bootstrap_draws: int = 10_000, analysis_seed: int = 0) -> None:
        if bootstrap_draws < 1:
            raise ValueError("bootstrap_draws must be positive")
        self.bootstrap_draws = bootstrap_draws
        self.analysis_seed = analysis_seed

    def compare(
        self,
        *,
        baseline: Mapping[int, float],
        intervention: Mapping[int, float],
        contrast_id: str = "paired_effect",
    ) -> PairedEffect:
        seeds = sorted(set(baseline).intersection(intervention))
        if not seeds:
            raise ValueError("paired analysis requires shared seeds")
        baseline_values = np.asarray([baseline[seed] for seed in seeds], dtype=float)
        intervention_values = np.asarray([intervention[seed] for seed in seeds], dtype=float)
        if not np.isfinite(baseline_values).all() or not np.isfinite(intervention_values).all():
            raise ValueError("paired analysis inputs must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            differences = intervention_values - baseline_values
        if not np.isfinite(differences).all():
            raise ValueError("paired differences must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            mean_difference = float(differences.mean())
            median_difference = float(np.median(differences))
            difference_scale = float(np.max(np.abs(differences)))
            if len(differences) > 1 and difference_scale:
                standard_deviation = float(
                    (differences / difference_scale).std(ddof=1) * difference_scale
                )
            else:
                standard_deviation = 0.0
        if not all(
            math.isfinite(value)
            for value in (
                mean_difference,
                median_difference,
                difference_scale,
                standard_deviation,
            )
        ):
            raise ValueError("paired derived statistics must be finite")
        near_zero_variance = standard_deviation <= max(
            np.finfo(float).tiny,
            difference_scale * 1e-12,
        )
        standardized_status: Literal["available", "not_estimable"]
        standardized_reason: Literal["insufficient_pairs", "near_zero_variance"] | None
        if len(differences) == 1:
            standardized = None
            standardized_status = "not_estimable"
            standardized_reason = "insufficient_pairs"
        elif near_zero_variance:
            standardized = None
            standardized_status = "not_estimable"
            standardized_reason = "near_zero_variance"
        else:
            standardized = mean_difference / standard_deviation
            standardized_status = "available"
            standardized_reason = None
        bootstrap_seed = self._derive_seed(contrast_id, "bootstrap")
        randomization_seed = self._derive_seed(contrast_id, "randomization")
        p_value, p_value_method, randomization_draws = self._sign_flip_p_value(
            differences,
            randomization_seed,
        )
        low, high, interval_method, interval_reason = self._bootstrap_interval(
            differences,
            bootstrap_seed,
        )
        return PairedEffect(
            n_pairs=len(seeds),
            mean_difference=mean_difference,
            median_difference=median_difference,
            standardized_effect=standardized,
            standardized_effect_status=standardized_status,
            standardized_effect_reason=standardized_reason,
            p_value=p_value,
            p_value_method=p_value_method,
            randomization_draws=randomization_draws,
            ci_low=low,
            ci_high=high,
            interval_method=interval_method,
            interval_draws=self.bootstrap_draws,
            interval_reason=interval_reason,
            bootstrap_seed=bootstrap_seed,
            randomization_seed=randomization_seed,
        )

    def _derive_seed(self, contrast_id: str, method: str) -> int:
        payload = f"{self.analysis_seed}\0{contrast_id}\0{method}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8])

    def _bootstrap_interval(
        self,
        differences: NDArray[np.float64],
        bootstrap_seed: int,
    ) -> tuple[
        float,
        float,
        Literal["bca", "percentile"],
        Literal["insufficient_jackknife_variation"] | None,
    ]:
        rng = np.random.default_rng(bootstrap_seed)
        sample_indices = rng.integers(
            0,
            len(differences),
            size=(self.bootstrap_draws, len(differences)),
        )
        bootstrap_means = differences[sample_indices].mean(axis=1)
        percentile_low, percentile_high = np.quantile(bootstrap_means, [0.025, 0.975])
        if len(differences) < 2:
            return (
                float(percentile_low),
                float(percentile_high),
                "percentile",
                "insufficient_jackknife_variation",
            )

        observed = float(differences.mean())
        less_than_observed = float(np.count_nonzero(bootstrap_means < observed))
        less_than_or_equal = float(np.count_nonzero(bootstrap_means <= observed))
        probability = (less_than_observed + less_than_or_equal) / (2 * self.bootstrap_draws)
        probability = min(
            max(probability, 0.5 / self.bootstrap_draws),
            1 - 0.5 / self.bootstrap_draws,
        )
        normal = NormalDist()
        bias_correction = normal.inv_cdf(probability)
        jackknife = np.asarray(
            [np.delete(differences, index).mean() for index in range(len(differences))]
        )
        centered = jackknife.mean() - jackknife
        jackknife_scale = max(abs(observed), float(np.max(np.abs(jackknife))))
        centered_scale = float(np.max(np.abs(centered)))
        scaled_centered = centered / centered_scale if centered_scale else centered
        jackknife_variation = float(np.sqrt(np.sum(scaled_centered**2)))
        if centered_scale <= max(
            np.finfo(float).tiny,
            jackknife_scale * 1e-12,
        ):
            return (
                float(percentile_low),
                float(percentile_high),
                "percentile",
                "insufficient_jackknife_variation",
            )
        denominator = 6 * jackknife_variation**3
        acceleration = float(np.sum(scaled_centered**3)) / denominator

        adjusted_probabilities: list[float] = []
        for probability_level in (0.025, 0.975):
            quantile = normal.inv_cdf(probability_level)
            denominator_adjustment = 1 - acceleration * (bias_correction + quantile)
            adjusted = normal.cdf(
                bias_correction + (bias_correction + quantile) / denominator_adjustment
            )
            adjusted_probabilities.append(min(max(adjusted, 0.0), 1.0))
        low, high = np.quantile(bootstrap_means, adjusted_probabilities)
        return float(low), float(high), "bca", None

    @staticmethod
    def _sign_flip_p_value(
        differences: NDArray[np.float64],
        randomization_seed: int,
    ) -> tuple[float, Literal["exact_sign_flip", "monte_carlo_sign_flip"], int]:
        scale = float(np.max(np.abs(differences)))
        normalized = differences / scale if scale else differences
        observed = abs(float(normalized.mean()))
        n = len(normalized)
        tolerance = 8 * np.finfo(float).eps
        if n <= 20:
            extreme = 0
            total = 2**n
            for sign_values in itertools.product((-1.0, 1.0), repeat=n):
                statistic = abs(float((normalized * np.asarray(sign_values)).mean()))
                extreme += int(statistic >= observed - tolerance)
            return float(extreme / total), "exact_sign_flip", total
        rng = np.random.default_rng(randomization_seed)
        sign_matrix = rng.choice((-1.0, 1.0), size=(100_000, n))
        statistics = np.abs((sign_matrix * normalized).mean(axis=1))
        return (
            float((np.count_nonzero(statistics >= observed - tolerance) + 1) / 100_001),
            "monte_carlo_sign_flip",
            100_000,
        )
