"""One-to-one tool-call metrics and unbiased pass^k reliability statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .models import ToolCall, canonical_arguments


@dataclass(frozen=True)
class CallMetrics:
    true_positives: int
    expected_calls: int
    actual_calls: int
    exact_argument_matches: int
    precision: float
    recall: float
    f1: float
    argument_exact_match: float
    trajectory_success: bool


def match_tool_calls(
    expected: tuple[ToolCall, ...], actual: tuple[ToolCall, ...]
) -> tuple[int, int]:
    expected_by_name: dict[str, list[ToolCall]] = defaultdict(list)
    actual_by_name: dict[str, list[ToolCall]] = defaultdict(list)
    for call in expected:
        expected_by_name[call.name].append(call)
    for call in actual:
        actual_by_name[call.name].append(call)
    matched = 0
    exact = 0
    for name, expected_group in expected_by_name.items():
        actual_group = actual_by_name.get(name, [])
        matched += min(len(expected_group), len(actual_group))
        actual_exact: dict[str, list[ToolCall]] = defaultdict(list)
        for call in actual_group:
            actual_exact[canonical_arguments(call.arguments)].append(call)
        for call in expected_group:
            key = canonical_arguments(call.arguments)
            if actual_exact[key]:
                actual_exact[key].pop()
                exact += 1
    return matched, exact


def _ratio(numerator: int, denominator: int, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def call_metrics_from_counts(
    matched: int, expected: int, actual: int, exact_arguments: int
) -> CallMetrics:
    if min(matched, expected, actual, exact_arguments) < 0:
        raise ValueError("metric counts must be non-negative")
    if matched > min(expected, actual) or exact_arguments > matched:
        raise ValueError("inconsistent matching counts")
    precision = _ratio(matched, actual, float(expected == 0))
    recall = _ratio(matched, expected, float(actual == 0))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    argument_exact = _ratio(exact_arguments, matched, float(expected == actual == 0))
    return CallMetrics(
        matched,
        expected,
        actual,
        exact_arguments,
        precision,
        recall,
        f1,
        argument_exact,
        matched == expected == actual == exact_arguments,
    )


def evaluate_tool_calls(
    expected: tuple[ToolCall, ...], actual: tuple[ToolCall, ...]
) -> CallMetrics:
    matched, exact = match_tool_calls(expected, actual)
    return call_metrics_from_counts(matched, len(expected), len(actual), exact)


def pass_k(successes: int, trials: int, k: int) -> float:
    """Unbiased all-k success estimator: falling(c,k) / falling(n,k)."""
    _validate_reliability_inputs(successes, trials, k)
    if successes < k:
        return 0.0
    return math.prod((successes - offset) / (trials - offset) for offset in range(k))


def pass_k_moments(successes: int, trials: int, k: int) -> tuple[float, float]:
    """Return estimate and exact plug-in binomial variance of the pass^k estimator."""
    estimate = pass_k(successes, trials, k)
    probability = successes / trials
    second_moment = 0.0
    for observed_successes in range(trials + 1):
        mass = math.comb(trials, observed_successes) * probability**observed_successes
        if observed_successes < trials:
            mass *= (1 - probability) ** (trials - observed_successes)
        estimate_at_observation = pass_k(observed_successes, trials, k)
        second_moment += mass * estimate_at_observation**2
    expected_under_binomial = probability**k
    return estimate, max(0.0, second_moment - expected_under_binomial**2)


@dataclass(frozen=True)
class ReliabilitySummary:
    k: int
    tasks: int
    mean_pass_k: float
    variance_of_mean: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    between_task_variance: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "tasks": self.tasks,
            "mean_pass_k": self.mean_pass_k,
            "variance_of_mean": self.variance_of_mean,
            "standard_error": self.standard_error,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "between_task_variance": self.between_task_variance,
        }


def summarize_reliability(outcomes: list[tuple[int, int]], k: int) -> ReliabilitySummary:
    if not outcomes:
        raise ValueError("at least one task outcome is required")
    moments = [pass_k_moments(successes, trials, k) for successes, trials in outcomes]
    estimates = [estimate for estimate, _ in moments]
    mean = sum(estimates) / len(estimates)
    variance_of_mean = sum(variance for _, variance in moments) / len(moments) ** 2
    standard_error = math.sqrt(variance_of_mean)
    between_task_variance = (
        sum((estimate - mean) ** 2 for estimate in estimates) / (len(estimates) - 1)
        if len(estimates) > 1
        else 0.0
    )
    return ReliabilitySummary(
        k=k,
        tasks=len(outcomes),
        mean_pass_k=mean,
        variance_of_mean=variance_of_mean,
        standard_error=standard_error,
        ci95_low=max(0.0, mean - 1.96 * standard_error),
        ci95_high=min(1.0, mean + 1.96 * standard_error),
        between_task_variance=between_task_variance,
    )


def _validate_reliability_inputs(successes: int, trials: int, k: int) -> None:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 1 <= k <= trials:
        raise ValueError("require 1 <= k <= trials")
