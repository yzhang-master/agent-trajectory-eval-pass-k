import math

import pytest

from trajectory_eval.metrics import pass_k, pass_k_moments, summarize_reliability


def brute_pass_k_moments(successes: int, trials: int, k: int) -> tuple[float, float]:
    probability = successes / trials
    estimates = []
    masses = []
    for observed in range(trials + 1):
        mass = math.comb(trials, observed)
        mass *= probability**observed * (1 - probability) ** (trials - observed)
        if observed < k:
            estimate = 0.0
        else:
            estimate = math.prod(
                (observed - offset) / (trials - offset) for offset in range(k)
            )
        masses.append(mass)
        estimates.append(estimate)
    mean = sum(mass * estimate for mass, estimate in zip(masses, estimates, strict=True))
    second = sum(mass * estimate**2 for mass, estimate in zip(masses, estimates, strict=True))
    return mean, second - mean**2


def test_pass_k_is_probability_all_k_runs_pass() -> None:
    assert pass_k(2, 4, 1) == 0.5
    assert pass_k(2, 4, 2) == pytest.approx(1 / 6)
    assert pass_k(2, 4, 3) == 0
    assert pass_k(4, 4, 4) == 1


def test_pass_k_is_not_at_least_one_pass_at_k() -> None:
    pass_at_k = 1 - math.comb(4 - 2, 2) / math.comb(4, 2)
    assert pass_at_k == pytest.approx(5 / 6)
    assert pass_k(2, 4, 2) == pytest.approx(1 / 6)


@pytest.mark.parametrize(("successes", "trials", "k"), [(1, 2, 1), (2, 4, 2), (3, 7, 3), (5, 5, 2)])
def test_exact_binomial_variance_matches_independent_reference(successes, trials, k) -> None:
    estimate, variance = pass_k_moments(successes, trials, k)
    expected_under_binomial, expected_variance = brute_pass_k_moments(successes, trials, k)
    assert expected_under_binomial == pytest.approx((successes / trials) ** k)
    assert estimate == pytest.approx(pass_k(successes, trials, k))
    assert variance == pytest.approx(expected_variance)


def test_known_binomial_variance_for_pass_at_one() -> None:
    estimate, variance = pass_k_moments(1, 2, 1)
    assert estimate == 0.5
    assert variance == 0.125


def test_aggregate_mean_variance_and_between_task_variance() -> None:
    outcomes = [(2, 4), (4, 4)]
    summary = summarize_reliability(outcomes, 2)
    variances = [pass_k_moments(*outcome, 2)[1] for outcome in outcomes]
    assert summary.mean_pass_k == pytest.approx(7 / 12)
    assert summary.variance_of_mean == pytest.approx(sum(variances) / 4)
    assert summary.standard_error == pytest.approx(math.sqrt(sum(variances) / 4))
    assert summary.between_task_variance == pytest.approx((1 / 6 - 7 / 12) ** 2 * 2)
    assert summary.ci95_low <= summary.mean_pass_k <= summary.ci95_high


def test_deterministic_success_has_zero_model_variance() -> None:
    summary = summarize_reliability([(8, 8), (8, 8)], 3)
    assert summary.mean_pass_k == 1
    assert summary.standard_error == 0
    assert summary.ci95_low == summary.ci95_high == 1


@pytest.mark.parametrize("outcome", [(-1, 4), (5, 4), (1, 0)])
def test_invalid_outcomes_fail(outcome) -> None:
    with pytest.raises(ValueError):
        pass_k(*outcome, 1)


@pytest.mark.parametrize("k", [0, 3])
def test_invalid_k_fails(k) -> None:
    with pytest.raises(ValueError):
        pass_k(1, 2, k)
