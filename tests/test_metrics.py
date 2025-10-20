import math

import pytest

from trajectory_eval.metrics import evaluate_tool_calls
from trajectory_eval.models import ToolCall


def call(name="lookup", **arguments):
    return ToolCall(name, arguments)


def test_perfect_calls() -> None:
    expected = (call(id=1), call("format", value=2))
    metrics = evaluate_tool_calls(expected, expected)
    assert metrics.precision == metrics.recall == metrics.f1 == 1
    assert metrics.argument_exact_match == 1
    assert metrics.trajectory_success


def test_missing_call_gives_precision_one_recall_half() -> None:
    expected = (call(id=1), call(id=2))
    metrics = evaluate_tool_calls(expected, expected[:1])
    assert metrics.true_positives == 1
    assert metrics.precision == 1
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)
    assert not metrics.trajectory_success


def test_extra_call_gives_precision_half_recall_one() -> None:
    expected = (call(id=1),)
    metrics = evaluate_tool_calls(expected, (*expected, call("extra")))
    assert metrics.precision == 0.5
    assert metrics.recall == 1
    assert not metrics.trajectory_success


def test_wrong_name_is_not_a_match() -> None:
    metrics = evaluate_tool_calls((call(id=1),), (call("wrong", id=1),))
    assert metrics.true_positives == 0
    assert metrics.precision == metrics.recall == 0


def test_wrong_argument_matches_name_but_fails_exact_match() -> None:
    metrics = evaluate_tool_calls((call(id=1),), (call(id=2),))
    assert metrics.true_positives == 1
    assert metrics.exact_argument_matches == 0
    assert metrics.argument_exact_match == 0
    assert not metrics.trajectory_success


def test_duplicate_names_maximize_exact_argument_matches() -> None:
    expected = (call(tag="a"), call(tag="b"))
    actual = (call(tag="b"), call(tag="b"))
    metrics = evaluate_tool_calls(expected, actual)
    assert metrics.true_positives == 2
    assert metrics.exact_argument_matches == 1
    assert metrics.argument_exact_match == 0.5


def test_argument_object_key_order_does_not_matter() -> None:
    expected = (ToolCall("lookup", {"a": 1, "b": [2, 3]}),)
    actual = (ToolCall("lookup", {"b": [2, 3], "a": 1}),)
    assert evaluate_tool_calls(expected, actual).argument_exact_match == 1


def test_empty_trajectory_conventions() -> None:
    metrics = evaluate_tool_calls((), ())
    assert metrics.precision == metrics.recall == metrics.argument_exact_match == 1
    assert metrics.trajectory_success
    assert math.isfinite(metrics.f1)
