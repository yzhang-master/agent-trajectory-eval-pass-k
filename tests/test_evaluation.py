import pytest

from trajectory_eval.agents import ScriptedAgent, default_tasks
from trajectory_eval.evaluation import evaluate_agent, evaluate_trajectories
from trajectory_eval.models import Task, ToolCall


def test_scripted_agent_replays_exactly() -> None:
    first = evaluate_agent(ScriptedAgent(), default_tasks(), repetitions=24, k=3, seed=17)
    second = evaluate_agent(ScriptedAgent(), default_tasks(), repetitions=24, k=3, seed=17)
    different = evaluate_agent(ScriptedAgent(), default_tasks(), repetitions=24, k=3, seed=18)
    assert first.as_dict() == second.as_dict()
    assert [task.successes for task in first.tasks] != [
        task.successes for task in different.tasks
    ]


def test_every_controlled_failure_mode_is_exercised() -> None:
    task = Task("task", (ToolCall("lookup", {"id": 1}),), 0)
    outcomes = [ScriptedAgent().run(task, repetition, 7).actual for repetition in range(40)]
    assert () in outcomes
    assert any(len(calls) == 2 for calls in outcomes)
    assert any(calls[0].arguments.get("__wrong__") for calls in outcomes)
    assert any(calls and calls[0].name == "lookup_wrong" for calls in outcomes)


def test_perfect_agent_has_complete_calls_and_reliability() -> None:
    tasks = [Task("one", (ToolCall("lookup", {"id": 1}),), 1)]
    report = evaluate_agent(ScriptedAgent(), tasks, repetitions=12, k=4)
    assert report.tool_calls.precision == report.tool_calls.recall == 1
    assert report.tool_calls.argument_exact_match == 1
    assert report.reliability.mean_pass_k == 1
    assert report.reliability.standard_error == 0


def test_always_failing_agent_has_zero_reliability() -> None:
    tasks = [Task("one", (ToolCall("lookup", {"id": 1}),), 0)]
    report = evaluate_agent(ScriptedAgent(), tasks, repetitions=12, k=4)
    assert report.reliability.mean_pass_k == 0
    assert report.tool_calls.true_positives < report.tool_calls.expected_calls


def test_trajectory_count_is_validated() -> None:
    task = Task("one", (ToolCall("lookup", {}),), 1)
    trajectory = ScriptedAgent().run(task, 0, 7)
    with pytest.raises(ValueError, match="expected 2"):
        evaluate_trajectories([trajectory], [task], repetitions=2, k=1)


def test_repetition_count_is_validated() -> None:
    with pytest.raises(ValueError):
        evaluate_agent(ScriptedAgent(), default_tasks(), repetitions=0)
