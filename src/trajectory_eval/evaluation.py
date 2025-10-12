"""Repeated stochastic trajectory evaluation with deterministic replay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .agents import ScriptedAgent
from .metrics import (
    CallMetrics,
    ReliabilitySummary,
    call_metrics_from_counts,
    evaluate_tool_calls,
    pass_k_moments,
    summarize_reliability,
)
from .models import Task, Trajectory


@dataclass(frozen=True)
class TaskReport:
    task_id: str
    successes: int
    trials: int
    pass_k: float
    pass_k_variance: float
    tool_calls: CallMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "successes": self.successes,
            "trials": self.trials,
            "pass_k": self.pass_k,
            "pass_k_variance": self.pass_k_variance,
            "tool_calls": self.tool_calls.__dict__,
        }


@dataclass(frozen=True)
class EvaluationReport:
    agent: str
    seed: int
    repetitions: int
    tool_calls: CallMetrics
    reliability: ReliabilitySummary
    tasks: tuple[TaskReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "tool_calls": self.tool_calls.__dict__,
            "reliability": self.reliability.as_dict(),
            "tasks": [task.as_dict() for task in self.tasks],
        }


def evaluate_trajectories(
    trajectories: Iterable[Trajectory], tasks: Iterable[Task], repetitions: int, k: int
) -> tuple[CallMetrics, ReliabilitySummary, tuple[TaskReport, ...]]:
    task_list = tuple(tasks)
    by_task: dict[str, list[Trajectory]] = {task.task_id: [] for task in task_list}
    for trajectory in trajectories:
        if trajectory.task_id not in by_task:
            raise ValueError(f"trajectory references unknown task {trajectory.task_id!r}")
        by_task[trajectory.task_id].append(trajectory)
    total_matched = total_expected = total_actual = total_exact = 0
    task_reports: list[TaskReport] = []
    outcomes: list[tuple[int, int]] = []
    for task in task_list:
        runs = by_task[task.task_id]
        if len(runs) != repetitions:
            raise ValueError(f"task {task.task_id!r} has {len(runs)} runs, expected {repetitions}")
        successes = 0
        matched = expected = actual = exact = 0
        for run in runs:
            metrics = evaluate_tool_calls(run.expected, run.actual)
            successes += metrics.trajectory_success
            matched += metrics.true_positives
            expected += metrics.expected_calls
            actual += metrics.actual_calls
            exact += metrics.exact_argument_matches
        total_matched += matched
        total_expected += expected
        total_actual += actual
        total_exact += exact
        estimate, variance = pass_k_moments(successes, repetitions, k)
        outcomes.append((successes, repetitions))
        task_reports.append(
            TaskReport(
                task.task_id,
                successes,
                repetitions,
                estimate,
                variance,
                call_metrics_from_counts(matched, expected, actual, exact),
            )
        )
    return (
        call_metrics_from_counts(total_matched, total_expected, total_actual, total_exact),
        summarize_reliability(outcomes, k),
        tuple(task_reports),
    )


def evaluate_agent(
    agent: ScriptedAgent, tasks: Iterable[Task], repetitions: int = 24, k: int = 3, seed: int = 7
) -> EvaluationReport:
    task_list = tuple(tasks)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    trajectories = [
        agent.run(task, repetition, seed)
        for task in task_list
        for repetition in range(repetitions)
    ]
    tool_calls, reliability, task_reports = evaluate_trajectories(
        trajectories, task_list, repetitions, k
    )
    return EvaluationReport(
        agent.name, seed, repetitions, tool_calls, reliability, task_reports
    )
