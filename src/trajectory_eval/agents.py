"""Reproducible stochastic agents with controlled tool-call failures."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from .models import Task, ToolCall, Trajectory


class ScriptedAgent:
    def __init__(self, name: str = "scripted"):
        self.name = name

    def run(self, task: Task, repetition: int, seed: int) -> Trajectory:
        success_draw, failure_draw = self._draws(task.task_id, repetition, seed)
        if success_draw < task.success_probability:
            actual = task.expected
        else:
            actual = self._failed_calls(task.expected, int(failure_draw * 4))
        return Trajectory(task.task_id, repetition, task.expected, actual)

    def _draws(self, task_id: str, repetition: int, seed: int) -> tuple[float, float]:
        digest = hashlib.sha256(
            f"{self.name}:{seed}:{task_id}:{repetition}".encode()
        ).digest()
        success = int.from_bytes(digest[:16], "big") / 2**128
        failure = int.from_bytes(digest[16:], "big") / 2**128
        return success, failure

    @staticmethod
    def _failed_calls(expected: tuple[ToolCall, ...], mode: int) -> tuple[ToolCall, ...]:
        if not expected:
            return (ToolCall("unexpected", {}),)
        if mode == 0:
            return expected[:-1]
        if mode == 1:
            return (*expected, ToolCall("unexpected", {"reason": "extra"}))
        first = expected[0]
        if mode == 2:
            wrong = ToolCall(first.name, {**first.arguments, "__wrong__": True})
            return (wrong, *expected[1:])
        wrong = ToolCall(f"{first.name}_wrong", first.arguments)
        return (wrong, *expected[1:])


def default_tasks() -> tuple[Task, ...]:
    return (
        Task(
            "weather",
            (ToolCall("get_weather", {"city": "Berlin", "units": "celsius"}),),
            1.0,
        ),
        Task(
            "stock",
            (
                ToolCall("get_price", {"symbol": "ACME"}),
                ToolCall("format_currency", {"value": 12.5, "currency": "EUR"}),
            ),
            0.8,
        ),
        Task(
            "duplicate-tool",
            (
                ToolCall("add_tag", {"tag": "agent"}),
                ToolCall("add_tag", {"tag": "evaluation"}),
            ),
            0.6,
        ),
        Task(
            "search",
            (ToolCall("search", {"query": "reliability", "limit": 3}),),
            0.35,
        ),
    )


def with_reliability(task: Task, probability: float) -> Task:
    return replace(task, success_probability=probability)
