"""Data models for tool calls and evaluation tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        canonical_arguments(self.arguments)


@dataclass(frozen=True)
class Task:
    task_id: str
    expected: tuple[ToolCall, ...]
    success_probability: float

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task id must not be empty")
        if not 0 <= self.success_probability <= 1:
            raise ValueError("success probability must be in [0, 1]")


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    repetition: int
    expected: tuple[ToolCall, ...]
    actual: tuple[ToolCall, ...]
