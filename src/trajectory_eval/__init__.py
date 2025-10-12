"""Deterministic tool-trajectory reliability evaluation."""

from .agents import ScriptedAgent, default_tasks
from .evaluation import EvaluationReport, evaluate_agent, evaluate_trajectories
from .metrics import evaluate_tool_calls, pass_k, pass_k_moments, summarize_reliability
from .models import Task, ToolCall, Trajectory

__version__ = "0.1.0"

__all__ = [
    "EvaluationReport",
    "ScriptedAgent",
    "Task",
    "ToolCall",
    "Trajectory",
    "default_tasks",
    "evaluate_agent",
    "evaluate_tool_calls",
    "evaluate_trajectories",
    "pass_k",
    "pass_k_moments",
    "summarize_reliability",
]
