"""Run deterministic repeated trajectory evaluations and write reliability statistics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from trajectory_eval.agents import ScriptedAgent, default_tasks, with_reliability
from trajectory_eval.evaluation import evaluate_agent
from trajectory_eval.metrics import summarize_reliability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    tasks = default_tasks()
    profiles = {
        "perfect": tuple(with_reliability(task, 1.0) for task in tasks),
        "mixed": tasks,
        "weak": tuple(with_reliability(task, task.success_probability * 0.5) for task in tasks),
    }
    reports = []
    for name, profile in profiles.items():
        report = evaluate_agent(
            ScriptedAgent(name), profile, repetitions=args.repetitions, k=args.k, seed=args.seed
        )
        value = report.as_dict()
        outcomes = [(task.successes, task.trials) for task in report.tasks]
        value["pass_1_summary"] = summarize_reliability(outcomes, 1).as_dict()
        reports.append(value)
    result = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "definition": "pass^k estimates probability that all k independent trajectories succeed",
        "deterministic": True,
        "external_api_calls": 0,
        "agents": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
