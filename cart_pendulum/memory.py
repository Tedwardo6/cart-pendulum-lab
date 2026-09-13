"""Persistent experiment evidence for later proposals; never includes holdout data."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .learning import write_json, score
from .environment import EnvConfig

# Increment when observations, rewards, dynamics or evaluation semantics change.
TASK_VERSION = "planar-balance-v1"
DEFAULT_MEMORY = Path("runs/experiment_memory.json")


def task_version(config):
    return TASK_VERSION if config.task == "balance" else "planar-swingup-v1"


class ExperimentMemory:
    def __init__(self, path=DEFAULT_MEMORY):
        self.path = Path(path)

    def records(self, config):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported experiment memory version.")
        return [r for r in data["records"] if r["task_version"] == task_version(config)
                and asdict(EnvConfig(**r["environment"])) == asdict(config)]

    def remember(self, run, record):
        run = Path(run)
        trial = run / record["trial"]
        config = json.loads((trial / "config.json").read_text())
        training = json.loads((trial / "training.json").read_text())
        metrics = json.loads((trial / "validation.json").read_text())
        identity = hashlib.sha256(str(trial.resolve()).encode()).hexdigest()
        entry = {"id": identity, "status": "completed", "task_version": task_version(EnvConfig(**config["environment"])), "environment": config["environment"],
                 "network": config["network"], "rationale": record.get("rationale", ""),
                 "training_seed": config["seed"], "requested_steps": config["requested_steps"],
                 "actual_steps": training["actual_steps"],
                 "validation_seeds": [e["seed"] for e in metrics["episodes"]],
                 "validation": {k: metrics[k] for k in ("task", "mean_duration", "mean_return", "success_rate",
                                                       "mean_upright_time", "mean_final_hold") if k in metrics},
                 "source_run": str(run.resolve()), "source_trial": record["trial"]}
        self._save(entry)

    def remember_failure(self, run, trial, config, phase, error_type, network=None, rationale=""):
        identity = hashlib.sha256(str((Path(run) / trial).resolve()).encode()).hexdigest()
        self._save({"id": identity, "status": "failed", "task_version": task_version(config),
            "environment": asdict(config), "source_run": str(Path(run).resolve()), "source_trial": trial,
            "phase": phase, "error_type": error_type, "network": asdict(network) if network else None,
            "rationale": rationale})

    def _save(self, entry):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "records": []}
            if data.get("version") != 1:
                raise ValueError("Unsupported experiment memory version.")
            previous = next((r for r in data["records"] if r["id"] == entry["id"]), None)
            if previous and previous["status"] == "completed" and entry["status"] == "failed":
                # An archive error must not erase a successfully evaluated result.
                entry = {**previous, "post_evaluation_error": {
                    "phase": entry["phase"], "error_type": entry["error_type"]}}
            data["records"] = [r for r in data["records"] if r["id"] != entry["id"]]
            data["records"].append(entry)
            write_json(self.path, data)


def proposal_evidence(records, limit=8):
    """Retain all evidence on disk; send a bounded mix of best and recent results."""
    records = [r for r in records if r.get("status", "completed") == "completed"]
    selected = []
    candidates = sorted(records, key=lambda r: score(r["validation"]), reverse=True)[:limit//2]
    candidates += records[-limit:]
    for record in candidates:
        # Exclude local filesystem paths; include conditions needed to interpret scores.
        compact = {k: record.get(k) for k in ("id", "network", "training_seed", "requested_steps",
                   "actual_steps", "validation_seeds", "validation")}
        compact["rationale"] = record.get("rationale", "")[:600]
        if compact not in selected:
            selected.append(compact)
    # Highest-ranked candidates first, then the most recent distinct candidates.
    return selected[:limit//2] + selected[limit//2:][-(limit-limit//2):]


def proposal_failures(records):
    return [{"network": r.get("network"), "phase": r["phase"], "error_type": r["error_type"],
             "rationale": r.get("rationale", "")[:300]} for r in records if r.get("status") == "failed"][-3:]


def main():
    parser = argparse.ArgumentParser(description="Import completed runs into persistent experiment memory.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    args = parser.parse_args()
    memory = ExperimentMemory(args.memory_file)
    for run in args.runs:
        for record in json.loads((run / "history.json").read_text()):
            memory.remember(run, record)
    print(f"Experiment memory saved: {memory.path.resolve()}")


if __name__ == "__main__":
    main()
