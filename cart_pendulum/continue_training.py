"""Continue an existing PPO controller in evaluated blocks; no API key needed."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .archives import retain_architecture
from .environment import EnvConfig
from .learning import NetworkConfig, evaluate, score, train, write_json
from .memory import DEFAULT_MEMORY, ExperimentMemory
from .run_files import initialize_run, recorded_run


def copy_checkpoint(trial, output, name):
    for original, target in (("model.zip", f"{name}_model.zip"), ("config.json", f"{name}_config.json")):
        temporary = output / (target + ".tmp")
        shutil.copyfile(trial / original, temporary)
        temporary.replace(output / target)


@recorded_run
def continue_training(output, source, *, blocks=4, steps=65536, max_minutes=60,
                      seed=101, validation_episodes=8, checkpoint="best", memory_file=DEFAULT_MEMORY):
    from stable_baselines3 import PPO
    import torch

    if type(blocks) is not int or not 1 <= blocks <= 100:
        raise ValueError("Use 1–100 training blocks.")
    if type(steps) is not int or steps < 512 or steps % 512:
        raise ValueError("Steps per block must be a positive multiple of 512.")
    if not np.isfinite(max_minutes) or max_minutes <= 0 or not 1 <= validation_episodes <= 32:
        raise ValueError("Use a positive time limit and 1–32 validation episodes.")
    if checkpoint not in ("best", "latest"):
        raise ValueError("Checkpoint must be best or latest.")
    source, output = Path(source), Path(output)
    saved = json.loads((source / "best_config.json").read_text())
    config, network = EnvConfig(**saved["environment"]), NetworkConfig(**saved["network"])
    start_config = json.loads((source / f"{checkpoint}_config.json").read_text())
    if EnvConfig(**start_config["environment"]) != config or NetworkConfig(**start_config["network"]) != network:
        raise ValueError("Latest and best checkpoints must have identical environment and network settings.")
    source_model = source / f"{checkpoint}_model.zip"
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    settings = {"mode": "continuation", "environment": asdict(config), "network": asdict(network),
                "source_run": str(source.resolve()), "checkpoint": checkpoint, "blocks": blocks,
                "steps_per_block": steps, "max_minutes": max_minutes, "seed": seed,
                "validation_episodes": validation_episodes, "memory_file": str(Path(memory_file).resolve())}
    write_json(output / "settings.json", settings)
    initialize_run(output, settings)
    deadline = time.monotonic() + max_minutes * 60
    memory = ExperimentMemory(memory_file)
    write_json(output / "memory_at_start.json", {"records": memory.records(config)})
    shutil.copyfile(source_model, output / "starting_model.zip")
    write_json(output / "starting_config.json", start_config)
    write_json(output / "parent.json", {"checkpoint": str(source_model.resolve()),
               "sha256": hashlib.sha256((output / "starting_model.zip").read_bytes()).hexdigest()})
    # Keep the parent's best as the incumbent even when continuing its latest weights.
    initial = output / "trial_000"
    initial.mkdir()
    shutil.copyfile(source / "best_model.zip", initial / "model.zip")
    policy = PPO.load(initial / "model.zip", device="cpu")
    write_json(initial / "config.json", {**saved, "environment": asdict(config),
               "training_mode": "imported_checkpoint", "initial_steps": policy.num_timesteps,
               "requested_steps": policy.num_timesteps, "requested_additional_steps": 0,
               "parent_checkpoint": str((source / "best_model.zip").resolve())})
    write_json(initial / "training.json", {"actual_steps": policy.num_timesteps,
               "initial_steps": policy.num_timesteps, "added_steps": 0,
               "gradient_updates": policy._n_updates, "added_gradient_updates": 0,
               "completed_requested_steps": True, "interrupted": False, "seconds": 0})
    validation_seeds = list(range(10000, 10000 + validation_episodes))
    print(f"Evaluating starting best controller ({policy.num_timesteps:,} prior steps)...", flush=True)
    metrics = evaluate(policy, config, validation_seeds, initial / "validation_trajectory.csv")
    write_json(initial / "validation.json", metrics)
    best = {"trial": initial.name, "network": asdict(network), "validation": metrics,
            "rationale": "Imported prior best checkpoint; evaluated before additional training."}
    history = [best]
    copy_checkpoint(initial, output, "best")
    shutil.copyfile(output / "starting_model.zip", output / "latest_model.zip")
    write_json(output / "latest_config.json", start_config)
    retain_architecture(output, best)
    summary = {"settings": settings, "completed_trials": 0, "completed_blocks": 0,
               "stop_reason": "running", "best": best, "initial_validation": metrics,
               "warning": "Best means best validated checkpoint, not necessarily a solved task."}

    def persist():
        summary["best"] = best
        write_json(output / "history.json", history)
        write_json(output / "summary.json", summary)

    persist()
    parent = output / "starting_model.zip"
    reason = "block_limit"
    for index in range(1, blocks + 1):
        if time.monotonic() >= deadline or (output / "STOP").exists():
            reason = "time_limit_or_stop_file"
            break
        trial = output / f"trial_{index:03}"
        phase = "training"
        rationale = f"Continue unchanged PPO for up to {steps} additional steps; block {index}."
        print(f"Block {index}/{blocks}: {steps:,} additional steps", flush=True)
        try:
            try:
                policy = train(config, network, steps, seed + index - 1, trial,
                               deadline=deadline, checkpoint=parent, report_every=8192)
            finally:
                # Interrupted/partially trained weights remain available as latest.
                if (trial / "model.zip").is_file() and (trial / "config.json").is_file():
                    copy_checkpoint(trial, output, "latest")
            phase = "validation"
            metrics = evaluate(policy, config, validation_seeds, trial / "validation_trajectory.csv")
            write_json(trial / "validation.json", metrics)
            record = {"trial": trial.name, "network": asdict(network), "validation": metrics, "rationale": rationale}
            history.append(record)
            if score(metrics) > score(best["validation"]):
                best = record
                copy_checkpoint(trial, output, "best")
            summary["completed_blocks"] += 1
            summary["completed_trials"] += 1
            persist()
            phase = "saving_results"
            memory.remember(output, record)
            retain_architecture(output, record)
            details = json.loads((trial / "training.json").read_text())
            print(f"  total steps: {details['actual_steps']:,}; validation: {metrics['mean_duration']:.3f}s; "
                  f"success: {metrics['success_rate']:.0%}; return: {metrics['mean_return']:.2f}", flush=True)
            if config.task == "swingup":
                print(f"  mean final settled hold: {metrics['mean_final_hold']:.3f}s", flush=True)
            parent = trial / "model.zip"
            if not details["completed_requested_steps"]:
                reason = "time_limit_or_stop_file"
                break
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
            break
        except Exception as error:
            reason = f"error:{type(error).__name__}"
            write_json(output / f"error_{index:03}.json", {"phase": phase, "error_type": type(error).__name__})
            memory.remember_failure(output, trial.name, config, phase, type(error).__name__, network, rationale)
            print(f"Stopped after {type(error).__name__}; best and latest checkpoints are preserved.", flush=True)
            break
    summary["stop_reason"] = reason
    persist()
    if reason != "keyboard_interrupt" and not reason.startswith("error:"):
        print("Evaluating the selected best checkpoint on held-out episodes...", flush=True)
        selected = PPO.load(output / "best_model.zip", device="cpu")
        summary["holdout"] = evaluate(selected, config, range(20000, 20000 + validation_episodes),
                                      output / "holdout_trajectory.csv")
        persist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing experiment folder.")
    parser.add_argument("--output", type=Path, required=True, help="New continuation folder.")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=65536, help="Additional steps per block.")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    args = parser.parse_args()
    result = continue_training(args.output, args.source, blocks=args.blocks, steps=args.steps,
                               max_minutes=args.max_minutes, seed=args.seed,
                               validation_episodes=args.evaluation_episodes, checkpoint=args.checkpoint,
                               memory_file=args.memory_file)
    print(json.dumps({"completed_blocks": result["completed_blocks"], "stop_reason": result["stop_reason"],
                      "output": str(args.output)}, indent=2))
    if result["stop_reason"].startswith("error:"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
