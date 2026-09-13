"""Train swing-up from balancing skills, with bounded optional API reward proposals."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

from .archives import retain_architecture
from .continue_training import copy_checkpoint
from .curriculum_proposals import allowed_stages, propose_local, propose_openai
from .environment import EnvConfig
from .experiments import ApiBudget
from .learning import NetworkConfig, evaluate, score, train, write_json
from .memory import DEFAULT_MEMORY, ExperimentMemory
from .run_files import initialize_run, recorded_run
from .training_tasks import EVALUATION_VERSION, ResetConfig


def probe(policy, config, stage, episodes):
    easy = evaluate(policy, config, range(30000, 30000 + episodes), evaluation_version=EVALUATION_VERSION,
                    reset_config=ResetConfig(stage=0, easy_fraction=1, hanging_fraction=0))
    frontier = evaluate(policy, config, range(40000, 40000 + episodes), evaluation_version=EVALUATION_VERSION,
                        reset_config=ResetConfig(stage=stage, easy_fraction=0, hanging_fraction=0, frontier_only=True))
    return {"stage": stage, "retention_success": easy["success_rate"], "frontier_success": frontier["success_rate"],
            "easy": easy, "frontier": frontier}


def compact_probes(probes):
    return {key: probes[key] for key in ("stage", "retention_success", "frontier_success")}


@recorded_run
def run_curriculum(output, source, *, blocks=8, max_steps=65536, max_minutes=60,
                   seed=501, validation_episodes=8, proposer="local", api_budget=20,
                   budget_ledger=".api-budget.json", memory_file=DEFAULT_MEMORY):
    from stable_baselines3 import PPO
    import torch
    if type(blocks) is not int or not 1 <= blocks <= 20:
        raise ValueError("Use 1–20 curriculum blocks.")
    if type(max_steps) is not int or max_steps < 512 or max_steps % 512:
        raise ValueError("Maximum steps per block must be a positive multiple of 512.")
    if not np.isfinite(max_minutes) or max_minutes <= 0 or not 1 <= validation_episodes <= 32:
        raise ValueError("Use a positive time limit and 1–32 evaluation episodes.")
    if proposer not in ("local", "openai"):
        raise ValueError("Unknown proposer.")
    if proposer == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Use --ask-api-key or set OPENAI_API_KEY locally.")
    budget = ApiBudget(budget_ledger, api_budget)
    source, output = Path(source), Path(output)
    saved = json.loads((source / "best_config.json").read_text())
    config = replace(EnvConfig(**saved["environment"]), task="swingup")
    # Curriculum reruns build on the last controller that retained its balancing skill.
    start_kind = "accepted" if (source / "accepted_model.zip").exists() else "best"
    start = json.loads((source / f"{start_kind}_config.json").read_text())
    if replace(EnvConfig(**start["environment"]), task="swingup") != config:
        raise ValueError("Source best and accepted checkpoints have incompatible physics.")
    network = NetworkConfig(**start["network"])
    previous_spec = start.get("training_spec")
    stage = previous_spec["reset"]["stage"] if previous_spec else 0
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    settings = {"mode": "curriculum", "environment": asdict(config), "evaluation_version": EVALUATION_VERSION,
                "source_run": str(source.resolve()), "source_checkpoint": start_kind, "blocks": blocks,
                "max_steps_per_block": max_steps, "max_minutes": max_minutes, "seed": seed,
                "validation_episodes": validation_episodes, "proposer": proposer,
                "memory_file": str(Path(memory_file).resolve()), "api_budget": api_budget,
                "budget_ledger": str(Path(budget_ledger).resolve())}
    write_json(output / "settings.json", settings)
    initialize_run(output, settings)
    deadline = time.monotonic() + max_minutes * 60
    memory = ExperimentMemory(memory_file)
    write_json(output / "memory_at_start.json", {"records": memory.records(config, EVALUATION_VERSION)})
    parent = output / "starting_model.zip"
    shutil.copyfile(source / f"{start_kind}_model.zip", parent)
    write_json(output / "starting_config.json", start)
    write_json(output / "parent.json", {"checkpoint": str((source / f"{start_kind}_model.zip").resolve()),
               "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()})
    current = PPO.load(parent, device="cpu")
    transfer_origin = (previous_spec or {}).get("transfer_origin") or {
        "task": start["environment"].get("task", "balance"), "prior_steps": current.num_timesteps}
    print("Checking retained balancing skill and current recovery difficulty...", flush=True)
    probes = probe(current, config, stage, validation_episodes)
    write_json(output / "initial_probes.json", probes)
    if probes["retention_success"] < .75:
        raise ValueError("The source fails the easy-start settling check. Start from a successful balance controller.")
    for name in ("accepted", "latest"):
        shutil.copyfile(parent, output / f"{name}_model.zip")
        write_json(output / f"{name}_config.json", start)
    initial = output / "trial_000"
    initial.mkdir()
    shutil.copyfile(source / "best_model.zip", initial / "model.zip")
    policy = PPO.load(initial / "model.zip", device="cpu")
    write_json(initial / "config.json", {**saved, "environment": asdict(config),
               "evaluation_version": EVALUATION_VERSION, "training_mode": "imported_checkpoint",
               "requested_steps": policy.num_timesteps, "initial_steps": policy.num_timesteps,
               "requested_additional_steps": 0, "parent_checkpoint": str((source / "best_model.zip").resolve())})
    write_json(initial / "training.json", {"actual_steps": policy.num_timesteps, "initial_steps": policy.num_timesteps,
               "added_steps": 0, "gradient_updates": policy._n_updates, "added_gradient_updates": 0,
               "completed_requested_steps": True, "interrupted": False, "seconds": 0})
    validation_seeds = list(range(10000, 10000 + validation_episodes))
    print("Evaluating the original best controller from hanging-down starts...", flush=True)
    metrics = evaluate(policy, config, validation_seeds, initial / "validation_trajectory.csv", evaluation_version=EVALUATION_VERSION)
    write_json(initial / "validation.json", metrics)
    best = {"trial": initial.name, "network": saved["network"], "validation": metrics,
            "rationale": "Reevaluate parent best on the fixed hanging-start objective before changing training."}
    history = [best]
    copy_checkpoint(initial, output, "best")
    retain_architecture(output, best)
    summary = {"settings": settings, "completed_trials": 0, "completed_blocks": 0,
               "stop_reason": "running", "best": best, "initial_validation": metrics,
               "warning": "Curriculum probe success is not hanging-start swing-up success."}

    def persist():
        summary["best"] = best
        write_json(output / "summary.json", summary)
        write_json(output / "history.json", history)
        write_json(output / "curriculum_state.json", {"stage": stage, "parent": str(parent.resolve()),
                   "probes": compact_probes(probes), "training_spec": previous_spec})

    persist()
    reason = "block_limit"
    for index in range(1, blocks + 1):
        if time.monotonic() >= deadline or (output / "STOP").exists():
            reason = "time_limit_or_stop_file"
            break
        trial = output / f"trial_{index:03}"
        phase, rationale, spec, candidate = "proposal", "", None, network
        try:
            stages = allowed_stages(stage, probes["frontier_success"], probes["retention_success"])
            if proposer == "openai":
                candidate, spec, steps, rationale = propose_openai(config, network, stage, stages, max_steps,
                    memory.records(config, EVALUATION_VERSION), compact_probes(probes), budget, output / f"api_{index:03}.json")
            else:
                candidate, spec, steps, rationale = propose_local(network, stage, stages, max_steps,
                    previous_spec["reward"] if previous_spec else None)
            spec["transfer_origin"] = transfer_origin
            write_json(output / f"proposal_{index:03}.json", {"network": asdict(candidate), "training_spec": spec,
                       "steps": steps, "rationale": rationale, "allowed_stages": stages})
            if time.monotonic() >= deadline or (output / "STOP").exists():
                reason = "time_limit_or_stop_file"
                break
            reset_critic = previous_spec is None or previous_spec["reward"] != spec["reward"]
            phase = "training"
            print(f"Block {index}/{blocks}: stage {spec['reset']['stage']}; {steps:,} steps; critic reset={reset_critic}", flush=True)
            try:
                policy = train(config, candidate, steps, seed + index - 1, trial, deadline=deadline,
                    checkpoint=parent, report_every=8192, training_spec=spec,
                    update_hyperparameters=True, reset_critic=reset_critic)
            finally:
                if (trial / "model.zip").exists() and (trial / "config.json").exists():
                    copy_checkpoint(trial, output, "latest")
            phase = "validation"
            metrics = evaluate(policy, config, validation_seeds, trial / "validation_trajectory.csv", evaluation_version=EVALUATION_VERSION)
            candidate_probes = probe(policy, config, spec["reset"]["stage"], validation_episodes)
            accepted = candidate_probes["retention_success"] >= .75
            write_json(trial / "validation.json", metrics)
            write_json(trial / "probes.json", candidate_probes)
            record = {"trial": trial.name, "network": asdict(candidate), "training_spec": spec, "rationale": rationale,
                      "validation": metrics, "probes": compact_probes(candidate_probes), "accepted_for_training": accepted}
            history.append(record)
            if score(metrics) > score(best["validation"]):
                best = record
                copy_checkpoint(trial, output, "best")
            if accepted:
                parent, previous_spec, network = trial / "model.zip", spec, candidate
                stage, probes = spec["reset"]["stage"], candidate_probes
                copy_checkpoint(trial, output, "accepted")
            summary["completed_blocks"] += 1
            summary["completed_trials"] += 1
            persist()
            phase = "saving_results"
            memory.remember(output, record)
            retain_architecture(output, record)
            print(f"  hanging-start success: {metrics['success_rate']:.0%}; hold: {metrics['mean_final_hold']:.2f}s; "
                  f"common score: {metrics['mean_common_score']:.4f}", flush=True)
            print(f"  recovery: {candidate_probes['frontier_success']:.0%}; retention: {candidate_probes['retention_success']:.0%}; "
                  f"{'accepted' if accepted else 'rejected; previous training parent retained'}", flush=True)
            if not json.loads((trial / "training.json").read_text())["completed_requested_steps"]:
                reason = "time_limit_or_stop_file"
                break
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
            break
        except Exception as error:
            reason = f"error:{type(error).__name__}"
            write_json(output / f"error_{index:03}.json", {"phase": phase, "error_type": type(error).__name__})
            memory.remember_failure(output, trial.name, config, phase, type(error).__name__, candidate, rationale,
                                    evaluation_version=EVALUATION_VERSION, training_spec=spec)
            print(f"Stopped after {type(error).__name__}; checkpoints preserved.", flush=True)
            break
    summary["stop_reason"] = reason
    persist()
    if reason != "keyboard_interrupt" and not reason.startswith("error:"):
        print("Evaluating the selected winner on held-out hanging starts...", flush=True)
        policy = PPO.load(output / "best_model.zip", device="cpu")
        summary["holdout"] = evaluate(policy, config, range(20000, 20000 + validation_episodes),
                                      output / "holdout_trajectory.csv", evaluation_version=EVALUATION_VERSION)
        persist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=65536)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--proposer", choices=["local", "openai"], default="local")
    parser.add_argument("--api-budget", type=float, default=20.)
    parser.add_argument("--budget-ledger", type=Path, default=Path(".api-budget.json"))
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--ask-api-key", action="store_true")
    args = parser.parse_args()
    if args.ask_api_key:
        if args.proposer != "openai":
            parser.error("--ask-api-key requires --proposer openai.")
        import getpass
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key (hidden; not saved): ")
    result = run_curriculum(args.output, args.source, blocks=args.blocks, max_steps=args.max_steps,
        max_minutes=args.max_minutes, seed=args.seed, validation_episodes=args.evaluation_episodes,
        proposer=args.proposer, api_budget=args.api_budget, budget_ledger=args.budget_ledger, memory_file=args.memory_file)
    print(json.dumps({"completed_blocks": result["completed_blocks"], "stop_reason": result["stop_reason"], "output": str(args.output)}, indent=2))
    if result["stop_reason"].startswith("error:"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
