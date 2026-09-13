"""Keep the best evaluated checkpoint for every distinct MLP architecture."""

import argparse
import json
from pathlib import Path
import shutil

from .learning import NetworkConfig, score, write_json


def retain_architecture(run, record):
    """Group by activation and ordered layer sizes within one experiment run.

    Learning rate and other training settings are not part of the architecture.
    Only completed, evaluated trials qualify. Original trial files stay intact.
    """
    run = Path(run)
    trial_name = record["trial"]
    if Path(trial_name).name != trial_name or trial_name in (".", ".."):
        raise ValueError("Trial name must identify a directory inside the run.")
    trial = run / trial_name
    config = json.loads((trial / "config.json").read_text())
    network = NetworkConfig(**config["network"])
    metrics = json.loads((trial / "validation.json").read_text())
    if not (trial / "model.zip").is_file():
        raise FileNotFoundError("An evaluated model is required for archiving.")
    architecture = f"mlp-{network.activation}-" + "x".join(map(str, network.hidden_sizes))
    destination = run / "architectures" / architecture
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best_result.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    comparison = {"environment": config["environment"],
                  "validation_seeds": [episode["seed"] for episode in metrics["episodes"]]}
    if best and best["comparison"] != comparison:
        raise ValueError("Cannot rank models tested with different environments or validation seeds.")

    result = {"architecture": architecture, "trial": trial_name,
              "network": config["network"], "training_seed": config["seed"],
              "validation": metrics, "comparison": comparison,
              "rationale": record.get("rationale", ""),
              "selection_rule": "Mean validation survival first, mean return second; ties keep the incumbent."}
    improved = best is None or score(metrics) > score(best["validation"])
    if improved:
        # Each copied file is replaced only after the new copy is complete.
        for source, target in [("model.zip", "best_model.zip"), ("config.json", "best_config.json"),
                               ("validation_trajectory.csv", "validation_trajectory.csv"),
                               ("training.json", "training.json")]:
            if (trial / source).exists():
                temporary = destination / (target + ".tmp")
                shutil.copyfile(trial / source, temporary)
                temporary.replace(destination / target)
            elif (destination / target).exists():
                # Optional data from an older winner must not describe this model.
                (destination / target).unlink()
        write_json(best_path, result)
        best = result

    history_path = destination / "trials.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [entry for entry in history if entry["trial"] != trial_name]
    history.append(result)
    write_json(history_path, history)
    index_path = run / "architectures" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[architecture] = {"best_trial": best["trial"], "hidden_sizes": network.hidden_sizes,
                           "activation": network.activation, "tested_trials": len(history),
                           "mean_duration": best["validation"]["mean_duration"],
                           "mean_return": best["validation"]["mean_return"],
                           "model": f"{architecture}/best_model.zip"}
    write_json(index_path, index)
    return destination


def backfill(run):
    """Organize existing evaluated trials without retraining or API calls."""
    run = Path(run)
    for record in json.loads((run / "history.json").read_text()):
        retain_architecture(run, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        backfill(run)
        print(f"Architecture archive ready: {run / 'architectures'}")


if __name__ == "__main__":
    main()
