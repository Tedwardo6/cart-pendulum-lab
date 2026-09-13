"""Make a GIF of an experiment's selected best controller without API calls."""

import argparse
import json
from pathlib import Path
import numpy as np


def best_replay(run):
    """Prefer the winner's first held-out episode; fall back to its validation replay."""
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    best = summary.get("best")
    if not best:
        raise ValueError("This experiment has no selected best controller yet.")
    trial_name = best["trial"]
    if Path(trial_name).name != trial_name or trial_name in (".", ".."):
        raise ValueError("Invalid selected trial directory.")
    config = json.loads((run / trial_name / "config.json").read_text())
    if summary.get("holdout") and (run / "holdout_trajectory.csv").is_file():
        trajectory = run / "holdout_trajectory.csv"
        episode = summary["holdout"]["episodes"][0]
        label = "Held-out"
    else:
        trajectory = run / trial_name / "validation_trajectory.csv"
        episode = best["validation"]["episodes"][0]
        label = "Validation"
    data = np.loadtxt(trajectory, delimiter=",", skiprows=1, ndmin=2)
    n = config["environment"]["n_links"]
    if (data.shape[1] != 2 * (n + 1) + 2 or not np.all(np.isfinite(data))
            or data[0, 0] != 0 or np.any(np.diff(data[:, 0]) <= 0)
            or not np.isclose(data[-1, 0], episode["duration"])):
        raise ValueError("Saved trajectory is invalid or does not match episode duration.")
    return {"config": config["environment"], "network": config["network"],
            "data": data, "episode": episode, "title": f"Best: {trial_name}", "label": label}, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="An experiment folder containing summary.json.")
    parser.add_argument("--output", type=Path, help="GIF path; defaults to RUN/renders/best.gif.")
    args = parser.parse_args()
    output = args.output or args.run / "renders" / "best.gif"
    if output.suffix.lower() != ".gif":
        parser.error("Output must end in .gif.")
    try:
        record, trajectory = best_replay(args.run)
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.error(f"Cannot load a saved best-controller replay: {error}")
    from .rendering import render_trajectories
    print(f"Rendering {record['title']} — {record['label']} seed {record['episode']['seed']}...", flush=True)
    render_trajectories([record], output)
    output.with_suffix(".json").write_text(json.dumps({
        "source_run": str(args.run.resolve()), "source_trajectory": str(trajectory.resolve()),
        "controller": record["title"], "episode_kind": record["label"], "episode": record["episode"],
        "note": "First saved episode of the selected best controller, not the highest-scoring episode."
    }, indent=2) + "\n")
    print(f"GIF saved: {output.resolve()}")


if __name__ == "__main__":
    main()
