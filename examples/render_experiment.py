"""Replay saved validation trajectories side by side, without training or API calls."""

import argparse
import json
from pathlib import Path
import numpy as np
from cart_pendulum.rendering import render_trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = json.loads((args.run / "history.json").read_text())
    if not history:
        parser.error("The run has no completed trials.")
    records = []
    for record in history:
        trial = args.run / record["trial"]
        records.append({"config": json.loads((trial / "config.json").read_text())["environment"],
            "data": np.loadtxt(trial / "validation_trajectory.csv", delimiter=",", skiprows=1, ndmin=2),
            "episode": record["validation"]["episodes"][0], "network": record["network"],
            "title": record["trial"], "label": "Validation"})
    args.output.mkdir(parents=True, exist_ok=False)
    print(render_trajectories(records, args.output / "comparison.gif", args.output / "final-frame.png"))


if __name__ == "__main__":
    main()
