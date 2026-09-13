"""Reload a saved controller and run deterministic, unrendered evaluation."""

import argparse
import json
from pathlib import Path
from .environment import EnvConfig
from .learning import evaluate, write_json


def main():
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("Use 1–100 episodes.")
    config = EnvConfig(**json.loads((args.run / "best_config.json").read_text())["environment"])
    model = PPO.load(args.run / "best_model.zip", device="cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    results = evaluate(model, config, range(args.seed, args.seed + args.episodes), args.output / "trajectory.csv")
    write_json(args.output / "evaluation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
