"""PPO training and fixed-seed evaluation, independent of experiment proposals."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
import numpy as np

from .environment import CartPendulumEnv, EnvConfig


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple = (64, 64)
    activation: str = "tanh"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    entropy_coefficient: float = 0.0
    n_epochs: int = 10

    def __post_init__(self):
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if not 1 <= len(self.hidden_sizes) <= 3 or any(type(n) is not int or not 16 <= n <= 256 for n in self.hidden_sizes):
            raise ValueError("Use 1–3 hidden layers with 16–256 neurons each.")
        if self.activation not in ("tanh", "relu"):
            raise ValueError("Activation must be tanh or relu.")
        for name, lo, hi in [("learning_rate", 1e-5, 0.003), ("gamma", .9, .9999), ("entropy_coefficient", 0, .03)]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}].")
        if type(self.n_epochs) is not int or not 1 <= self.n_epochs <= 20:
            raise ValueError("n_epochs must be 1–20.")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(policy, config, seeds, trajectory=None):
    """Same initial conditions for every candidate; deterministic actions."""
    results = []
    env = CartPendulumEnv(config)
    for index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        history = [[0., *env.state, 0.]]
        total_reward, max_angle, force_squared, elapsed = 0., 0., 0., 0.
        while True:
            action = np.zeros(1) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_angle = max(max_angle, info["max_angle_error"])
            delta = info["time"] - elapsed
            force_squared += info["force"]**2 * delta
            elapsed = info["time"]
            history.append([elapsed, *env.state, info["force"]])
            if terminated or truncated:
                break
        results.append({"seed": int(seed), "duration": elapsed, "return": total_reward,
                        "success": info["success"], "end_reason": info["end_reason"],
                        "upright_time": info["upright_time"], "final_hold": info["final_hold"],
                        "max_angle_error": max_angle, "rms_force": float(np.sqrt(force_squared / max(elapsed, 1e-9)))})
        if trajectory is not None and index == 0:
            header = ["time", "x", *[f"theta_{i+1}" for i in range(config.n_links)],
                      "x_dot", *[f"omega_{i+1}" for i in range(config.n_links)], "force"]
            np.savetxt(trajectory, history, delimiter=",", header=",".join(header), comments="")
    env.close()
    return {"task": config.task,
            "mean_upright_time": float(np.mean([r["upright_time"] for r in results])),
            "mean_final_hold": float(np.mean([r["final_hold"] for r in results])),
            "mean_duration": float(np.mean([r["duration"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_return": float(np.mean([r["return"] for r in results])), "episodes": results}


def score(metrics):
    """Task-specific selection. Hanging for the full episode is not swing-up success."""
    if metrics.get("task") == "swingup":
        return metrics["success_rate"], metrics["mean_final_hold"], metrics["mean_return"]
    return metrics["mean_duration"], metrics["mean_return"]


def train(config, network, steps, seed, output, deadline=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    if type(steps) is not int or steps < 512 or steps % 512:
        raise ValueError("Training steps must be a positive multiple of 512.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    write_json(output / "config.json", {"environment": asdict(config), "network": asdict(network),
                                       "seed": seed, "requested_steps": steps})
    class BudgetCallback(BaseCallback):
        def _on_step(self):
            return not ((output.parent / "STOP").exists() or (deadline and time.monotonic() >= deadline))

    env = Monitor(CartPendulumEnv(config), str(output / "episodes.csv"))
    policy = PPO("MlpPolicy", env, policy_kwargs={"net_arch": list(network.hidden_sizes),
                 "activation_fn": torch.nn.Tanh if network.activation == "tanh" else torch.nn.ReLU},
                 learning_rate=network.learning_rate, gamma=network.gamma,
                 ent_coef=network.entropy_coefficient, n_epochs=network.n_epochs,
                 n_steps=512, batch_size=64, seed=seed, device="cpu", verbose=0)
    started = time.monotonic()
    interrupted = False
    failure = None
    try:
        policy.learn(total_timesteps=steps, callback=BudgetCallback())
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = error
    finally:
        policy.save(output / "model.zip")
        env.close()
    write_json(output / "training.json", {"actual_steps": policy.num_timesteps,
               "gradient_updates": policy._n_updates,
               "seconds": time.monotonic() - started, "interrupted": interrupted,
               "completed_requested_steps": policy.num_timesteps >= steps})
    if failure is not None:
        raise failure
    if interrupted:
        raise KeyboardInterrupt
    if policy._n_updates == 0:
        raise RuntimeError("Stopped before any neural-network weight updates; untrained checkpoint saved.")
    return policy
