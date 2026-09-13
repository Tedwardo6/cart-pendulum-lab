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


def evaluate(policy, config, seeds, trajectory=None, *, evaluation_version=None, reset_config=None):
    """Same initial conditions for every candidate; deterministic actions."""
    results = []
    from .training_tasks import CurriculumEnv, EVALUATION_VERSION, common_score
    if evaluation_version not in (None, EVALUATION_VERSION):
        raise ValueError("Unknown evaluation version.")
    if evaluation_version is not None and config.task != "swingup":
        raise ValueError("The fixed swing-up evaluator requires task=swingup.")
    env = CartPendulumEnv(config) if reset_config is None else CurriculumEnv(config, reset_config)
    for index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        history = [[0., *env.state, 0.]]
        total_reward, max_angle, force_squared, elapsed = 0., 0., 0., 0.
        common_total = 0.
        peak_x, peak_speed, saturated_time, edge_time = abs(env.state[0]), 0., 0., 0.
        while True:
            action = np.zeros(1) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_angle = max(max_angle, info["max_angle_error"])
            delta = info["time"] - elapsed
            if evaluation_version is not None:
                common_total += common_score(env.state, config) * delta
            force_squared += info["force"]**2 * delta
            elapsed = info["time"]
            peak_x = max(peak_x, abs(env.state[0]))
            peak_speed = max(peak_speed, abs(env.state[config.n_links + 1]))
            saturated_time += delta * (abs(info["force"]) >= .95 * config.max_force)
            edge_time += delta * (abs(env.state[0]) >= .8 * config.track_limit)
            history.append([elapsed, *env.state, info["force"]])
            if terminated or truncated:
                break
        results.append({"seed": int(seed), "duration": elapsed, "return": total_reward,
                        "success": info["success"], "end_reason": info["end_reason"],
                        "upright_time": info["upright_time"], "final_hold": info["final_hold"],
                        "common_score": common_total / config.duration,
                        "peak_cart_displacement": float(peak_x), "peak_cart_speed": float(peak_speed),
                        "force_saturation_fraction": float(saturated_time / max(elapsed, 1e-9)),
                        "near_edge_fraction": float(edge_time / max(elapsed, 1e-9)),
                        "final_cart_position": float(env.state[0]),
                        "final_cart_speed": float(env.state[config.n_links + 1]),
                        "max_angle_error": max_angle, "rms_force": float(np.sqrt(force_squared / max(elapsed, 1e-9)))})
        if trajectory is not None and index == 0:
            header = ["time", "x", *[f"theta_{i+1}" for i in range(config.n_links)],
                      "x_dot", *[f"omega_{i+1}" for i in range(config.n_links)], "force"]
            np.savetxt(trajectory, history, delimiter=",", header=",".join(header), comments="")
    env.close()
    metrics = {"task": config.task,
            "mean_upright_time": float(np.mean([r["upright_time"] for r in results])),
            "mean_final_hold": float(np.mean([r["final_hold"] for r in results])),
            "mean_duration": float(np.mean([r["duration"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_return": float(np.mean([r["return"] for r in results])), "episodes": results}
    if evaluation_version is not None:
        metrics.update(evaluation_version=evaluation_version,
                       mean_common_score=float(np.mean([r["common_score"] for r in results])))
    metrics["cart_diagnostics"] = {
        "track_exit_rate": float(np.mean([r["end_reason"] == "track_limit" for r in results])),
        "mean_peak_displacement": float(np.mean([r["peak_cart_displacement"] for r in results])),
        "mean_peak_speed": float(np.mean([r["peak_cart_speed"] for r in results])),
        "mean_force_saturation_fraction": float(np.mean([r["force_saturation_fraction"] for r in results])),
        "mean_near_edge_fraction": float(np.mean([r["near_edge_fraction"] for r in results]))}
    return metrics


def score(metrics):
    """Task-specific selection. Hanging for the full episode is not swing-up success."""
    if metrics.get("task") == "swingup":
        return (metrics["success_rate"], metrics["mean_final_hold"],
                metrics["mean_common_score"] if metrics.get("evaluation_version") else metrics["mean_return"])
    return metrics["mean_duration"], metrics["mean_return"]


def train(config, network, steps, seed, output, deadline=None, checkpoint=None, report_every=0,
          training_spec=None, update_hyperparameters=False, reset_critic=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    if type(steps) is not int or steps < 512 or steps % 512:
        raise ValueError("Training steps must be a positive multiple of 512.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    class BudgetCallback(BaseCallback):
        def _on_step(self):
            added = self.model.num_timesteps - initial_steps
            if report_every and added % report_every == 0:
                print(f"  training: {added:,}/{steps:,} additional steps", flush=True)
            return not ((output.parent / "STOP").exists() or (deadline and time.monotonic() >= deadline))

    if training_spec is None:
        training_env = CartPendulumEnv(config)
    else:
        from .training_tasks import CurriculumEnv, RewardConfig, ResetConfig
        training_env = CurriculumEnv(config, ResetConfig(**training_spec["reset"]), RewardConfig(**training_spec["reward"]))
    env = Monitor(training_env, str(output / "episodes.csv"))
    if checkpoint is not None:
        policy = PPO.load(checkpoint, env=env, device="cpu")
        # New episodes/exploration stream, while weights and Adam state are retained.
        policy.set_random_seed(seed)
        if policy.n_steps != 512 or policy.n_envs != 1:
            env.close()
            raise ValueError("Continuation expects this project's single-environment, 512-step PPO setup.")
        if update_hyperparameters:
            policy.learning_rate = network.learning_rate
            policy._setup_lr_schedule()
            policy.ent_coef, policy.n_epochs, policy.gamma = network.entropy_coefficient, network.n_epochs, network.gamma
            policy.rollout_buffer.gamma = network.gamma
        if reset_critic:
            reset_value_network(policy)
    else:
        policy = PPO("MlpPolicy", env, policy_kwargs={"net_arch": list(network.hidden_sizes),
                 "activation_fn": torch.nn.Tanh if network.activation == "tanh" else torch.nn.ReLU},
                 learning_rate=network.learning_rate, gamma=network.gamma,
                 ent_coef=network.entropy_coefficient, n_epochs=network.n_epochs,
                 n_steps=512, batch_size=64, seed=seed, device="cpu", verbose=0)
    if training_spec is not None:
        target_kl = training_spec.get("ppo_target_kl")
        if target_kl is not None and (type(target_kl) not in (int, float) or not np.isfinite(target_kl) or not .001 <= target_kl <= .05):
            env.close()
            raise ValueError("PPO target KL must be in [.001, .05].")
        # Limit update size within a rollout; recovery evaluation still decides acceptance.
        policy.target_kl = target_kl
    initial_steps, initial_updates = policy.num_timesteps, policy._n_updates
    write_json(output / "config.json", {"environment": asdict(config), "network": asdict(network),
               "seed": seed, "requested_steps": initial_steps + steps,
               "requested_additional_steps": steps, "initial_steps": initial_steps,
               "training_mode": "continued" if checkpoint is not None else "fresh",
               "training_spec": training_spec, "critic_reset": reset_critic,
               "evaluation_version": "swingup-fixed-v2" if training_spec is not None else None,
               "parent_checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None})
    started = time.monotonic()
    interrupted = False
    failure = None
    try:
        policy.learn(total_timesteps=steps, callback=BudgetCallback(), reset_num_timesteps=checkpoint is None)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = error
    finally:
        policy.save(output / "model.zip")
        env.close()
    write_json(output / "training.json", {"actual_steps": policy.num_timesteps,
               "initial_steps": initial_steps, "added_steps": policy.num_timesteps - initial_steps,
               "gradient_updates": policy._n_updates,
               "added_gradient_updates": policy._n_updates - initial_updates,
               "seconds": time.monotonic() - started, "interrupted": interrupted,
               "ppo_target_kl": policy.target_kl,
               "completed_requested_steps": policy.num_timesteps >= initial_steps + steps})
    if failure is not None:
        raise failure
    if interrupted:
        raise KeyboardInterrupt
    if policy._n_updates == initial_updates:
        raise RuntimeError("Stopped before any neural-network weight updates; untrained checkpoint saved.")
    return policy


def reset_value_network(model):
    """A changed reward changes the critic's target, not the actor's learned behavior."""
    import torch
    modules = (model.policy.mlp_extractor.value_net, model.policy.value_net)
    for container in modules:
        for layer in container.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=1. if container is model.policy.value_net else np.sqrt(2))
                torch.nn.init.zeros_(layer.bias)
        for parameter in container.parameters():
            model.policy.optimizer.state.pop(parameter, None)
