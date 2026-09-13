"""The agent-facing world: normalized force in, observation and reward out."""

from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .integration import rk4_step
from .planar import PlanarChain


@dataclass(frozen=True)
class EnvConfig:
    n_links: int = 2
    cart_mass: float = 1.5
    rod_mass: float = 0.6
    total_length: float = 2.0
    max_force: float = 20.0
    control_dt: float = 0.02
    physics_dt: float = 0.002
    duration: float = 12.0
    track_limit: float = 2.4
    angle_limit: float = 0.35
    initial_angle_range: float = 0.03
    task: str = "balance"
    hold_seconds: float = 2.0
    settle_angular_speed: float = 1.0
    settle_cart_speed: float = 1.0

    def __post_init__(self):
        if type(self.n_links) is not int or not 1 <= self.n_links <= 4:
            raise ValueError("n_links must be 1, 2, 3, or 4.")
        if self.task not in ("balance", "swingup"):
            raise ValueError("task must be balance or swingup.")
        for key, value in vars(self).items():
            if key == "task":
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if self.initial_angle_range >= self.angle_limit:
            raise ValueError("Initial angles must be inside the balancing limit.")
        if self.task == "swingup" and self.hold_seconds > self.duration:
            raise ValueError("The swing-up hold time cannot exceed episode duration.")
        for ratio in (self.control_dt / self.physics_dt, self.duration / self.control_dt):
            if ratio < 1 or not np.isclose(ratio, round(ratio)):
                raise ValueError("Timing intervals must divide evenly.")


class CartPendulumEnv(gym.Env):
    """Balancing or hanging-start swing-up. No drawing or future observations.

    Action: shape (1,), bounded [-1, 1], scaled into newtons.
    Observation: [x/track_limit, x_dot/5, sin(theta), cos(theta), omega/10, ...].
    The angular triples are interleaved, one triple per rod.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or EnvConfig()
        c = self.config
        self.chain = PlanarChain(c.cart_mass, (c.total_length / c.n_links,) * c.n_links,
                                 (c.rod_mass,) * c.n_links)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 + 3*c.n_links,), dtype=np.float32)
        self.state = None
        self.time = 0.0
        self._done = True

    def observation(self):
        c = self.config
        q, v = self.chain._split_state(self.state)
        rods = np.column_stack((np.sin(q[1:]), np.cos(q[1:]), v[1:] / 10))
        return np.concatenate(([q[0] / c.track_limit, v[0] / 5], rods.ravel())).astype(np.float32)

    def angle_errors(self):
        angles = self.state[1:self.config.n_links + 1]
        return np.arctan2(np.sin(angles - np.pi), np.cos(angles - np.pi))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self.state = np.zeros(2*(c.n_links + 1))
        self.state[0] = self.np_random.uniform(-0.02, 0.02)
        target = np.pi if c.task == "balance" else 0.0
        self.state[1:c.n_links + 1] = target + self.np_random.uniform(
            -c.initial_angle_range, c.initial_angle_range, c.n_links)
        self.time = 0.0
        self.upright_time = 0.0
        self.final_hold = 0.0
        self._done = False
        return self.observation(), {"time": self.time}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain one finite number.")
        normalized = float(np.clip(action[0], -1, 1))
        force = normalized * self.config.max_force
        c = self.config
        reason = None
        elapsed = 0.0
        for _ in range(round(c.control_dt / c.physics_dt)):
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    next_state = rk4_step(lambda s: self.chain.derivative(s, force), self.state, c.physics_dt)
                if not np.all(np.isfinite(next_state)):
                    raise FloatingPointError()
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                reason = "numerical_failure"
                break
            self.state = next_state
            self.time += c.physics_dt
            elapsed += c.physics_dt
            if abs(self.state[0]) > c.track_limit:
                reason = "track_limit"
                break
            if c.task == "swingup":
                v = self.state[c.n_links + 1:]
                settled = (np.max(np.abs(self.angle_errors())) <= c.angle_limit
                           and np.max(np.abs(v[1:])) <= c.settle_angular_speed
                           and abs(v[0]) <= c.settle_cart_speed)
                self.final_hold = self.final_hold + c.physics_dt if settled else 0.0
                self.upright_time += c.physics_dt if settled else 0.0
            if c.task == "balance" and np.max(np.abs(self.angle_errors())) > c.angle_limit:
                reason = "angle_limit"
                break
        terminated = reason is not None
        truncated = not terminated and self.time >= c.duration - 1e-9
        self._done = terminated or truncated
        angles = self.angle_errors()
        velocity = self.state[c.n_links + 1:]
        cost = (.4 * np.mean((angles / c.angle_limit)**2)
                + .1 * (self.state[0] / c.track_limit)**2
                + .01 * (velocity[0] / 5)**2 + .01 * np.mean((velocity[1:] / 10)**2)
                + .005 * normalized**2)
        reward = float((1 - cost) * elapsed / c.control_dt - float(terminated))
        if c.task == "swingup":
            # 0 hanging, 1 upright; smooth feedback before the first successful swing.
            height = float(np.mean((1 - np.cos(self.state[1:c.n_links + 1])) / 2))
            centered = 1 - .25 * min(1., (self.state[0] / c.track_limit)**2)
            slow = .5 + .5 / (1 + np.mean((velocity[1:] / 5)**2))
            reward = float((height * centered * slow * (1 - .05 * normalized**2)
                            + .5 * float(self.final_hold > 0)) * elapsed / c.control_dt
                           - 5 * float(terminated))
        info = {"time": self.time, "force": force, "end_reason": reason or ("time_limit" if truncated else None),
                "max_angle_error": float(np.max(np.abs(angles))), "cart_position": float(self.state[0]),
                "success": bool(truncated and (c.task == "balance" or self.final_hold >= c.hold_seconds - 1e-9)),
                "upright_time": self.upright_time, "final_hold": self.final_hold}
        return self.observation(), reward, terminated, truncated, info
