"""Training-only resets and rewards. The swing-up evaluation stays separate."""

from dataclasses import dataclass
import numpy as np
from .environment import CartPendulumEnv

EVALUATION_VERSION = "swingup-fixed-v2"
STAGES = ((5, .1), (15, .25), (30, .5), (60, 1.), (100, 1.5), (140, 2.), (180, 2.))
CURRICULUM_VERSION = "recovery-v2"
RECOVERY_STAGES = ((3, 0.), (5, 0.), (7.5, 0.), (10, 0.), (12.5, 0.), (15, 0.),
                   (15, .1), (15, .25), (20, .25), (25, .25), (30, .25), (30, .5),
                   (45, .5), (60, .5), (60, 1.), (90, 1.), (120, 1.), (150, 1.), (180, 1.), (180, 2.))


@dataclass(frozen=True)
class RewardConfig:
    progress: float = .25
    together: float = 1.
    catch: float = 1.
    near_top_speed: float = .1
    effort: float = .002
    centering: float = .05
    braking: float = 0.  # Zero preserves old serialized reward configurations.

    def __post_init__(self):
        bounds = {"progress": (.05, .5), "together": (.5, 3.), "catch": (.5, 3.),
                  "near_top_speed": (0., .5), "effort": (0., .02), "centering": (0., .2), "braking": (0., .5)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Reward {key} must be in [{low}, {high}].")


@dataclass(frozen=True)
class ResetConfig:
    stage: int = 0
    easy_fraction: float = .2
    hanging_fraction: float = .1
    frontier_only: bool = False
    schedule: str = "legacy-v1"

    def __post_init__(self):
        if self.schedule not in ("legacy-v1", CURRICULUM_VERSION):
            raise ValueError("Unknown recovery schedule.")
        stages = STAGES if self.schedule == "legacy-v1" else RECOVERY_STAGES
        if type(self.stage) is not int or not 0 <= self.stage < len(stages):
            raise ValueError(f"Curriculum stage must be 0–{len(stages)-1}.")
        for value in (self.easy_fraction, self.hanging_fraction):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Reset fractions must be in [0, 1].")
        if self.easy_fraction + self.hanging_fraction > 1:
            raise ValueError("Reset fractions cannot exceed 1 in total.")


def state_features(state, config):
    n = config.n_links
    upright = (1 - np.cos(state[1:n+1])) / 2
    v = state[n+1:]
    together = float(np.prod(upright))
    speed = float(np.mean(v[1:]**2))
    catch = together * float(np.exp(-.25 * speed - v[0]**2))
    return float(np.mean(upright)), together, catch, speed, float(v[0])


def shaped_reward(state, config, force_fraction, terminated, elapsed, reward):
    progress, together, catch, speed, cart_speed = state_features(state, config)
    edge = np.clip((abs(state[0]) / config.track_limit - .5) / .5, 0., 1.)
    outward_speed = max(0., np.sign(state[0]) * cart_speed)
    value = (reward.progress * progress + reward.together * together + reward.catch * catch
             - reward.near_top_speed * together * min(speed / 25, 20)
             - reward.effort * force_fraction**2
             - reward.centering * (state[0] / config.track_limit)**2
             - reward.braking * edge**2 * min(outward_speed**2, 25.))
    return float(value * elapsed / config.control_dt - 5 * terminated)


def common_score(state, config):
    """Fixed, bounded diagnostic: joint uprightness and a slow catch, not training return."""
    _, together, catch, _, _ = state_features(state, config)
    return .25 * together + .75 * catch


class CurriculumEnv(CartPendulumEnv):
    def __init__(self, config, reset_config, reward_config=None):
        if config.task != "swingup":
            raise ValueError("Curriculum requires swingup dynamics/termination settings.")
        super().__init__(config)
        self.reset_config = reset_config
        self.reward_config = reward_config

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        r, c = self.reset_config, self.config
        draw = self.np_random.random()
        if not r.frontier_only and draw < r.hanging_fraction:
            kind = "hanging"
        else:
            easy = not r.frontier_only and draw < r.hanging_fraction + r.easy_fraction
            stages = STAGES if r.schedule == "legacy-v1" else RECOVERY_STAGES
            angle, speed = (np.rad2deg(c.initial_angle_range), 0.) if easy else stages[r.stage]
            deviations = self.np_random.uniform(-np.deg2rad(angle), np.deg2rad(angle), c.n_links)
            if r.frontier_only:
                # Require at least one rod near the current difficulty boundary.
                i = int(self.np_random.integers(c.n_links))
                deviations[i] = self.np_random.choice([-1, 1]) * self.np_random.uniform(.75, 1.) * np.deg2rad(angle)
            self.state[1:c.n_links+1] = np.pi + deviations
            self.state[c.n_links+2:] = self.np_random.uniform(-speed, speed, c.n_links)
            self.state[c.n_links+1] = self.np_random.uniform(-min(speed, .5), min(speed, .5))
            kind = "easy" if easy else "frontier"
        return self.observation(), {**info, "reset_kind": kind, "stage": r.stage}

    def step(self, action):
        before = self.time
        obs, reward, terminated, truncated, info = super().step(action)
        if self.reward_config is not None:
            reward = shaped_reward(self.state, self.config, info["force"] / self.config.max_force,
                                   terminated, self.time - before, self.reward_config)
        return obs, reward, terminated, truncated, info
