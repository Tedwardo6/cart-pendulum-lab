from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from cart_pendulum.environment import CartPendulumEnv, EnvConfig
from cart_pendulum.experiments import ApiBudget, propose_openai, run_experiments
from cart_pendulum.learning import NetworkConfig, score, write_json
from cart_pendulum.memory import ExperimentMemory


class SwingupTests(unittest.TestCase):
    def test_hanging_reset_and_rotations_allowed_one_to_four_rods(self):
        for n in (1, 2, 3, 4):
            env = CartPendulumEnv(EnvConfig(task="swingup", n_links=n))
            obs, _ = env.reset(seed=4)
            np.testing.assert_array_equal(obs, env.reset(seed=4)[0])
            self.assertTrue(np.all(np.abs(env.state[1:n+1]) <= .03))
            self.assertLess(env.chain.joint_positions(env.state[:n+1])[-1, 1], -1.99)
            env.state[:] = 0
            env.state[1:n+1] = 2 * np.pi
            _, reward, terminated, truncated, info = env.step([0])
            self.assertFalse(terminated or truncated)
            self.assertAlmostEqual(reward, 0)
            self.assertFalse(info["success"])

    def test_hanging_at_horizon_is_not_success_and_upright_reward_is_higher(self):
        config = EnvConfig(task="swingup", duration=.04, hold_seconds=.04)
        rewards = []
        for angle in (0., np.pi):
            env = CartPendulumEnv(config)
            env.reset(seed=0)
            env.state[:] = 0
            env.state[1:3] = angle
            env.step([0])
            _, reward, terminated, truncated, info = env.step([0])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["success"], angle == np.pi)
            rewards.append(reward)
        self.assertGreater(rewards[1], rewards[0])

    def test_hold_must_be_continuous_and_slow(self):
        env = CartPendulumEnv(EnvConfig(task="swingup", duration=.08, hold_seconds=.04))
        env.reset(seed=0)
        env.state[:] = 0
        env.state[1:3] = np.pi
        self.assertAlmostEqual(env.step([0])[-1]["final_hold"], .02)
        env.state[1:3] = 0
        self.assertEqual(env.step([0])[-1]["final_hold"], 0)
        env.state[:] = 0
        env.state[1:3] = np.pi
        env.state[4:] = 2.
        self.assertEqual(env.step([0])[-1]["final_hold"], 0)
        env.state[:] = 0
        env.state[1:3] = np.pi
        self.assertFalse(env.step([0])[-1]["success"])

    def test_track_limit_still_terminates_and_invalid_task_rejected(self):
        env = CartPendulumEnv(EnvConfig(task="swingup"))
        env.reset(seed=0)
        env.state[0] = 2.5
        self.assertEqual(env.step([0])[-1]["end_reason"], "track_limit")
        for kwargs in ({"task": "unknown"}, {"task": "swingup", "duration": .02}, {"hold_seconds": 0}):
            with self.assertRaises(ValueError):
                EnvConfig(**kwargs)

    def test_selection_does_not_reward_merely_lasting_longer(self):
        idle = dict(task="swingup", success_rate=0, mean_final_hold=0, mean_return=0, mean_duration=12)
        moving = dict(idle, mean_return=50, mean_duration=8)
        settled = dict(idle, success_rate=1, mean_final_hold=2, mean_return=20)
        self.assertGreater(score(moving), score(idle))
        self.assertGreater(score(settled), score(moving))

    def test_legacy_balance_memory_still_loads_but_not_for_swingup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = asdict(EnvConfig())
            for key in ("task", "hold_seconds", "settle_angular_speed", "settle_cart_speed"):
                config.pop(key)
            path = Path(tmp)/"memory.json"
            write_json(path, {"version": 1, "records": [{"task_version": "planar-balance-v1", "environment": config}]})
            memory = ExperimentMemory(path)
            self.assertEqual(len(memory.records(EnvConfig())), 1)
            self.assertEqual(memory.records(EnvConfig(task="swingup")), [])

    def test_api_prompt_explains_swingup_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = SimpleNamespace(
                id="mock", model="gpt-5-mini", status="completed", usage=None,
                output_text=json.dumps(dict(asdict(NetworkConfig()), rationale="Try swing-up.")))
            propose_openai([], EnvConfig(task="swingup"), 512, ApiBudget(Path(tmp)/"budget.json", 1), Path(tmp)/"api.json")
            instructions = client.return_value.responses.create.call_args.kwargs["instructions"]
            self.assertIn("hanging downward", instructions)
            self.assertIn("success_rate first", instructions)
            self.assertNotIn("survival duration first", instructions)

    def test_real_swingup_run_records_task_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = EnvConfig(task="swingup", duration=.1, hold_seconds=.04)
            result = run_experiments(root/"run", config, trials=1, steps=512,
                                     validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(result["completed_trials"], 1)
            record = ExperimentMemory(root/"memory.json").records(config)[0]
            self.assertEqual(record["validation"]["task"], "swingup")
            self.assertIn("mean_final_hold", record["validation"])
            self.assertTrue((root/"run/best_model.zip").is_file())
