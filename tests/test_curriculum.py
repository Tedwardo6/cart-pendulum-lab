import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO

from cart_pendulum.curriculum import run_curriculum
from cart_pendulum.curriculum_proposals import allowed_stages, propose_local, propose_openai, validate_proposal
from cart_pendulum.environment import EnvConfig
from cart_pendulum.experiments import ApiBudget
from cart_pendulum.learning import NetworkConfig, evaluate, reset_value_network, score, train
from cart_pendulum.memory import ExperimentMemory, proposal_evidence
from cart_pendulum.render_best import best_replay
from cart_pendulum.training_tasks import CurriculumEnv, EVALUATION_VERSION, ResetConfig, RewardConfig, common_score, shaped_reward


class CurriculumTests(unittest.TestCase):
    def test_reward_joint_uprightness_and_near_top_speed(self):
        config = EnvConfig(task="swingup")
        state = np.zeros(6)
        def reward(s):
            return shaped_reward(s, config, 0, False, .02, RewardConfig())
        self.assertAlmostEqual(reward(state), 0)
        state[1] = np.pi
        one = reward(state)
        state[2] = np.pi
        both = reward(state)
        self.assertGreater(both, 4 * one)
        state[4:] = 3
        self.assertLess(reward(state), both)
        state[1:3] = 0
        self.assertAlmostEqual(reward(state), 0)  # no speed penalty down below
        self.assertLessEqual(common_score(state, config), 1)
        with self.assertRaises(ValueError):
            RewardConfig(catch=100)

    def test_resets_seeded_mixed_and_frontier_covers_boundary(self):
        config = EnvConfig(task="swingup")
        for n in (1, 2, 3, 4):
            env = CurriculumEnv(replace(config, n_links=n), ResetConfig(stage=2))
            kinds = set()
            for seed in range(50):
                obs, info = env.reset(seed=seed)
                np.testing.assert_array_equal(obs, env.reset(seed=seed)[0])
                kinds.add(info["reset_kind"])
            self.assertEqual(kinds, {"easy", "frontier", "hanging"})
        env = CurriculumEnv(config, ResetConfig(stage=3, frontier_only=True))
        env.reset(seed=1)
        self.assertGreaterEqual(np.max(np.abs(env.angle_errors())), np.deg2rad(45))
        self.assertLessEqual(np.max(np.abs(env.angle_errors())), np.deg2rad(60))

    def test_curriculum_does_not_change_fixed_evaluation_or_reward_ranking(self):
        config = EnvConfig(task="swingup", duration=.04, hold_seconds=.04)
        first = evaluate(None, config, [10000], evaluation_version=EVALUATION_VERSION)
        # Extreme training weights cannot influence a fresh fixed evaluator.
        env = CurriculumEnv(config, ResetConfig(), RewardConfig(together=3, catch=3))
        env.reset(seed=1)
        env.step([1])
        self.assertEqual(first, evaluate(None, config, [10000], evaluation_version=EVALUATION_VERSION))
        good = {**first, "mean_common_score": .5, "mean_return": -100}
        bad = {**first, "mean_common_score": .1, "mean_return": 100000}
        self.assertGreater(score(good), score(bad))
        self.assertEqual(first["success_rate"], 0)

    def test_stage_gates_and_proposal_bounds(self):
        self.assertEqual(allowed_stages(2, .5, 1), [1, 2])
        self.assertEqual(allowed_stages(2, 1, .5), [1, 2])
        self.assertEqual(allowed_stages(2, .75, 1), [1, 2, 3])
        net, spec, steps, _ = propose_local(NetworkConfig(), 0, [0, 1], 512)
        self.assertEqual(spec["reset"]["stage"], 1)
        value = {"stage": 6, "steps": 512, "reward": asdict(RewardConfig()), "learning_rate": .0003,
                 "entropy_coefficient": 0, "n_epochs": 1, "rationale": "Test"}
        with self.assertRaises(ValueError):
            validate_proposal(value, net, [0, 1], [512])
        value["stage"] = 0
        value["steps"] = 1024
        with self.assertRaises(ValueError):
            validate_proposal(value, net, [0], [512])
        value["steps"] = 512
        value["angle_limit"] = 100
        with self.assertRaises(ValueError):
            validate_proposal(value, net, [0], [512])

    def source(self, root):
        config = EnvConfig(duration=.1, hold_seconds=.04)
        network = NetworkConfig(hidden_sizes=(16,), n_epochs=1)
        folder = root / "source"
        model = train(config, network, 512, 7, folder)
        shutil.copyfile(folder / "model.zip", folder / "best_model.zip")
        shutil.copyfile(folder / "config.json", folder / "best_config.json")
        return config, network, model, folder

    def test_critic_reset_preserves_actor_and_actor_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, model, _ = self.source(Path(tmp))
            actor = [*model.policy.mlp_extractor.policy_net.parameters(), *model.policy.action_net.parameters(), model.policy.log_std]
            actor_weights = [p.detach().clone() for p in actor]
            moments = [copy.deepcopy(model.policy.optimizer.state[p]) for p in actor]
            value_before = model.policy.value_net.weight.detach().clone()
            reset_value_network(model)
            self.assertFalse(torch.equal(value_before, model.policy.value_net.weight))
            for p, before, state in zip(actor, actor_weights, moments):
                self.assertTrue(torch.equal(p, before))
                for key in state:
                    self.assertTrue(torch.equal(model.policy.optimizer.state[p][key], state[key]))
            for p in model.policy.value_net.parameters():
                self.assertNotIn(p, model.policy.optimizer.state)

    def test_api_receives_reward_history_but_no_holdout_or_paths(self):
        with tempfile.TemporaryDirectory() as tmp, patch("openai.OpenAI") as client:
            root = Path(tmp)
            value = {"stage": 0, "steps": 512, "reward": asdict(RewardConfig()), "learning_rate": .0001,
                     "entropy_coefficient": .01, "n_epochs": 2, "rationale": "Increase smooth catch feedback."}
            client.return_value.responses.create.return_value = SimpleNamespace(id="mock", model="gpt-5-mini",
                status="completed", usage=None, output_text=json.dumps(value))
            records = [{"status": "completed", "training_spec": {"reward": asdict(RewardConfig())},
                "validation": {"task": "swingup", "evaluation_version": EVALUATION_VERSION,
                    "success_rate": 0, "mean_final_hold": 0, "mean_common_score": .1},
                "holdout": "SECRET_HOLDOUT", "source_run": "/private/source"}]
            network, spec, steps, _ = propose_openai(EnvConfig(task="swingup"), NetworkConfig(), 0, [0], 512,
                records, {"stage": 0}, ApiBudget(root/"budget.json", 1), root/"api.json")
            request = client.return_value.responses.create.call_args.kwargs
            self.assertIn('"catch"', request["input"])
            self.assertNotIn("SECRET_HOLDOUT", request["input"])
            self.assertNotIn("/private/source", request["input"])
            self.assertEqual(network.learning_rate, .0001)
            self.assertEqual(steps, 512)
            self.assertTrue((root/"api_context.json").exists())

    def test_real_curriculum_block_and_memory_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _, _, source = self.source(root)
            result = run_curriculum(root/"run", source, blocks=1, max_steps=512,
                                    validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(result["completed_blocks"], 1)
            self.assertEqual(result["best"]["validation"]["evaluation_version"], EVALUATION_VERSION)
            memory = ExperimentMemory(root/"memory.json")
            self.assertEqual(memory.records(replace(config, task="swingup")), [])
            records = memory.records(replace(config, task="swingup"), EVALUATION_VERSION)
            self.assertEqual(len(records), 1)
            self.assertIsNotNone(records[0]["training_spec"])
            self.assertEqual(records[0]["added_steps"], 512)
            self.assertEqual(PPO.load(root/"run/latest_model.zip").num_timesteps, 1024)
            self.assertEqual(best_replay(root/"run")[0]["label"], "Held-out")
            self.assertTrue(json.loads((root/"run/trial_001/config.json").read_text())["critic_reset"])
            self.assertTrue((root/"run/accepted_model.zip").is_file())
            second = run_curriculum(root/"second", root/"run", blocks=1, max_steps=512,
                                    validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(second["completed_blocks"], 1)
            self.assertEqual(PPO.load(root/"second/latest_model.zip").num_timesteps, 1536)
            self.assertFalse(json.loads((root/"second/trial_001/config.json").read_text())["critic_reset"])

    def test_mock_api_loop_applies_controls_and_remembers_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, source = self.source(root)
            value = {"stage": 0, "steps": 512, "reward": asdict(RewardConfig(catch=1.2)),
                     "learning_rate": .0001, "entropy_coefficient": .012, "n_epochs": 2, "rationale": "Test catch reward."}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "mock-only"}), patch("openai.OpenAI") as client:
                client.return_value.responses.create.return_value = SimpleNamespace(id="mock", model="gpt-5-mini",
                    status="completed", usage=None, output_text=json.dumps(value))
                result = run_curriculum(root/"run", source, blocks=2, max_steps=512,
                    validation_episodes=1, proposer="openai", budget_ledger=root/"budget.json", memory_file=root/"memory.json")
                self.assertEqual(result["completed_blocks"], 2)
                calls = client.return_value.responses.create.call_args_list
                self.assertEqual(len(calls), 2)
                context = json.loads(calls[1].kwargs["input"])
                self.assertEqual(context["prior_trials"][0]["training_spec"]["reward"]["catch"], 1.2)
            latest = PPO.load(root/"run/latest_model.zip")
            self.assertEqual(latest.n_epochs, 2)
            self.assertEqual(latest.ent_coef, .012)
            self.assertEqual(latest.lr_schedule(1), .0001)
            self.assertEqual(latest.num_timesteps, 1536)
            self.assertAlmostEqual(json.loads((root/"budget.json").read_text())["reserved_usd"], .4)

    def test_failed_retention_keeps_training_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, source = self.source(root)
            probes = [{"stage": 0, "retention_success": 1., "frontier_success": 0.},
                      {"stage": 0, "retention_success": 0., "frontier_success": 0.}]
            with patch("cart_pendulum.curriculum.probe", side_effect=probes):
                result = run_curriculum(root/"run", source, blocks=1, max_steps=512,
                                        validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(result["completed_blocks"], 1)
            self.assertEqual(PPO.load(root/"run/accepted_model.zip").num_timesteps, 512)
            self.assertEqual(PPO.load(root/"run/latest_model.zip").num_timesteps, 1024)
            self.assertFalse(json.loads((root/"run/history.json").read_text())[-1]["accepted_for_training"])


if __name__ == "__main__":
    unittest.main()
