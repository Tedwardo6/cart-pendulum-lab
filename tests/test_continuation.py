import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from stable_baselines3 import PPO

from cart_pendulum.continue_training import continue_training
from cart_pendulum.environment import EnvConfig
from cart_pendulum.learning import NetworkConfig, train
from cart_pendulum.memory import ExperimentMemory
from cart_pendulum.render_best import best_replay


class ContinuationTests(unittest.TestCase):
    def source(self, root):
        config = EnvConfig(duration=.04)
        network = NetworkConfig(hidden_sizes=(16,), n_epochs=1)
        source = root / "source"
        policy = train(config, network, 512, 7, source)
        shutil.copyfile(source / "model.zip", source / "best_model.zip")
        shutil.copyfile(source / "config.json", source / "best_config.json")
        return source, config, network, policy

    def test_weights_optimizer_and_counters_are_restored_before_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, network, policy = self.source(root)
            weights = copy.deepcopy(policy.policy.state_dict())
            optimizer = copy.deepcopy(policy.policy.optimizer.state_dict())
            original_learn = PPO.learn
            captured = []

            def learn(model, *args, **kwargs):
                self.assertFalse(kwargs["reset_num_timesteps"])
                self.assertEqual(model.num_timesteps, 512)
                for name, tensor in model.policy.state_dict().items():
                    self.assertTrue(torch.equal(tensor, weights[name]))
                for key, state in model.policy.optimizer.state_dict()["state"].items():
                    for name, tensor in state.items():
                        self.assertTrue(torch.equal(tensor, optimizer["state"][key][name]))
                captured.append(True)
                return original_learn(model, *args, **kwargs)

            with patch.object(PPO, "learn", learn):
                resumed = train(config, network, 512, 101, root/"continued", checkpoint=source/"model.zip")
            self.assertTrue(captured)
            self.assertEqual(resumed.num_timesteps, 1024)
            self.assertGreater(resumed._n_updates, policy._n_updates)
            self.assertTrue(any(not torch.equal(v, weights[k]) for k, v in resumed.policy.state_dict().items()))
            stats = json.loads((root/"continued/training.json").read_text())
            self.assertEqual(stats["added_steps"], 512)
            self.assertEqual(stats["initial_steps"], 512)

    def test_blocks_accumulate_and_previous_best_survives_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, _, _ = self.source(root)
            original_hash = hashlib.sha256((source/"best_model.zip").read_bytes()).hexdigest()
            from cart_pendulum.learning import evaluate
            calls = []

            def regressing(*args, **kwargs):
                metrics = evaluate(*args, **kwargs)
                metrics["mean_return"] = 100 if not calls else -100
                calls.append(True)
                return metrics

            with patch("cart_pendulum.continue_training.evaluate", side_effect=regressing):
                result = continue_training(root/"continued", source, blocks=2, steps=512,
                                           validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(result["completed_blocks"], 2)
            self.assertEqual(result["best"]["trial"], "trial_000")
            self.assertEqual(PPO.load(root/"continued/latest_model.zip").num_timesteps, 1536)
            self.assertEqual(PPO.load(root/"continued/best_model.zip").num_timesteps, 512)
            self.assertEqual(hashlib.sha256((source/"best_model.zip").read_bytes()).hexdigest(), original_hash)
            records = ExperimentMemory(root/"memory.json").records(config)
            self.assertEqual([r["actual_steps"] for r in records], [1024, 1536])
            self.assertEqual([r["added_steps"] for r in records], [512, 512])
            self.assertTrue(all(r["training_mode"] == "continued" for r in records))
            self.assertEqual(best_replay(root/"continued")[0]["title"], "Best: trial_000")
            # A later invocation can start from latest, retaining the parent's best.
            next_run = continue_training(root/"next", root/"continued", checkpoint="latest",
                                         blocks=1, steps=512, validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(next_run["completed_blocks"], 1)
            self.assertEqual(PPO.load(root/"next/latest_model.zip").num_timesteps, 2048)

    def test_interrupt_preserves_latest_and_best_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, _ = self.source(root)
            with patch.object(PPO, "learn", side_effect=KeyboardInterrupt):
                result = continue_training(root/"continued", source, blocks=1, steps=512,
                                           validation_episodes=1, memory_file=root/"memory.json")
            self.assertEqual(result["stop_reason"], "keyboard_interrupt")
            self.assertEqual(result["completed_blocks"], 0)
            self.assertTrue((root/"continued/latest_model.zip").exists())
            self.assertTrue((root/"continued/best_model.zip").exists())
            self.assertEqual(best_replay(root/"continued")[0]["label"], "Validation")
            self.assertTrue(json.loads((root/"continued/trial_001/training.json").read_text())["interrupted"])


if __name__ == "__main__":
    unittest.main()
