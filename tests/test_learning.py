from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from cart_pendulum.environment import CartPendulumEnv, EnvConfig
from cart_pendulum.experiments import ApiBudget, propose_openai
from cart_pendulum.learning import NetworkConfig, evaluate, train


class LearningTests(unittest.TestCase):
    def test_reject_unbounded_architecture(self):
        for kwargs in ({"hidden_sizes": [1000000]}, {"activation": "exec"}, {"learning_rate": float("nan")}, {"n_epochs": True}):
            with self.assertRaises(ValueError):
                NetworkConfig(**kwargs)

    def test_budget_persists_and_stops_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ApiBudget(path, .2).reserve()
            with self.assertRaises(RuntimeError):
                ApiBudget(path, 20).reserve()
            self.assertEqual(json.loads(path.read_text())["calls"], 1)

    def test_mock_api_proposal_and_usage_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(asdict(NetworkConfig()), rationale="Try a compact network.")
            response = SimpleNamespace(id="test", model="gpt-5-mini", status="completed",
                output_text=json.dumps(config), usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 50}))
            with patch("openai.OpenAI") as client:
                client.return_value.responses.create.return_value = response
                network, _ = propose_openai([], EnvConfig(), 512, ApiBudget(Path(tmp)/"budget.json", 1), Path(tmp)/"audit.json")
                self.assertEqual(network.hidden_sizes, (64, 64))
                self.assertEqual(client.call_args.kwargs["max_retries"], 0)
                self.assertEqual(client.return_value.responses.create.call_args.kwargs["max_output_tokens"], 4096)
                self.assertTrue((Path(tmp)/"audit.json").exists())

    def test_real_training_save_reload_and_deterministic_evaluation(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as tmp:
            config = EnvConfig(duration=.2)
            policy = train(config, NetworkConfig(hidden_sizes=(16,), n_epochs=1), 512, 7, Path(tmp)/"trial")
            loaded = PPO.load(Path(tmp)/"trial/model.zip", device="cpu")
            obs, _ = CartPendulumEnv(config).reset(seed=123)
            np.testing.assert_array_equal(policy.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
            self.assertEqual(evaluate(policy, config, [10000]), evaluate(loaded, config, [10000]))
            self.assertEqual(policy.num_timesteps, 512)
            self.assertGreater(policy._n_updates, 0)

    def test_incomplete_api_response_stops_and_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp)/"budget.json"
            with patch("openai.OpenAI") as client:
                client.return_value.responses.create.return_value = SimpleNamespace(
                    id="incomplete-test", model="gpt-5-mini", status="incomplete", output_text="", usage=None)
                with self.assertRaises(RuntimeError):
                    propose_openai([], EnvConfig(), 512, ApiBudget(ledger, 1), Path(tmp)/"audit.json")
                self.assertEqual(json.loads(ledger.read_text())["reserved_usd"], .2)
                self.assertEqual(client.return_value.responses.create.call_count, 1)
