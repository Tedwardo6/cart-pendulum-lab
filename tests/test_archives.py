from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from cart_pendulum.archives import retain_architecture, backfill
from cart_pendulum.environment import EnvConfig
from cart_pendulum.learning import NetworkConfig, write_json


class ArchitectureArchiveTests(unittest.TestCase):
    def make_trial(self, root, number, duration, *, network=None, reward=1):
        name = f"trial_{number:03}"
        trial = root / name
        trial.mkdir()
        config = {"environment": asdict(EnvConfig()), "network": asdict(network or NetworkConfig()), "seed": 7}
        metrics = {"mean_duration": duration, "mean_return": reward, "episodes": [{"seed": 10000}]}
        write_json(trial / "config.json", config)
        write_json(trial / "validation.json", metrics)
        (trial / "model.zip").write_bytes(name.encode())
        (trial / "validation_trajectory.csv").write_text(name)
        return {"trial": name, "network": config["network"], "validation": metrics}

    def test_best_for_each_structure_and_group_training_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_trial(root, 0, 2)
            folder = retain_architecture(root, first)
            weaker = self.make_trial(root, 1, 1, network=NetworkConfig(learning_rate=.001))
            self.assertEqual(retain_architecture(root, weaker), folder)
            self.assertEqual((folder / "best_model.zip").read_bytes(), b"trial_000")
            stronger = self.make_trial(root, 2, 3)
            retain_architecture(root, stronger)
            self.assertEqual((folder / "best_model.zip").read_bytes(), b"trial_002")
            self.assertEqual((folder / "validation_trajectory.csv").read_text(), "trial_002")
            other = self.make_trial(root, 3, .5, network=NetworkConfig(hidden_sizes=(128, 64)))
            other_folder = retain_architecture(root, other)
            self.assertNotEqual(other_folder, folder)
            self.assertTrue((other_folder / "best_model.zip").exists())
            relu = self.make_trial(root, 4, 1, network=NetworkConfig(activation="relu"))
            self.assertNotEqual(retain_architecture(root, relu), folder)
            self.assertTrue((root / "trial_000" / "model.zip").exists())

    def test_ties_and_backfill_are_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [self.make_trial(root, i, 2) for i in range(2)]
            write_json(root / "history.json", records)
            backfill(root)
            backfill(root)
            folder = root / "architectures/mlp-tanh-64x64"
            self.assertEqual((folder / "best_model.zip").read_bytes(), b"trial_000")
            self.assertEqual(len(json.loads((folder / "trials.json").read_text())), 2)

    def test_incomparable_or_missing_models_do_not_replace_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = retain_architecture(root, self.make_trial(root, 0, 2))
            bad = self.make_trial(root, 1, 5)
            path = root / "trial_001/validation.json"
            metrics = json.loads(path.read_text())
            metrics["episodes"][0]["seed"] = 20000
            write_json(path, metrics)
            with self.assertRaises(ValueError):
                retain_architecture(root, bad)
            (root / "trial_001/model.zip").unlink()
            with self.assertRaises(FileNotFoundError):
                retain_architecture(root, bad)
            self.assertEqual((folder / "best_model.zip").read_bytes(), b"trial_000")
