import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from cart_pendulum.render_best import best_replay


class BestReplayTests(unittest.TestCase):
    def fixture(self, root):
        trial = root / "trial_001"
        trial.mkdir()
        (trial / "config.json").write_text(json.dumps({"environment": {"n_links": 2}, "network": {"hidden_sizes": [32]}}))
        data = np.zeros((2, 8))
        data[1, 0] = .02
        np.savetxt(trial / "validation_trajectory.csv", data, delimiter=",", header="header", comments="")
        np.savetxt(root / "holdout_trajectory.csv", data, delimiter=",", header="header", comments="")
        summary = {"best": {"trial": "trial_001", "validation": {"episodes": [{"seed": 10000, "duration": .02}]}},
                   "holdout": {"episodes": [{"seed": 20000, "duration": .02}]}}
        (root / "summary.json").write_text(json.dumps(summary))
        return summary

    def test_selected_winner_uses_held_out_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            record, path = best_replay(root)
            self.assertEqual(record["title"], "Best: trial_001")
            self.assertEqual(record["episode"]["seed"], 20000)
            self.assertEqual(path.name, "holdout_trajectory.csv")

    def test_partial_run_uses_winners_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "holdout_trajectory.csv").unlink()
            record, path = best_replay(root)
            self.assertEqual(record["episode"]["seed"], 10000)
            self.assertEqual(path.parent.name, "trial_001")

    def test_missing_winner_or_mismatched_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self.fixture(root)
            summary["holdout"]["episodes"][0]["duration"] = 1
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                best_replay(root)
            (root / "summary.json").write_text('{"best": null}')
            with self.assertRaises(ValueError):
                best_replay(root)
