from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cart_pendulum.environment import EnvConfig
from cart_pendulum.experiments import ApiBudget, propose_openai, run_experiments
from cart_pendulum.learning import NetworkConfig, write_json
from cart_pendulum.memory import ExperimentMemory, proposal_evidence


class MemoryTests(unittest.TestCase):
    def remember_trial(self, root, memory, n=2, duration=1):
        trial = root / 'trial_000'
        trial.mkdir(parents=True)
        write_json(trial/'config.json', {'environment': asdict(EnvConfig(n_links=n)),
            'network': asdict(NetworkConfig()), 'seed': 7, 'requested_steps': 512})
        write_json(trial/'training.json', {'actual_steps': 512})
        write_json(trial/'validation.json', {'mean_duration': duration, 'mean_return': 2,
            'success_rate': 0, 'episodes': [{'seed': 10000}]})
        write_json(root/'summary.json', {'holdout': {'secret_test_value': 'NEVER_IN_PROMPT'}})
        record = {'trial': 'trial_000', 'rationale': 'Test a compact network.'}
        memory.remember(root, record)
        return record

    def test_persistence_idempotence_task_filter_and_no_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'memory.json'
            memory = ExperimentMemory(path)
            record = self.remember_trial(root/'double', memory)
            memory.remember(root/'double', record)
            self.remember_trial(root/'triple', memory, n=3)
            reopened = ExperimentMemory(path)
            self.assertEqual(len(reopened.records(EnvConfig())), 1)
            self.assertEqual(len(reopened.records(EnvConfig(n_links=3))), 1)
            self.assertEqual(len(reopened.records(EnvConfig(max_force=10))), 0)
            self.assertNotIn('NEVER_IN_PROMPT', path.read_text())
            self.assertEqual(reopened.records(EnvConfig())[0]['actual_steps'], 512)

    def test_api_receives_prior_decision_and_validation_not_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = ExperimentMemory(root/'memory.json')
            self.remember_trial(root/'earlier', memory)
            reply = SimpleNamespace(id='test', model='gpt-5-mini', status='completed', usage=None,
                output_text=json.dumps(dict(asdict(NetworkConfig()), rationale='Use the earlier result.')))
            with patch('openai.OpenAI') as client:
                client.return_value.responses.create.return_value = reply
                propose_openai([], EnvConfig(), 1024, ApiBudget(root/'budget.json',1), root/'api.json',
                               prior_records=memory.records(EnvConfig()))
                prompt = client.return_value.responses.create.call_args.kwargs['input']
                self.assertIn('Test a compact network.', prompt)
                self.assertIn('512', prompt)
                self.assertNotIn('NEVER_IN_PROMPT', prompt)
                self.assertNotIn(str(root), prompt)
                self.assertTrue((root/'api_context.json').exists())

    def test_archive_failure_preserves_completed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = ExperimentMemory(root/'memory.json')
            self.remember_trial(root/'run', memory)
            memory.remember_failure(root/'run', 'trial_000', EnvConfig(), 'saving_results', 'OSError')
            record = memory.records(EnvConfig())[0]
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['validation']['mean_duration'], 1)
            self.assertEqual(record['post_evaluation_error']['error_type'], 'OSError')

    def test_bounded_context_keeps_old_best_and_latest(self):
        records = [{'id': str(i), 'network': {}, 'rationale': '', 'validation': {
            'mean_duration': 100 if i == 0 else i, 'mean_return': 0}} for i in range(30)]
        selected = proposal_evidence(records)
        self.assertLessEqual(len(selected), 8)
        self.assertIn('0', [r['id'] for r in selected])
        self.assertIn('29', [r['id'] for r in selected])

    def test_two_real_runs_share_memory_and_save_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_file = root/'memory.json'
            config = EnvConfig(duration=.02)
            for name in ('first','second'):
                result = run_experiments(root/name, config, trials=1, steps=512,
                    validation_episodes=1, memory_file=memory_file)
                self.assertEqual(result['completed_trials'], 1)
                self.assertEqual(json.loads((root/name/'run.json').read_text())['status'], 'completed')
                self.assertTrue((root/name/'source/cart_pendulum/environment.py').exists())
                self.assertTrue((root/name/'proposal_000.json').exists())
                self.assertTrue((root/name/'files.json').exists())
            prior = json.loads((root/'second/memory_at_start.json').read_text())['records']
            self.assertEqual(len(prior), 1)
            self.assertEqual(len(ExperimentMemory(memory_file).records(config)), 2)

    def test_failed_proposal_is_recorded_and_existing_run_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('cart_pendulum.experiments.propose_local', side_effect=ValueError('invalid')):
                result = run_experiments(root/'failed', EnvConfig(duration=.02), trials=1, steps=512,
                    validation_episodes=1, memory_file=root/'memory.json')
            self.assertEqual(result['stop_reason'], 'error:ValueError')
            self.assertEqual(ExperimentMemory(root/'memory.json').records(EnvConfig(duration=.02))[0]['phase'], 'proposal')
            manifest = root/'failed/run.json'
            before = manifest.read_bytes()
            with self.assertRaises(FileExistsError):
                run_experiments(root/'failed', EnvConfig(), trials=1, steps=512, memory_file=root/'memory.json')
            self.assertEqual(manifest.read_bytes(), before)
