# Run, evaluate, and iterate neural controllers

Use a terminal in the repository root. Python 3.12 is recommended for the complete training toolchain. The initial physics lessons also support Python 3.9.

## Install

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,agent,render]'
python -m unittest discover -s tests -v
```

On the original development machine, `.venv` is already installed. Just activate it. `requirements-tested.txt` records the exact development environment versions; install it first if reproducing that environment on a compatible platform.

## A local experiment with no API charges

```sh
python -m cart_pendulum.experiments --links 2 --proposer local \
  --trials 3 --steps 32768 --max-minutes 30 --output runs/double-001
```

Use `--links 3` or `--links 4` and a new output directory for the other tasks. Every candidate starts with fresh weights. The environment is fixed throughout a run. Training steps count controller decisions, not physics substeps, and must be a multiple of 512. These defaults are an initial experiment budget, not a promise of successful balancing.

For a fast end-to-end check, use `--trials 2 --steps 512 --evaluation-episodes 2`.

## OpenAI-guided configuration search

```sh
python -m cart_pendulum.experiments --links 2 --proposer openai \
  --ask-api-key --api-budget 20 --trials 3 --steps 32768 \
  --max-minutes 30 --output runs/double-openai-001
```

The terminal asks for the API key with hidden input. It stays in the process environment and is not written to a file. Alternatively set `OPENAI_API_KEY` in your local environment. Do not paste keys into chat or commit them. No API calls are made in local mode.

The initial proposer is `gpt-5-mini`, chosen for inexpensive structured configuration proposals. Published standard text prices checked on 2026-09-13 were $0.25 per million input tokens and $2 per million output tokens: [official model page](https://developers.openai.com/api/docs/models/gpt-5-mini). Availability and rates can change; recheck before future use. Only short text requests are sent; the network trains locally on CPU. The model never receives your API key as prompt content.

Each API request reserves $0.20 in `.api-budget.json` **before** it is sent. This is deliberately larger than the possible text charge at the recorded rates and bounded context/output size. Failed, refused and timed-out requests retain their reservations; automatic retries are disabled. The ledger persists across output directories when commands run from the same repository root. Do not delete it or choose a different ledger to reset an ongoing budget. Existing ledger limits cannot be raised by a later command. The ledger uses POSIX file locking (macOS/Linux).

Actual token usage and estimated cost appear in `api_*.json`; conservative reserved dollars differ from actual billed dollars. The initial three-trial command reserves at most $0.60. The maximum accepted per-ledger limit is $20. This controls this program at the documented prices, not unrelated account usage or future provider price changes. This integration was tested with mocked API responses; live account access still needs verification with your key.

## Memory across runs

Every new experiment automatically reads and updates `runs/experiment_memory.json` when you run commands from the repository root. Give each run a new output folder; keep the same memory file. OpenAI-guided runs use this history to inform their next proposals. Local runs contribute results too, although the local proposer only mutates candidates from its current run.

Memory retains each completed trial's architecture, training settings, rationale, training seed, requested and actual training steps, validation seeds and scores, and the location of its saved model. Failed trials record their configuration when available, rationale, failure stage and exception type. The file is updated after each evaluated trial, with atomic writes and a lock for concurrent writers on macOS/Linux.

The API receives a bounded selection of up to eight best/recent completed results and three recent failures. All records remain on disk, even when omitted from an individual request. Only records with identical environment settings and the same task version are eligible, so double, triple and quadruple pendulums have separate evidence. Different training budgets and evaluation seeds are included explicitly so the proposer can account for them. Historical rationales are hypotheses, not established conclusions.

This is persistent experiment context, not a change to the OpenAI model's weights. Each candidate still trains from fresh weights. Held-out test results are saved in the run folder but excluded from proposal memory to keep them separate from architecture selection.

Use `--memory-file /absolute/path/to/experiment_memory.json` if launching from another working directory or sharing history between checkouts. Back up both the memory file and the run folders: the memory is an index of evidence, not a replacement for model files. Import completed experiments created before this feature with:

```sh
python -m cart_pendulum.memory runs/double-001 runs/triple-001
```

Importing the same run twice does not duplicate it. Old runs gain memory entries; they cannot retroactively gain source snapshots or API context that were never recorded.

## Stop behavior

- Trial count is a hard cap: 1–20 candidates.
- The per-trial decision budget is fixed; the model cannot change it.
- The training time limit is cooperative: no new trial starts after the deadline, and training stops on its next callback. A pending API request, PPO update, validation, or final holdout can finish after the deadline.
- Create an empty file named `STOP` in the run's output directory to request a stop. The current model is saved and completed evaluations/results are preserved.
- Ctrl-C during training saves that trial's current model, then exits the search. Previously selected best models are preserved.
- Proposal/training errors stop the run rather than repeatedly spending money. Error files record the exception type without API request bodies.
- Use a new output directory for a new run. Automatic resume of interrupted searches is not implemented.

## Saved outputs

- `run.json`: unique run ID, timestamps, status, stop reason, settings, Python/package versions and Git revision.
- `source/`: a snapshot of the actual Python implementation and dependency specifications used for the run, including uncommitted code changes.
- `files.json`: file sizes and SHA-256 checksums, written on normal completion, cooperative stops and caught failures. A forced process kill can leave the manifest marked `running` without a final inventory; already written trials and memory survive.
- `memory_at_start.json`: the matching prior evidence available when the run began.
- `api_*_context.json`: the exact evidence sent for each API proposal; `api_*.json` records the returned text, response ID, model, token usage and cost estimate. The source snapshot records the proposal instructions and schema.
- `proposal_*.json` and `api_*_proposal.json`: proposed settings and rationale, with parsed API proposals kept separately.
- `error_*.json`: failure stage and exception type. API keys, environment variables and exception request bodies are not captured.
- `architectures/mlp-tanh-64x64/` (and one folder for every other tested structure): that architecture's `best_model.zip`, `best_config.json`, `best_result.json`, training details, validation trajectory and `trials.json`. Layer sizes in order and activation define a structure; different learning rates or training settings compete within that structure. Rankings use the same validation scores as overall selection. These archives are per experiment run, so different pendulum tasks are kept separate.
- `architectures/index.json`: all tested structures, their best trials and scores. Every completed trial remains in its original folder too. "Best" refers to the best evaluated trial checkpoint, not an unmeasured moment during training.
- `best_model.zip`: the best trained actor/critic and weights among completed candidates; reloadable by SB3.
- `best_config.json`: exact environment, architecture, seed and requested training steps.
- `summary.json`: selected model, zero-force baseline, validation and separate holdout results.
- `history.json`: candidate configurations, rationales and validation scores.
- `trial_*/model.zip`: every completed trial's model, plus a saved model if training was interrupted.
- `trial_*/episodes.csv.monitor.csv`: training episode returns, durations and timings.
- `holdout_trajectory.csv`: one held-out controlled trajectory with states and applied forces.

"Best" does not mean successful or even better than zero force. Always inspect the baseline and success rate. A 12-second success rate of zero means balancing has not been solved.

To organize an older run without retraining:

```sh
python -m cart_pendulum.archives runs/double-001
```

You can evaluate an architecture's saved winner by passing its folder to the existing evaluator, for example `python -m cart_pendulum.evaluate runs/double-001/architectures/mlp-tanh-64x64 --output runs/retest-64x64`.

## Reload and test, without training

```sh
python -m cart_pendulum.evaluate runs/double-001 \
  --episodes 8 --seed 30000 --output runs/double-001-retest
```

This produces numerical metrics and a CSV trajectory. No HTML, rendering, OpenAI connection, or ongoing API access is needed to run the trained network.
