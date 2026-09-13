# Learn swing-up from a balancing controller

This workflow retains an existing actor's learned behavior, introduces progressively harder starting states, and optionally lets the API propose training rewards and hyperparameters. It keeps the physical swing-up evaluation fixed. It does not guarantee successful swing-up.

## Start your first run

From the repository root with `.venv` activated, use your successful balance experiment as the source:

```sh
python -m cart_pendulum.curriculum \
  runs/double-balance-continued-20260913-124011 \
  --proposer local \
  --blocks 8 --max-steps 65536 --max-minutes 60 \
  --output "runs/double-curriculum-$(date +%Y%m%d-%H%M%S)"
```

Start with local mode: it makes no API calls. To use API proposals, replace `--proposer local` with `--proposer openai --ask-api-key --api-budget 20`. Paste the key only at the hidden prompt. Rewards are now fixed for the entire run by default, including in API mode. Add `--reward-mode adaptive` to explicitly enable API reward changes. Local mode uses learning rate at most 0.0001 and at most 10 update epochs to make continued training more conservative.

The source supplies the rod count, masses, lengths, force limit, observation conventions, architecture and starting weights. The environment switches to swing-up termination rules so tilted rods do not immediately end the episode. The source files are preserved.

Each block uses at most `--max-steps` new controller decisions. The API chooses from one-quarter, one-half or the full limit, rounded down to multiples of 512; the minimum is 512. Local mode uses the full limit. Eight blocks with this command can add at most 524,288 training steps across all branches. More steps do not guarantee better results. Rejected branches still consume training time.

The same shared budget ledger is used as other API experiments: $0.20 reserved per request, no automatic retries, one request per block, at most 20 blocks. Eight requests reserve at most $1.60; reservations are not actual billed cost. Changing training steps does not add API calls. Actual token usage and estimates are saved. The existing ledger's lower limit, if any, continues to apply.

## How the curriculum works

Training episodes are sampled from a fixed mixture:

- 20% near-upright, low-speed starts to retain balancing skill.
- 10% hanging-down starts to expose the actual swing-up task throughout training.
- 70% progressively harder recovery starts around upright.

The `recovery-v2` schedule starts at 3°, then 5°, 7.5°, 10°, 12.5° and 15°, all at zero initial velocity. It then introduces ±0.1 and ±0.25 rad/s at 15°. Later stages increase angle and speed separately: (20°, .25), (25°, .25), (30°, .25), (30°, .5), (45°, .5), (60°, .5), (60°, 1), (90°, 1), (120°, 1), (150°, 1), (180°, 1), (180°, 2). Cart velocity is limited to ±0.5 m/s at reset. Each rod's deviation and velocity is sampled separately. Even the final stage is a mixture; success is always evaluated separately from hanging down.

The program measures two probe sets using separate fixed seeds. The **retention probe** starts close to upright with zero initial speeds. The **recovery probe** uses the current stage's velocity range and requires at least one rod to start in the outer 25% of its angle range. Probe success uses the same final settled-hold requirement as swing-up, but with these easier initial conditions.

A stage may advance by one only when recovery success is at least 70% and retention is at least 80%. The API may stay at the current stage or retreat one stage. Local mode chooses the highest permitted stage. No proposal can skip the gate.

A candidate must retain at least 75% easy-start success **and improve recovery** to become the next parent. Recovery is ranked by success rate, then mean final hold, then fixed common score; ties keep the incumbent. When changing stages, the parent is first evaluated on the proposed stage, so the comparison uses identical starts. A per-stage champion is saved under `recovery/`, and revisiting that stage within the run restores its better checkpoint when needed. Rejected candidates are saved, but the next block starts from the protected parent. This measured check helps detect forgetting; it is not a guarantee of robustness beyond the tested seeds. The source itself must pass the 75% retention check before starting.

## The new training reward

Angles are still measured from downward. For each rod define `u = (1 - cos(theta))/2`: zero hanging and one upright. The reward combines:

- **Progress:** mean `u`, providing some partial credit during exploration.
- **Together:** product of every rod's `u`, emphasizing simultaneous uprightness.
- **Catch:** the product multiplied by `exp(-mean(omega²)/4 - cart_speed²)`, smoothly favoring low speed near upright.
- A speed penalty multiplied by joint uprightness, so it has little effect down below where momentum is needed.
- Small force and cart-displacement penalties, and a fixed penalty of 5 on failure.
- A braking penalty for outward cart velocity in the outer half of the track. It is zero while moving inward and near the center; this gives feedback before the cart exits.

In adaptive mode the API can choose these coefficients within bounds. New transfers use these defaults; existing reward weights are otherwise inherited. Legacy transfers raise centering to at least 0.1 and set braking to 0.1:

| Coefficient | Default | Allowed range |
|---|---:|---:|
| Progress | 0.25 | 0.05–0.5 |
| Together | 1 | 0.5–3 |
| Catch | 1 | 0.5–3 |
| Near-top speed penalty | 0.1 | 0–0.5 |
| Force penalty | 0.002 | 0–0.02 |
| Centering penalty | 0.1 | 0–0.2 |
| Outward braking penalty | 0.1 | 0–0.5 |

The exact per-step expression is implemented in `training_tasks.shaped_reward` and copied into each run's source snapshot. Coefficients are saved under `training_spec.reward`.

New curriculum blocks also set PPO’s `target_kl` to 0.01. This lets PPO stop update epochs early when its estimated policy change exceeds its threshold. It limits changes within each rollout, not cumulative drift over an entire block, so checkpoint rejection remains necessary. The setting is saved in each training specification and checkpoint.

The actor weights and its optimizer moments survive transfer. When entering this reward system, or changing its reward coefficients, the critic's value network and its optimizer moments are reset because its old value predictions refer to a different reward. Changing only curriculum difficulty preserves the critic. This is explicitly recorded as `critic_reset`.

The API may also choose learning rate, entropy coefficient and PPO update epochs within the existing bounds. Architecture and gamma stay fixed. It proposes structured configuration, not executable code. Requests use the [OpenAI structured-output interface](https://developers.openai.com/api/docs/guides/structured-outputs), followed by local validation of every control.

## Compare results fairly

Every block is tested on the same hanging-start validation seeds. The physical task and original success thresholds remain fixed: stay inside the track, finish the 12-second episode, and keep every rod within 0.35 radians of upright, every angular speed at most 1 rad/s, and cart speed at most 1 m/s for the final two seconds. Shorter test configurations may use different explicitly saved settings; the CLI inherits your source's settings.

Models are ranked by **success rate**, then **mean final settled hold**, then a **fixed common score**. The common score is the time integral of `0.25*together + 0.75*catch`, divided by the full episode duration, with the definitions above and fixed coefficients. It lies between 0 and 1. Early termination leaves the remaining time contributing zero. The API cannot edit it. Training return is never used to select a winner across reward formulas.

The `mean_return` field in evaluation files is still the fixed legacy evaluation reward, not the configurable training reward. Training returns appear in each trial's monitor CSV. The objective is versioned as `swingup-fixed-v2`; old swing-up return rankings are not mixed with the new score.

The best full-task checkpoint gets a final held-out evaluation after all blocks; held-out results never enter API prompts or curriculum gates. Retention and recovery successes are training diagnostics, not evidence that hanging-start swing-up is solved.

## Saved files and future runs

- `best_model.zip` / `best_config.json`: best checkpoint on the fixed hanging-start objective, even if it is the imported initial controller.
- `accepted_model.zip` / `accepted_config.json`: protected training parent that passed retention and the recovery comparison. This is the parent for future curriculum training.
- `latest_model.zip` / `latest_config.json`: latest attempted checkpoint, possibly rejected or interrupted.
- `trial_000/`: reevaluated source best; no new training.
- `trial_001/`, etc.: each trained block, its settings, model, evaluation trajectory, counters and full `probes.json`.
- `recovery/stage_*/`: best recovery model, configuration and probes for each visited stage in this run.
- `baseline_*.json`: parent probes on the exact stage used for each candidate comparison.
- `curriculum_state.json`: stage, accepted parent and last accepted probe results.
- `proposal_*.json`, `api_*.json`, `api_*_context.json`: proposed controls, rationale, response/usage, exact API input, instructions and schema.
- `history.json`, `summary.json`, `architectures/`: the evaluated candidates and their winners. Rejected training candidates remain available.

The shared memory file includes reward coefficients, reset settings, probe results, acceptance status, inherited/new training steps and parent checkpoint paths. API evidence is filtered by identical physical/task settings, evaluation version and curriculum version. This separates old stage numbers from the new schedule. Different training rewards and stages remain together within that objective so their effects can be compared; their settings are included explicitly. Old balancing and legacy swing-up histories remain intact and separate. Only a bounded best/recent subset is sent in each prompt.

To continue, pass a prior **curriculum folder** to the same command with a new output folder. It automatically loads `accepted_model.zip` and restores the active stage from `curriculum_state.json`. When importing an older curriculum, it retains the learned weights but reassesses difficulty from new stage 0; old stage numbers are not reinterpreted. Each new run initializes its recovery archive from the selected parent; earlier runs retain their other stage archives. Use this command rather than the older `continue_training` command, which does not manage curriculum rewards. A rejected or interrupted latest checkpoint is preserved for inspection, but is not automatically adopted as the next parent.

Each evaluation also records cart exit rate, peak displacement and speed, time near the track edge, and force saturation. Recovery diagnostics are printed after each block and included in API context. They do not change the fixed success definition.

Make a GIF with `python -m cart_pendulum.render_best runs/YOUR-CURRICULUM-FOLDER`. It shows the fixed hanging-start episode of the selected best controller. The existing evaluator also recognizes the new evaluation version.

Ctrl-C saves current training weights when possible. A `STOP` file in the output folder or the cooperative time limit stops further training; a pending API request, PPO update or evaluation may finish afterward. Use a new output directory on every invocation. Historical manifests and source runs are not overwritten.
