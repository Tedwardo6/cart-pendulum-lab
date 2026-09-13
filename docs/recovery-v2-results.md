# Recovery improvements: local checks, September 13, 2026

Six local training blocks used 65,536 additional environment steps across separate branches. No OpenAI API calls were made. All checkpoints and evaluations are retained under `runs/`; generated runs are not committed to Git.

| Run folder | New steps across branches | Result |
|---|---:|---|
| `recovery-v2-local-check-20260913` | 2 × 8,192 | At 3°, recovery improved from 4/8 to 6/8; easy retention stayed 8/8. One failed branch was rejected. |
| `recovery-v2-continuation-check-20260913` | 2 × 16,384 | Both attempts at 5° lost easy retention and were rejected. |
| `recovery-v2-guard-check-20260913` | 2 × 8,192 | With the update guard enabled, 5° recovery improved from 2/8 to 3/8 with 8/8 easy retention. One branch was rejected. |

Every run still had **0% hanging-start swing-up success**, including the final held-out evaluation. These are early recovery improvements, not a solved swing-up controller. Runs used different training seeds and block lengths, so they do not isolate the effect of the update guard or any individual change.

A separate 16-episode audit at 3°, using seeds 50000–50015 that were not used for curriculum selection, gave 11/16 successes for the original balance controller and 13/16 for the first run's accepted controller. That is encouraging but a small sample. The complete audit is in `runs/recovery-v2-local-check-20260913/independent_recovery_audit.json`; it is not fed into API memory.

The new system preserves a parent when a candidate loses recovery performance, saves a champion for each visited difficulty, adds cart diagnostics and outward-motion penalties near the track boundary, and freezes training rewards by default. New stages increase angles and velocity separately. A PPO target-KL threshold of 0.01 limits individual update cycles; it cannot prevent all forgetting.

To continue the latest accepted controller, run from the repository root with the virtual environment activated:

```sh
python -m cart_pendulum.curriculum \
  runs/recovery-v2-guard-check-20260913 \
  --proposer local --blocks 8 --max-steps 8192 --max-minutes 30 \
  --seed 801 \
  --output "runs/double-recovery-$(date +%Y%m%d-%H%M%S)"
```

This restores the accepted controller and active 5° stage, not either rejected branch. New training is not guaranteed to improve it. See [the curriculum guide](curriculum.md) for API mode, saved files, and scoring rules.
