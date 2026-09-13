# Lesson 3: a controller interacts with the simulator

## Follow one decision through the code

1. `CartPendulumEnv.reset(seed=...)` creates a small disturbance near upright. It returns an observation, not the future trajectory.
2. PPO's actor network converts that observation into a force action. During training it samples a Gaussian distribution to explore; evaluation uses its deterministic mean. Actions are clipped to [-1, 1] before scaling into newtons. This differs from the earlier illustrative tanh-output sketch: tanh can be the hidden activation, but SB3 PPO's action output is a clipped Gaussian.
3. `env.step(action)` holds that force over ten RK4 physics steps. The force is constant for 0.02 simulated seconds, while each 0.002-second physics step recalculates acceleration four times.
4. The environment returns a new observation, reward, and ending flags. It checks angle and track limits inside every physics substep.
5. PPO collects 512 decisions, then adjusts network weights using that batch. Training repeats until the configured decision budget or a stop condition is reached.

## Observation and action

The observation is `[x/2.4, x_dot/5, sin(theta_1), cos(theta_1), omega_1/10, ...]`. The exact track normalization uses the environment's track limit. Sine and cosine handle angle wraparound. Velocity divisors are fixed scales, not physical velocity limits. There are 8 inputs for two rods, 11 for three, and 14 for four. Each task gets a separately trained model because dimensions differ.

The action is a one-element array in [-1,1]. It scales to a maximum 20 N in either direction. The environment clips out-of-range finite values and rejects malformed/nonfinite actions. Force timing is quantized to 20 ms; holding the same action extends its duration.

## Reward and endings

The task is balancing from near upright, not swinging up. Angles start within 0.03 radians (about 1.7 degrees) of upright; cart position starts within 2 cm of center; velocities start at zero. Uniform rods share a total 2 m length. Rod mass is 0.6 kg each; cart mass is 1.5 kg.

Reward gives credit for survival and subtracts scaled penalties for angle error, cart displacement, velocity and force. This implementation uses squared wrapped angle error, which is useful near upright, rather than the earlier illustrative `1 + cos(theta)` cost. Episode success means surviving all 12 seconds within limits. Failure occurs when any angle exceeds 0.35 radians from upright, the cart exceeds 2.4 m, or integration fails. Angle errors are measured from pi, since our angles are measured from downward.

`terminated=True` means a physical/task failure. `truncated=True` means the 12-second time limit was reached. Keeping these different lets PPO bootstrap its value estimate correctly at time limits.

## The two networks in PPO

The actor chooses actions. The critic predicts expected future reward and helps the actor learn. SB3 builds separate actor and critic MLPs with the chosen hidden sizes. PPO adjusts their weights using its clipped policy objective. The experiment manager chooses architecture and hyperparameters; it does not calculate those weight updates itself.

## The outer loop

`experiments.py` proposes a configuration, trains a fresh PPO model, evaluates it, and keeps the highest-ranked trained candidate. It ranks mean survival time first and return second. A fixed set of validation seeds is reused to compare candidates. Separate holdout seeds are evaluated only for the selected model, after search finishes. Do not use holdout results repeatedly to tune this same search and still call them unseen.

Local mode makes reproducible mutations of the best previous configuration. OpenAI mode sends the task definition and previous validation summaries to the API, which returns structured configuration. It can select layer sizes, depth, activation, learning rate, discount factor, entropy coefficient and PPO epochs. It does not execute generated Python or let the model alter physics, rewards, seed sets, or budgets. New layer types or new learning algorithms require explicit implementation later.

Short runs verify the machinery, not robust balance. A fixed training seed and a few evaluation episodes are only a first benchmark; compare multiple training seeds and broader disturbances before claiming reliability.

## Reading order

1. `environment.py`: reset, observation, step, reward.
2. `learning.py`: network configuration, PPO training, evaluation.
3. `experiments.py`: candidate selection, iteration, API accounting.
4. `evaluate.py`: reload a saved network and test it without retraining.

References: [SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html), [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
