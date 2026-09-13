import unittest
import numpy as np
from cart_pendulum.environment import CartPendulumEnv, EnvConfig
from cart_pendulum.integration import rk4_step


class EnvironmentTests(unittest.TestCase):
    def test_seeded_reset_and_observation_order(self):
        for n in (2, 3, 4):
            env = CartPendulumEnv(EnvConfig(n_links=n))
            obs, _ = env.reset(seed=123)
            np.testing.assert_array_equal(obs, env.reset(seed=123)[0])
            self.assertEqual(obs.shape, (2 + 3*n,))
            self.assertTrue(env.observation_space.contains(obs))
            np.testing.assert_allclose(obs[2::3], np.sin(env.state[1:n+1]), atol=1e-7)
            np.testing.assert_allclose(obs[3::3], np.cos(env.state[1:n+1]), atol=1e-7)

    def test_action_held_through_all_physics_steps(self):
        env = CartPendulumEnv()
        env.reset(seed=2)
        expected = env.state.copy()
        force = .25 * env.config.max_force
        for _ in range(10):
            expected = rk4_step(lambda s: env.chain.derivative(s, force), expected, .002)
        _, _, terminated, truncated, info = env.step([.25])
        np.testing.assert_allclose(env.state, expected)
        self.assertAlmostEqual(info["time"], .02)
        self.assertEqual(info["force"], force)
        self.assertFalse(terminated or truncated)

    def test_clipping_invalid_actions_and_episode_end(self):
        env = CartPendulumEnv(EnvConfig(duration=.02))
        with self.assertRaises(RuntimeError):
            env.step([0])
        env.reset(seed=4)
        for action in ([np.nan], [1, 2], 1):
            with self.assertRaises(ValueError):
                env.step(action)
        env.state[:] = 0
        env.state[1:3] = np.pi
        _, _, term, trunc, _ = env.step([0])
        self.assertFalse(term)
        self.assertTrue(trunc)
        with self.assertRaises(RuntimeError):
            env.step([0])
        env.reset(seed=4)
        self.assertEqual(env.step([99])[-1]["force"], env.config.max_force)

    def test_failure_is_not_a_time_limit(self):
        env = CartPendulumEnv()
        env.reset(seed=4)
        env.state[1] = np.pi + .4
        _, _, term, trunc, info = env.step([0])
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["end_reason"], "angle_limit")

    def test_invalid_configuration(self):
        for kwargs in ({"n_links": 0}, {"max_force": 0}, {"physics_dt": .003}, {"duration": .011}):
            with self.assertRaises(ValueError):
                EnvConfig(**kwargs)
