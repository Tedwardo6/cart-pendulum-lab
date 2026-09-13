import unittest

import numpy as np

from cart_pendulum import PlanarChain
from cart_pendulum.integration import simulate


class DynamicsTests(unittest.TestCase):
    def test_hanging_equilibrium(self):
        for n in range(1, 5):
            chain = PlanarChain(1.2, (0.7,) * n, (0.5,) * n)
            np.testing.assert_allclose(chain.derivative(np.zeros(2 * (n + 1))), 0)

    def test_single_rod_analytic_acceleration(self):
        chain = PlanarChain(1.5, (0.8,), (0.6,))
        theta, omega, force = .7, -.4, 1.1
        # Eliminate angular acceleration from the two one-rod equations.
        m, length, g = .6, .8, chain.gravity
        xdd = (force + m * length / 2 * np.sin(theta) * omega**2
               + 3 * m * g / 4 * np.sin(theta) * np.cos(theta)) / (
                   chain.cart_mass + m - 3 * m / 4 * np.cos(theta)**2)
        tdd = -3 / (2 * length) * (g * np.sin(theta) + xdd * np.cos(theta))
        np.testing.assert_allclose(chain.derivative([0, theta, .3, omega], force), [.3, omega, xdd, tdd])

    def test_energy_rate_equals_motor_power(self):
        rng = np.random.default_rng(13)
        for n in range(1, 5):
            chain = PlanarChain(1.3, tuple(rng.uniform(.5, 1.2, n)), tuple(rng.uniform(.3, 1, n)))
            for _ in range(8):
                state = rng.normal(scale=.5, size=2 * (n + 1))
                force = .8
                rate = chain.derivative(state, force)
                eps = 1e-6
                energy_rate = (chain.energy(state + eps * rate) - chain.energy(state - eps * rate)) / (2 * eps)
                self.assertAlmostEqual(energy_rate, force * state[n + 1], delta=2e-7)

    def test_free_motion_conserves_energy_and_horizontal_momentum(self):
        for n in range(1, 5):
            chain = PlanarChain(1.5, (2 / n,) * n, (.6,) * n)
            state = np.concatenate(([0.0], np.linspace(.6, -.3, n), np.zeros(n + 1)))
            _, states = simulate(chain.derivative, state, duration=1, dt=.002)
            energies = [chain.energy(s) for s in states[::20]]
            momenta = [chain.mass_matrix(s[1:n+1])[0] @ s[n+1:] for s in states[::20]]
            np.testing.assert_allclose(energies, energies[0], rtol=0, atol=1e-6)
            np.testing.assert_allclose(momenta, 0, rtol=0, atol=1e-6)
