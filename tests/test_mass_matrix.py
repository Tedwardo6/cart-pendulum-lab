"""Check the matrix against independently calculated rigid-body kinetic energy."""

import unittest

import numpy as np

from cart_pendulum import PlanarChain


class MassMatrixTests(unittest.TestCase):
    def test_single_uniform_rod(self):
        chain = PlanarChain(2.0, (3.0,), (4.0,))
        expected = [[6.0, 6.0 * np.cos(0.7)], [6.0 * np.cos(0.7), 12.0]]
        np.testing.assert_allclose(chain.mass_matrix([0.7]), expected)

    def test_matches_center_of_mass_and_rotation_energy(self):
        rng = np.random.default_rng(42)
        for n in range(1, 5):
            for _ in range(20):
                lengths = rng.uniform(0.3, 2, n)
                masses = rng.uniform(0.2, 3, n)
                angles = rng.uniform(-np.pi, np.pi, n)
                velocity = rng.normal(size=n + 1)
                chain = PlanarChain(1.7, lengths, masses)

                # Independently accumulate each rod's center velocity from geometry.
                pivot_velocity = np.array([velocity[0], 0.0])
                energy = 0.5 * chain.cart_mass * velocity[0]**2
                for length, mass, angle, omega in zip(lengths, masses, angles, velocity[1:]):
                    relative_tip_velocity = length * omega * np.array([np.cos(angle), np.sin(angle)])
                    center_velocity = pivot_velocity + 0.5 * relative_tip_velocity
                    inertia = mass * length**2 / 12
                    energy += 0.5 * mass * (center_velocity @ center_velocity)
                    energy += 0.5 * inertia * omega**2
                    pivot_velocity += relative_tip_velocity

                matrix = chain.mass_matrix(angles)
                np.testing.assert_allclose(0.5 * velocity @ matrix @ velocity, energy)
                np.testing.assert_allclose(matrix, matrix.T)
                self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0))

    def test_invalid_parameters_and_angles(self):
        for args in [(0, (1,), (1,)), (1, (), ()), (1, (1, 2), (1,)), (1, (np.nan,), (1,))]:
            with self.assertRaises(ValueError):
                PlanarChain(*args)
        chain = PlanarChain(1, (1,), (1,))
        for angles in [[], [0, 1], [np.nan]]:
            with self.assertRaises(ValueError):
                chain.mass_matrix(angles)


if __name__ == "__main__":
    unittest.main()
