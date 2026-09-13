"""Uniform rods connected by ideal revolute joints in a vertical plane.

Angles are absolute, measured from downward; positive tilts right.
Coordinates: q = [x, theta_1, ..., theta_n]. Units: kg, m, s, radians.
This lesson implements parameters and the mass matrix, not time evolution.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarChain:
    """Physical parameters; one length and mass per rod, starting at the cart."""

    cart_mass: float
    lengths: tuple[float, ...]
    masses: tuple[float, ...]
    gravity: float = 9.81

    def __post_init__(self):
        # Copy inputs into immutable tuples so lists also work safely.
        object.__setattr__(self, "lengths", tuple(self.lengths))
        object.__setattr__(self, "masses", tuple(self.masses))
        if not self.lengths or len(self.lengths) != len(self.masses):
            raise ValueError("Provide the same nonzero number of lengths and masses.")
        positive = np.asarray((self.cart_mass, *self.lengths, *self.masses))
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0):
            raise ValueError("Cart mass, rod masses, and lengths must be finite and positive.")
        if not np.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("Gravity must be finite and nonnegative.")

    @property
    def n_links(self):
        return len(self.lengths)

    def mass_matrix(self, angles):
        """Return M(q) in M(q) q_ddot = b(q, q_dot, force).

        Row/column 0 belongs to the cart; rod i uses row/column i + 1.
        Cart position is absent because horizontal translation changes no energy.
        """
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (self.n_links,) or not np.all(np.isfinite(angles)):
            raise ValueError("Provide one finite angle per rod.")

        lengths = np.asarray(self.lengths)
        masses = np.asarray(self.masses)
        # s_i: mass of all rods beyond rod i, excluding rod i itself.
        downstream = np.array([masses[i + 1:].sum() for i in range(self.n_links)])
        a = lengths * (masses / 2 + downstream)
        d = lengths**2 * (masses / 3 + downstream)

        matrix = np.zeros((self.n_links + 1, self.n_links + 1))
        matrix[0, 0] = self.cart_mass + masses.sum()

        for i in range(self.n_links):
            matrix[0, i + 1] = a[i] * np.cos(angles[i])
            matrix[i + 1, 0] = matrix[0, i + 1]
            matrix[i + 1, i + 1] = d[i]

            for j in range(i + 1, self.n_links):
                # j is farther from the cart, so c_ij = l_i * a_j.
                coupling = lengths[i] * a[j] * np.cos(angles[i] - angles[j])
                matrix[i + 1, j + 1] = coupling
                matrix[j + 1, i + 1] = coupling

        return matrix
