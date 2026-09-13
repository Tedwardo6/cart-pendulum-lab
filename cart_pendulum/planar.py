"""Uniform rods connected by ideal revolute joints in a vertical plane.

Angles are absolute, measured from downward; positive tilts right.
Coordinates: q = [x, theta_1, ..., theta_n]. Units: kg, m, s, radians.
Dynamics return a state derivative; integration and rendering live separately.
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

    def _split_state(self, state):
        state = np.asarray(state, dtype=float)
        size = self.n_links + 1
        if state.shape != (2 * size,) or not np.all(np.isfinite(state)):
            raise ValueError(f"State must contain {2 * size} finite positions and velocities.")
        return state[:size], state[size:]

    def derivative(self, state, force=0.0):
        """Return [q_dot, q_ddot] for the current state and horizontal force."""
        q, velocity = self._split_state(state)
        if not np.isfinite(force):
            raise ValueError("Force must be finite.")
        angles, omega = q[1:], velocity[1:]
        lengths, masses = np.asarray(self.lengths), np.asarray(self.masses)
        downstream = np.array([masses[i + 1:].sum() for i in range(self.n_links)])
        a = lengths * (masses / 2 + downstream)

        rhs = np.zeros(self.n_links + 1)
        rhs[0] = force + np.sum(a * np.sin(angles) * omega**2)
        rhs[1:] = -self.gravity * a * np.sin(angles)
        for i in range(self.n_links):
            for j in range(i + 1, self.n_links):
                coupling = lengths[i] * a[j]
                sine = np.sin(angles[i] - angles[j])
                rhs[i + 1] -= coupling * sine * omega[j]**2
                rhs[j + 1] += coupling * sine * omega[i]**2

        acceleration = np.linalg.solve(self.mass_matrix(angles), rhs)
        return np.concatenate((velocity, acceleration))

    def joint_positions(self, q):
        """Return cart pivot and rod endpoints as an (n + 1, 2) array."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_links + 1,) or not np.all(np.isfinite(q)):
            raise ValueError("Provide cart position followed by one angle per rod.")
        lengths = np.asarray(self.lengths)
        offsets = np.column_stack((lengths * np.sin(q[1:]), -lengths * np.cos(q[1:])))
        points = np.zeros((self.n_links + 1, 2))
        points[0] = (q[0], 0.0)
        points[1:] = points[0] + np.cumsum(offsets, axis=0)
        return points

    def energy(self, state):
        """Total kinetic plus gravitational potential energy, in joules."""
        q, velocity = self._split_state(state)
        points = self.joint_positions(q)
        center_heights = (points[:-1, 1] + points[1:, 1]) / 2
        kinetic = 0.5 * velocity @ self.mass_matrix(q[1:]) @ velocity
        potential = self.gravity * np.dot(self.masses, center_heights)
        return float(kinetic + potential)
