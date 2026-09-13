# Cart Pendulum Lab

A Python learning project: derive the physics of a cart carrying a chain of pendulums, build a simulator, and explore control algorithms.

## Current lesson

Lesson 1 implements physical parameters and the planar mass matrix for any positive number of uniform rods, including double, triple, and quadruple pendulums. It does **not yet integrate motion or implement a controller**.

From the repository root, with Python 3.9 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m examples.lesson_01_mass_matrix
python -m unittest discover -s tests -v
```

Read [the first lesson](docs/lesson_01.md), then [the implementation](cart_pendulum/planar.py).

## Physics conventions

- SI units: kilograms, meters, seconds, radians; input force will be in newtons.
- A cart moves along one horizontal axis; uniform rigid rods move in a vertical plane.
- Rods are numbered outward from the cart. Angles are absolute, measured from downward, positive toward the right.
- Coordinates: `q = [x, theta_1, ..., theta_n]`.
- Planned state: `[q, q_dot]`, with `2 * (n + 1)` entries.
- Ideal joints, no friction, and no track end stops in the initial model.

## Roadmap

1. Parameters and mass matrix (implemented and checked against rigid-body kinetic energy).
2. Force and gravity terms, accelerations, and the state derivative.
3. Numerical integration, energy checks, and animation.
4. Control environment, constraints, and baseline controllers.
5. Automated controller code generation, fixed evaluations, and logged iterations with explicit run and cost limits.
6. Explore a spatial model after deciding joint types, rod inertia, and cart actuation.

## Future 3D work

The planar model lives in its own module. Future integration and controller interfaces should accept a state derivative function so a spatial model can supply different physics. Three-dimensional dynamics need a fresh choice of coordinates and constraints; adding one more angle to these formulas is not sufficient. A spherical joint, a hinge, and a twisting rigid body have different degrees of freedom. Whether the cart moves on a line or a plane also matters.

## Validation

Tests cover the analytic single-rod matrix and compare matrix-based energy with independently computed center-of-mass and rotational energy for 80 sampled configurations across one to four rods. These checks validate this lesson's mass matrix, not a complete simulator.
