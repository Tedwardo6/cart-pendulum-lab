# Cart Pendulum Lab

A Python learning project: derive the physics of a cart carrying a chain of pendulums, build a simulator, and explore control algorithms.

## Current lesson

The planar model now computes accelerations, integrates free motion, and renders animations. It supports any positive number of uniform rods; the examples and checks cover one through four. A controller is not implemented yet.

From the repository root, with Python 3.9 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m examples.lesson_01_mass_matrix
python -m unittest discover -s tests -v
```

To render a 12-second free-motion demonstration with playback controls:

```sh
python -m pip install -e '.[render]'
python -m examples.render_simulation --links 2 --output runs/double
```

Open `runs/double/simulation.html` for play/pause and frame controls, or `simulation.gif` for a looping preview. Use `--links 3` or `--links 4` for additional rods. The cart moves in reaction to the swinging rods even with zero motor force. Physics uses 2 ms RK4 steps; animation samples at 25 frames per second. RK4 is approximate: check convergence and energy drift when changing parameters or increasing speeds.

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
2. Force and gravity terms, accelerations, and the state derivative (implemented).
3. Numerical integration, energy checks, and animation (implemented).
4. Control environment, constraints, and baseline controllers.
5. Automated controller code generation, fixed evaluations, and logged iterations with explicit run and cost limits.
6. Explore a spatial model after deciding joint types, rod inertia, and cart actuation.

## Future 3D work

The planar model lives in its own module. Future integration and controller interfaces should accept a state derivative function so a spatial model can supply different physics. Three-dimensional dynamics need a fresh choice of coordinates and constraints; adding one more angle to these formulas is not sufficient. A spherical joint, a hinge, and a twisting rigid body have different degrees of freedom. Whether the cart moves on a line or a plane also matters.

## Validation

Tests compare the mass matrix with independently computed rigid-body energy and verify analytic single-rod acceleration, equilibrium, instantaneous motor power, and conservation of energy and horizontal momentum during short free-motion trajectories for one through four rods. These checks do not establish accuracy for every parameter choice or long chaotic trajectory.
