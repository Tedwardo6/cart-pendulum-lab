# Lesson 2: from equations to moving pictures

There are three independent jobs:

1. `PlanarChain.derivative(state, force)` builds the mass matrix and right-hand side, solves for accelerations, and returns velocities followed by accelerations.
2. `simulate(...)` integrates that derivative over time. It does not know what a pendulum is. RK4 evaluates the derivative four times per step, using intermediate estimates to account for changing motion.
3. `render_simulation.py` converts each saved configuration into cart and rod positions and draws it. It does not compute the forces.

For two rods, the state is `[x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]`. Its derivative is `[x_dot, theta_1_dot, theta_2_dot, x_ddot, theta_1_ddot, theta_2_ddot]`.

## Why the cart moves without a motor

The demonstration passes zero external horizontal force. The rods still exert forces on the cart, so its velocity changes. Total horizontal momentum of the cart and rods stays constant; each component's momentum need not. With all components initially at rest, the system's horizontal center of mass stays fixed while its parts move.

Gravity converts potential energy into kinetic energy and back. No friction is included, so there is no physical damping. Small numerical energy errors are possible and must be measured.

## Timing

The default animation represents 12 seconds of physics. The numerical integrator advances in 0.002-second steps, while the renderer draws every twentieth saved state (25 frames per second). Drawing fewer frames changes visual smoothness, not the underlying computed trajectory.

The GIF loops by restarting the initial trajectory after the final frame; that restart is a playback effect. The HTML player defaults to playing once and provides playback and frame controls.

## Validation

The tests check that hanging at rest is an equilibrium; single-rod acceleration agrees with the analytic equations; energy changes at the applied motor power `force * cart_velocity`; and short free trajectories conserve energy and horizontal momentum to a numerical tolerance.

For the default double-pendulum example, also compare a run with half the time step. This helps distinguish physical motion from numerical artifacts. Chaotic trajectories can diverge over long times even when both integrations are accurate over shorter intervals.

## Conceptual checkpoint

Which function would you change to apply a controller's force? Which would you change just to give the cart a different color? Keeping those answers in different modules is what lets us extend the project cleanly.
