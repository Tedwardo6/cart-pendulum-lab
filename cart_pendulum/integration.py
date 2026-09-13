"""Time integration independent of the pendulum geometry."""

import numpy as np


def rk4_step(derivative, state, dt):
    """Four evaluations estimate how the derivative changes within one step.

    derivative(state) returns the entire state derivative. Any applied force
    should be supplied by that function, using the intermediate state if needed.
    """
    k1 = derivative(state)
    k2 = derivative(state + dt * k1 / 2)
    k3 = derivative(state + dt * k2 / 2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate(derivative, initial_state, duration=12.0, dt=0.002):
    """Return times and states, shortening the final step to end at duration."""
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Duration and time step must be finite and positive.")
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.ndim != 1 or not np.all(np.isfinite(initial_state)):
        raise ValueError("Initial state must be a finite one-dimensional array.")
    count = int(np.ceil(duration / dt))
    times = np.minimum(np.arange(count + 1) * dt, duration)
    states = np.empty((count + 1, len(initial_state)))
    states[0] = initial_state
    for k in range(count):
        states[k + 1] = rk4_step(derivative, states[k], times[k + 1] - times[k])
        if not np.all(np.isfinite(states[k + 1])):
            raise FloatingPointError("Integration diverged; check parameters and time step.")
    return times, states
