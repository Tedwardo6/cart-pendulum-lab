# Lesson 1: turn the mass matrix into Python

## What this function does

Our equations will be `M(q) @ q_ddot = b(q, q_dot, u)`. This lesson builds only `M(q)`. It needs angles and physical parameters, not velocities or cart position. Velocities will enter `b` in the next lesson.

## Parameters

```python
chain = PlanarChain(
    cart_mass=1.0,
    lengths=(1.0, 1.0),
    masses=(1.0, 1.0),
)
```

Each tuple has one entry per rod. Four entries specify four rods. A dataclass holds these values together. `frozen=True` prevents accidentally changing a parameter midway through a simulation.

## Math-to-code map

For rod i, let s_i be the sum of masses beyond it:

- `downstream[i]` = s_i
- `a[i] = lengths[i] * (masses[i] / 2 + downstream[i])`
- `d[i] = lengths[i]**2 * (masses[i] / 3 + downstream[i])`

The halves arise from centers of mass halfway along uniform rods. The thirds include each rod's center translation and rotational inertia. For i < j, the rod coupling coefficient is `lengths[i] * a[j]`.

## Indexing

Python rod arrays start at zero. Matrix row 0 is reserved for the cart, so Python rod i maps to matrix row i + 1. The same applies to columns.

The upper-left entry is total mass. The remaining diagonal entries are d_i. Cart/rod entries are a_i cos(theta_i). Rod/rod entries are c_ij cos(theta_i - theta_j).

The matrix is symmetric because each cross term in kinetic energy couples the same pair of velocities in either order. It is not a transformation matrix; it contains the coefficients multiplying accelerations in the equations of motion.

## Read the example

The example sets the first rod downward and the second horizontally right. Its matrix is approximately:

```text
[[3.     1.5    0.    ]
 [1.5    1.3333 0.    ]
 [0.     0.     0.3333]]
```

The zero entries occur because the corresponding cosines vanish at this configuration. They do not mean the rods stay uncoupled as they move, or that gravity vanishes. Gravity belongs in the right-hand side, which comes next.

## Next lesson

Build b using the current angles, angular velocities, and applied cart force. Then call `np.linalg.solve(matrix, b)` for accelerations. Finally concatenate current velocities and accelerations to obtain the full state derivative.
