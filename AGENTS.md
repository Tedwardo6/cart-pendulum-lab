# Working on Cart Pendulum Lab

This is a teaching collaboration. Explain each small implementation step and connect it to the derived physics. Check conceptual understanding at meaningful milestones; avoid repetitive arithmetic exercises. Do not jump ahead and build the entire roadmap unless the user asks.

Keep planar physics separate from numerical integration, rendering, and controller evaluation. Support any positive number of uniform rods; validate at least one through four. A future spatial model must explicitly specify joints, inertia, coordinates, and cart actuation.

Use radians and SI units. Angles are absolute from downward. Planned state order is positions followed by velocities. Preserve these conventions in documentation and tests.

Run `python3 -m unittest discover -s tests -v` and the relevant example after changing physics. Validate against independent physical identities, not copies of implementation formulas. Never claim a complete simulator or controller exists before it does.

Do not commit credentials, environment files, or generated experiment runs. Future automated control experiments must keep evaluation rules fixed and use explicit budgets and stop conditions.
