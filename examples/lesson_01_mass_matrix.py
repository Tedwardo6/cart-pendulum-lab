"""Run from the repository root: python3 -m examples.lesson_01_mass_matrix"""

import numpy as np

from cart_pendulum import PlanarChain


def main():
    chain = PlanarChain(cart_mass=1.0, lengths=(1.0, 1.0), masses=(1.0, 1.0))
    angles = np.array([0.0, np.pi / 2])
    matrix = chain.mass_matrix(angles)

    np.set_printoptions(precision=4, suppress=True)
    print("First rod downward, second rod pointing right.")
    print("Rows and columns: cart, first rod, second rod.")
    print(matrix)
    print("\nNext lesson: build b and solve M @ accelerations = b.")


if __name__ == "__main__":
    main()
