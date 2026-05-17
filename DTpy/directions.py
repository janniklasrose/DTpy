from math import sqrt

import numpy as np
from numpy.typing import NDArray

from .mathematics import Vector3


class Direction(Vector3):
    def normalised(self) -> "Direction":
        """Returns a normalised (unit length) copy of itself."""
        scale = 1/sqrt(sum(x**2 for x in self))
        values = [x*scale for x in self]
        return Direction(*values)


class UnsupportedDirections(Exception):
    def __init__(self, ndir: int) -> None:
        super().__init__(f"Unsupported number of directions {ndir}")


class OptimalDirections:
    """
    See:
    "Optimal Strategies for Measuring Diffusion in Anisotropic Systems by MRI"
    by Jones DK, Horsfield MA, Simmons A. Tables 1 & 2
    """

    @classmethod
    def estimate_trace(cls, ndir: int) -> list[Direction]:
        if ndir == 3:
            return [
                Direction(1, 0, 0),
                Direction(0, 1, 0),
                Direction(0, 0, 1),
            ]
        elif ndir == 6:
            return [
                Direction(1, 0, 0),
                Direction(0, 1, 0),
                Direction(0, 0, 1),
                Direction(-0.666, -0.333, +0.667),
                Direction(+0.334, +0.667, +0.667),
                Direction(-0.667, +0.667, -0.333),
            ]
        elif ndir == 9:
            return [
                Direction(1, 0, 0),
                Direction(0, 1, 0),
                Direction(0, 0, 1),
                Direction(-0.708, -0.706, +0.004),
                Direction(-0.503, +0.508, +0.699),
                Direction(-0.496, +0.493, -0.715),
                Direction(+0.517, +0.475, -0.713),
                Direction(+0.490, +0.518, +0.701),
                Direction(+0.702, -0.711, +0.035),
            ]
        else:
            raise UnsupportedDirections(ndir)

    @classmethod
    def estimate_tensor(cls, ndir: int) -> list[Direction]:
        if ndir == 6:
            return [
                Direction(+1.000, +0.000, +0.000),
                Direction(+0.447, +0.895, +0.000),
                Direction(+0.447, +0.277, +0.850),
                Direction(+0.447, -0.724, -0.525),
                Direction(+0.447, -0.724, +0.525),
                Direction(-0.447, -0.277, +0.850),
            ]
        elif ndir == 7:
            return [
                Direction(+1.000, +0.000, +0.000),
                Direction(+0.333, +0.943, +0.000),
                Direction(-0.333, +0.472, +0.817),
                Direction(+0.577, -0.817, +0.000),
                Direction(+0.333, -0.472, +0.817),
                Direction(-0.577, -0.407, +0.707),
                Direction(+0.577, +0.407, +0.707),
            ]
        elif ndir == 8:
            return [
                Direction(+1.000, +0.000, +0.000),
                Direction(+0.488, +0.873, +0.000),
                Direction(-0.200, +0.266, +0.942),
                Direction(+0.678, -0.700, +0.224),
                Direction(+0.119, -0.599, +0.792),
                Direction(-0.686, -0.324, +0.651),
                Direction(+0.631, +0.265, +0.729),
                Direction(-0.229, +0.876, +0.425),
            ]
        elif ndir == 9:
            return [
                Direction(+1.000, +0.000, +0.000),
                Direction(+0.251, +0.968, +0.000),
                Direction(-0.078, +0.116, +0.990),
                Direction(+0.604, -0.786, +0.133),
                Direction(+0.000, -0.691, +0.723),
                Direction(-0.723, -0.366, +0.586),
                Direction(+0.671, -0.178, +0.720),
                Direction(-0.488, +0.628, +0.606),
                Direction(+0.548, +0.606, +0.577),
            ]
        elif ndir == 10:
            return [
                Direction(+1.000, +0.000, +0.000),
                Direction(+0.678, +0.735, +0.000),
                Direction(-0.556, +0.504, +0.661),
                Direction(+0.672, -0.733, +0.106),
                Direction(+0.012, -0.801, +0.598),
                Direction(-0.680, -0.310, +0.664),
                Direction(-0.045, -0.011, +0.999),
                Direction(-0.024, +0.966, +0.257),
                Direction(+0.458, +0.521, +0.721),
                Direction(+0.658, -0.250, +0.710),
            ]
        else:
            raise UnsupportedDirections(ndir)


class SimpleDirections:

    @classmethod
    def two_simultaneous_gradients(cls) -> list[Direction]:
        """
        NOTE: The directions are not normalised!
        ? bvalue has to be per direction (2 at the same time)
        """
        return [
            Direction(+1, +1,  0),
            Direction(+1, -1,  0),
            Direction(+1,  0, +1),
            Direction(+1,  0, -1),
            Direction( 0, +1, +1),
            Direction( 0, +1, -1),
        ]


def numpy_format(directions: list[Direction]) -> NDArray:
    """
    [[x, y, z]_0, [x, y, z]_1, ..., [x, y, z]_n-1]  where n = number of directions
    """
    return np.array([dir.to_tuple() for dir in directions])
