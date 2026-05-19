from dataclasses import dataclass
from functools import cached_property
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def dot(self, other: "Vector3") -> float:
        return sum(a * b for a, b in zip(self, other))

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert data to a tuple."""
        return self.x, self.y, self.z

    def to_numpy(self):
        """Convert data to a numpy array."""
        return np.array(self.to_tuple())


class Tensor3:
    """A 3x3 tensor."""

    def __init__(self, tensor: Union[list[float], ArrayLike]) -> None:
        """Ensure that data is in the correct format.
        This allows the user to provide either the 3x3 tensor or to input a
        9-element list of coefficients. Alternatively, if a 6-element list is
        provided, a symmetric tensor is assumed. In that case, the elements
        should be:
            tensor = [xx, xy, xz, yy, yz, zz]
        """
        if np.size(tensor) == 6:
            # assume symmetric tensor
            tmp = np.empty((3, 3))
            tmp[np.triu_indices(3)] = tensor  # includes main diagonal
            tmp[np.tril_indices(3, -1)] = tmp[np.triu_indices(3, +1)]
            tensor = tmp  # overwrite

        self.tensor = np.reshape(tensor, (3, 3))  # throws if bad input

    @cached_property
    def _eigendecomposition(self):
        """Eigendecomposition of the tensor.
        Returns the eigenvalues and eigenvectors of the tensor. The output is
        sorted by descending magnitude of the eigenvalues, such that the
        primary eigenvector has the largest corresponding eigenvalue.

        See <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>
        """
        eVal, eVec = np.linalg.eig(self.tensor)
        # the output of linalg.eig is not guaranteed to be sorted
        descending = np.argsort(-eVal, kind="stable")
        eVal = eVal[descending]
        eVec = eVec[:, descending]  # columns are vectors
        return eVal, eVec

    @property
    def EigenValues(self) -> tuple[float, float, float]:
        """Eigenvalues of the diffusion tensor.
        The eigenvalues are sorted in descending order.

        L = [L1, L2, L3]
        """
        L1: float
        L2: float
        L3: float
        L1, L2, L3 = self._eigendecomposition[0]
        return L1, L2, L3

    @property
    def EigenVectors(self) -> tuple[Vector3, Vector3, Vector3]:
        """Eigenvectors of the diffusion tensor.
        Returns a 3x3 array whose column vectors corresponds to the eigenvalue.

        E = [[E1x, E2x, E3x],
             [E1y, E2y, E3y],
             [E1z, E2z, E3z]]
        """
        vectors = self._eigendecomposition[1]
        E1: Vector3 = Vector3(*vectors[:, 0])
        E2: Vector3 = Vector3(*vectors[:, 1])
        E3: Vector3 = Vector3(*vectors[:, 2])
        return E1, E2, E3
