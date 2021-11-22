from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Tensor3:
    """A 3x3 tensor."""
    tensor: np.ndarray  # size: (3, 3)

    def __post_init__(self):
        """Ensure that data is in the correct format.
        This allows the user to provide either the 3x3 tensor
        or to input a 6-element list of coefficients.
        """
        self.tensor = np.reshape(self.tensor, (3, 3))  # throws if bad input

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
        ascending = np.argsort(eVal)
        descending = ascending[::-1]
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
