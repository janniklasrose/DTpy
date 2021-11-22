from math import sqrt

from .mathematics import Tensor3, Vector3


class DiffusionTensor(Tensor3):
    """A 3x3 diffusion tensor.

    D = [[D_xx, D_xy, D_xz],
         [D_yx, D_yy, D_yz],
         [D_zx, D_zy, D_zz]]
    """

    # Eigenvalues and vectors, available through their common names

    @property
    def L1(self) -> float:
        """Primary eigenvalue.
        This corresponds to the primary eigenvector (E1).
        """
        return self.EigenValues[0]

    @property
    def L2(self) -> float:
        """Secondary eigenvalue.
        This corresponds to the secondary eigenvector (E2).
        """
        return self.EigenValues[1]

    @property
    def L3(self) -> float:
        """Tertiary eigenvalue.
        This corresponds to the tertiary eigenvector (E3).
        """
        return self.EigenValues[2]

    @property
    def E1(self) -> Vector3:
        """Primary eigenvector.
        This corresponds to the primary eigenvalue (L1).
        """
        return self.EigenVectors[0]

    @property
    def E2(self) -> Vector3:
        """Secondary eigenvector.
        This corresponds to the secondary eigenvalue (L2).
        """
        return self.EigenVectors[1]

    @property
    def E3(self) -> Vector3:
        """Tertiary eigenvector.
        This corresponds to the tertiary eigenvalue (L3).
        """
        return self.EigenVectors[2]

    # Tensor size and shape parameters

    @property
    def MD(self) -> float:
        """Mean diffusivity.
        MD = <L>  # average of eigenvalues
        """
        return sum(self.EigenValues)/3

    @property
    def FA(self) -> float:
        """Fractional anisotropy."""
        scale = 3 / 2
        numerator = sum((x - self.MD)**2 for x in self.EigenValues)
        denominator = sum(x**2 for x in self.EigenValues)
        return sqrt(scale * numerator / denominator)

    @property
    def Mo(self) -> float:
        """Tensor mode."""
        moment2 = sum((x - self.MD)**2 for x in self.EigenValues)/3
        moment3 = sum((x - self.MD)**3 for x in self.EigenValues)/3
        return sqrt(2) * moment3 * moment2**(-3/2)
