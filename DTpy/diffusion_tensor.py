from math import prod, sqrt

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

    @property
    def Moment1(self) -> float:
        """First moment of the eigenvalues."""
        return self.MD

    @property
    def Moment2(self) -> float:
        """Second central moment of the eigenvalues."""
        return sum((x - self.Moment1)**2 for x in self.EigenValues) / 3

    @property
    def Moment3(self) -> float:
        """Third central moment of the eigenvalues."""
        return sum((x - self.Moment1)**3 for x in self.EigenValues) / 3

    # Tensor size and shape parameters
    # https://en.wikipedia.org/wiki/Diffusion_MRI#Measures_of_anisotropy_and_diffusivity

    @property
    def AxialDiffusivity(self) -> float:
        """Axial diffusivity.
        Equivalent to the primary eigenvalue.
        """
        return self.L1

    @property
    def RadialDiffusivity(self) -> float:
        """Radial diffusivity.
        Average of the secondary and tertiary eigenvalues.
        """
        return (self.L2 + self.L3)/2

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
        numerator = 3 * self.Moment2
        denominator = sum(x**2 for x in self.EigenValues)
        return sqrt(scale * numerator / denominator)

    @property
    def Mo(self) -> float:
        """Tensor mode."""
        return sqrt(2) * self.Sk

    @property
    def Sk(self) -> float:
        """Skewness."""
        return self.Moment3 * self.Moment2**(-3/2)

    @property
    def RA(self) -> float:
        """Relative Anisotropy."""
        sum_terms = sum((x - self.MD)**2 for x in self.EigenValues)
        return sqrt(sum_terms) / (sqrt(3) * self.MD)

    @property
    def VR(self) -> float:
        """Volume ratio."""
        return prod(self.EigenValues) / self.MD**3

    @property
    def C_l(self) -> float:
        """
        Linear case.
        L1 >> L2 ~ L3
        """
        return (self.L1 - self.L2) / self.Trace

    @property
    def C_p(self) -> float:
        """
        Planar case.
        L1 ~ L2 >> L3
        """
        return 2 * (self.L2 - self.L3) / self.Trace

    @property
    def C_s(self) -> float:
        """
        Spherical case.
        L1 ~ L2 ~ L3

        == self.L3/self.MD
        """
        return 3 * self.L3 / self.Trace

    @property
    def C_a(self) -> float:
        """
        Anisotropy measure.
        C_a = C_l + C_p = 1 - C_s = (L1 + L2 - 2*L3)/(L1 + L2 + L3)
        """
        return 1 - self.C_s
