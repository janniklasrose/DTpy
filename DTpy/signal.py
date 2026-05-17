from cmath import exp as cexp
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .diffusion_tensor import DiffusionTensor
from .mathematics import Tensor3, Vector3
from .sequence import Direction
from .spin import Spin


class B_Tensor(Tensor3):
    """B-tensor."""

    @classmethod
    def from_single_simulation(cls, dir: Direction, bvalue: float = 1):
        """
        bvalue has to be per direction (2 at the same time) ???? set to 1 if directions already is bvalue
        #TODO: change directions into bvalues_per_direction (bx,by,bz)_i
        signal_ratio and directions need to have same size among common (main) axis
        directions needs to have a second axis of length 3 (x,y,z)
        """
        tensor = np.array([
            dir.x * dir.x,
            dir.x * dir.y * 2,
            dir.x * dir.z * 2,
            dir.y * dir.y,
            dir.y * dir.z * 2,
            dir.z * dir.z,
        ]) * bvalue
        return super()(tensor)


def calculate_diffusion_tensor(signal_ratios: Union[list[float], NDArray], b_tensors: list[B_Tensor]) -> DiffusionTensor:
    """Calculate the diffusion tensor from the signal data."""
    row, col = np.triu_indices(3)  # xx, xy, xz, yy, yz, zz
    # prepare linear system
    b_matrix = np.empty((len(b_tensors), len(row)))
    for i, b_tensor in enumerate(b_tensors):
        b_matrix[i, :] = b_tensor.tensor[row, col]
    attenuation_vector = np.log(signal_ratios)
    # least squares solution
    coefficients = -np.linalg.lstsq(b_matrix, attenuation_vector)
    # assign output
    tensor = np.zeros((3, 3))
    tensor[row, col] = tensor[col, row] = coefficients  # use symmetry
    return DiffusionTensor(tensor)


def calculate_signal(phases: Union[list[float], NDArray]) -> float:
    N = len(phases)
    return abs(sum(cexp(complex(0, phi)) for phi in phases)/N)


def calculate_phase(
    unscaled_integral: Union[list[Vector3], NDArray],
    direction: Direction,
    spin: Optional[Spin] = None,
) -> list[float]:
    """
    In simulations, the integral can be calculated without specifying the
    gradient direction (only the gradient magnitude is needed). This way,
    different directions can be evaluated without repeated simulations.
    This requires simple acquisitions with constant gradient direction.
    """

    def dot(vec3, dir: Direction) -> float:
        # assume vec3 is iterable, e.g. if input is 2DArray
        return sum(a * b for a, b in zip(vec3, dir))
    phases = [dot(phi, direction) for phi in unscaled_integral]

    if spin is not None:
        phases = [x * spin.gamma for x in phases]
    return phases
