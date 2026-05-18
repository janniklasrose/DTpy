from typing import Optional, Protocol

from DTpy import Box, Voxel


class SimulationData(Protocol):
    @property
    def position(self):
        ...
    @property
    def phase(self):
        ...


def apparent_diffusion(
    initial: SimulationData,
    at_Delta: SimulationData,
    Delta: float,
    voxel: Optional[Voxel] = None,
):
    """Compute RMS based on positions between t=0 and t=Delta"""
    displacement = inside_voxel(at_Delta.position - initial.position, at_Delta.position, voxel)
    def rms(dim: float | list[float]):
        if isinstance(dim, float):
            dim = [dim]
        return np.mean(displacement[:, dim]**2) / (2 * len(dim) * Delta)
    return rms(0), rms(1), rms(2), rms([0, 1, 2])


def process_simulation(
    initial: SimulationData,
    final: SimulationData,
    voxel: Optional[Voxel] = None,
):
    pass


def is_inside(x: tuple[float, float, float], box: Box):
    return all(
        x[dim] >= box[2*dim] and x[dim] <= box[2*dim + 1] for dim in range(3)
    )

def inside_voxel(data, position, voxel: Optional[Voxel] = None):
    if voxel is None:
        return data
    inside = [is_inside(pos, voxel) for pos in position.T]
    return data[inside]


if __name__ == "__main__":
    computeResults(phase[insideVoxel], directions)
