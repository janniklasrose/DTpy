from dataclasses import dataclass

@dataclass
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


class Voxel(Box):
    pass
