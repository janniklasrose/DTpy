from dataclasses import dataclass, field

from .directions import Direction


@dataclass
class SequencePoint:
    dt: float
    G: float
    dir: Direction


@dataclass
class Sequence:
    points: list[SequencePoint] = field(default_factory=list)
