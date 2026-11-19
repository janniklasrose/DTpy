from dataclasses import dataclass, field

from ..directions import Direction


@dataclass
class SequencePoint:
    dt: float
    G: float
    dir: Direction


@dataclass
class Sequence:
    points: list[SequencePoint] = field(default_factory=list)
    timepoints: list[float] = field(default_factory=list)
    gradients: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.timepoints and self.gradients and len(self.timepoints) != len(self.gradients):
            raise ValueError("timepoints and gradients must have the same length")
        if any(t1 < t0 for t0, t1 in zip(self.timepoints, self.timepoints[1:])):
            raise ValueError("timepoints must be monotonically increasing")

    @property
    def t(self) -> list[float]:
        return self.timepoints

    @property
    def G(self) -> list[float]:
        return self.gradients
