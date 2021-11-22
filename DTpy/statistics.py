from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt, acos
from statistics import mean, median, stdev, variance
from typing import Any

from .mathematics import Vector3, Tensor3


class TooFewDataPoints(Exception):
    def __init__(self, N_actual: int, N_target: int) -> None:
        super().__init__(
            f"Too few data points provided: {N_actual} ({N_target} required)"
        )


class DataAnalysis(ABC):
    _min_data = 1

    @property
    @abstractmethod
    def data(self) -> list[Any]:
        raise NotImplementedError

    @property
    def N(self) -> int:
        """Size of data set."""
        return len(self.data)

    def __post_init__(self) -> None:
        """Verify that enough data points are present."""
        if (N := len(self.data)) < self._min_data:
            raise TooFewDataPoints(N, self._min_data)


@dataclass
class ScalarAnalysis(DataAnalysis):
    data: list[float] = field(default_factory=list)
    _min_data = 2

    @property
    def sorted(self) -> list[float]:
        """Sorted representation of the data."""
        return list(sorted(self.data))

    @property
    def min(self):
        """Minimum of the data."""
        return min(self.data)

    @property
    def max(self):
        """Maximum of the data."""
        return max(self.data)

    @property
    def median(self):
        """Median of the data."""
        return median(self.data)

    @property
    def mean(self):
        """Mean of the data."""
        return mean(self.data)

    @property
    def stdev(self):
        """Standard deviation of the data."""
        return stdev(self.data)

    @property
    def variance(self):
        """Variance of the data."""
        return variance(self.data)

    @property
    def interval_stdev(self) -> tuple[float, float]:
        """Interval of +/- one standard deviation around the mean."""
        return (
            max(self.min, self.mean - self.stdev),
            min(self.mean + self.stdev, self.max),
        )

    @property
    def interval_minmax(self) -> tuple[float, float]:
        """Interval bounded by the extreme values."""
        return self.min, self.max

    @property
    def interval_ci95_2sided(self) -> tuple[float, float]:
        """2-sided confidence interval."""
        return ci_2sided(self.data, 0.95)


@dataclass
class VectorAnalysis(DataAnalysis):
    data: list[Vector3] = field(default_factory=list)
    _min_data = 2

    @property
    def mean_dyadic_tensor(self) -> Tensor3:
        """Mean dyadic tensor.
        Calculated as the average of each vector's dyadic tensor."""
        dyadics = [dyadic_tensor(vec).tensor for vec in self.data]
        mdt = sum(dyadics) / self.N  # same as np.mean(_, 0)
        return Tensor3(mdt)

    @property
    def angles_from_mean(self) -> list[float]:
        """Angles of the vectors from their mean.
        The first eigenvector of the mean dyadic tensor is used as the mean.
        Results are in radians."""
        mean_E1 = self.mean_dyadic_tensor.EigenVectors[0]
        return [smallest_angle_from_ref(vec, mean_E1) for vec in self.data]

    @property
    def coherence(self) -> float:
        """Coherence of the mean dyadic tensor."""
        l1, l2, l3 = self.mean_dyadic_tensor.EigenValues
        return 1 - sqrt( (l2 + l3) / (2 * l1) )  #TODO: Fix?

    @property
    def cone_of_uncertainty(self) -> float:
        """Cone of uncertainty.
        Calculated using the 95% one-sided confidence interval."""
        return ci_1sided(self.angles_from_mean, 0.95)


def dyadic_tensor(vec: Vector3) -> Tensor3:
    """Dyadic tensor of a vector.
    Computed as the outer product of the vector with itself."""
    outer = [a * b for a in vec for b in vec]  # same as np.outer
    return Tensor3(outer)


def smallest_angle_from_ref(vec: Vector3, ref: Vector3) -> float:
    """Calculate the angle of the vector.
    Results are in radians."""
    x = ref.dot(vec)
    x = abs(x)  # ensures mod(x, 180deg)
    return acos(x)


def ci_1sided(x: list[float], ci: float) -> float:
    """Function to calculate the one-sided 95% confidence interval.
    Returns the finite bound of the interval.
    TODO: ?? This assumes the data is normally distributed."""
    N = len(x)
    idx = min(round(ci * N), N-1)
    x_sorted = sorted(x)
    return x_sorted[idx]


def ci_2sided(x: list[float], ci: float) -> tuple[float, float]:
    """Function to calculate the two-sided 95% confidence interval.
    TODO: ?? This assumes the data is normally distributed."""
    N = len(x)
    # find bounds
    left = (1 - ci)/2
    right = ci + left
    # obtain indices
    idx_L = round(left * N)  # no need for max(_, 0)
    idx_R = min(round(right * N), N-1)
    # get interval
    x_sorted = sorted(x)
    return x_sorted[idx_L], x_sorted[idx_R]
