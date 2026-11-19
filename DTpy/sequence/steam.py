import math

import numpy as np

from ..scanner import Scanner, DefaultScanner
from .sequence import Sequence
from .sequence_helper import solve_trapezoidal_sequence
from ..spin import Proton, Spin


class InvalidSequenceException(Exception):
    pass


class STEAM():
    """STimulated Echo Acquisition Mode."""

    def __init__(
        self,
        bvalue: float,
        Delta: float,
        scanner: Scanner | None = None,
        spin: Spin | None = None,
        alpha90: float = 0,
        alphaRO: float = 0,
    ) -> None:
        self._spin = spin if spin is not None else Proton
        self._scanner = scanner if scanner is not None else DefaultScanner
        self._bvalue = bvalue

        timepoints, gradients = compute_STEAM(
            self._spin.gamma,
            self._scanner.gradientStrength,
            self._scanner.slewRate,
            Delta,
            bvalue,
            alpha90,
            alphaRO,
        )
        self.sequence = Sequence(timepoints=timepoints, gradients=gradients)

    @property
    def Delta(self) -> float:
        """Diffusion time"""
        return self.sequence.timepoints[8] - self.sequence.timepoints[2]

    @property
    def TE(self) -> float:
        """Echo time"""
        return (
            self.sequence.timepoints[6] - self.sequence.timepoints[1]
        ) + (
            self.sequence.timepoints[12] - self.sequence.timepoints[7]
        )

    @property
    def TM(self) -> float:
        """Mixing time"""
        return self.sequence.timepoints[7] - self.sequence.timepoints[6]

    @property
    def delta(self) -> float:
        """Gradient duration"""
        return self.sequence.timepoints[4] - self.sequence.timepoints[2]

    @property
    def diffusion_window(self) -> tuple[float, float]:
        return self.sequence.timepoints[1], self.sequence.timepoints[12]

    @property
    def eta(self) -> float:
        """Flat-top time"""
        return self.sequence.timepoints[4] - self.sequence.timepoints[3]

    @property
    def epsilon(self) -> float:
        """Ramp time"""
        return self.sequence.timepoints[3] - self.sequence.timepoints[2]

    @property
    def bvalue(self) -> float:
        return self._bvalue

    @property
    def Gmax(self) -> float:
        """Maximum gradient strength"""
        return max(self.sequence.gradients)


def verify_STEAM(t, TE, alpha90, TE_max=100e-3):
    TE_half = TE / 2

    if not math.isclose(t[6] - t[1], t[12] - t[7]):
        raise InvalidSequenceException("half echo times not matching")
    if not math.isclose(t[6] - t[1], TE_half):
        raise InvalidSequenceException("echo time inconsistent")
    if not np.all(np.diff(t) >= 0):
        raise InvalidSequenceException("time goes backwards")
    if TE > TE_max:
        raise InvalidSequenceException("TE_max exceeded, expect problems with T2")
    if t[6] - t[5] < alpha90:
        raise InvalidSequenceException("conflict with first 90deg RF pulse")
    if t[8] - t[7] < alpha90:
        raise InvalidSequenceException("conflict with second 90deg RF pulse")
    if t[7] - t[6] < 2 * alpha90:
        raise InvalidSequenceException("conflict with both 90deg RF pulses")
    #TODO: more checks?
    
    #TODO: calculate bvalue at the end
    # bvalue = solve_trapezoidal_sequence(
    #             g=self._spin.gamma,
    #             G=self.Gmax,
    #             D=self.Delta,
    #             n=self.eta,
    #             e=self.epsilon,
    #         )

    #TODO: assert that all inputs are in the output (Gmax, etc)


def define_STEAM(alpha90, epsilon, eta, Delta, alphaRO):
    """Create STEAM sequence timing and gradient state from key parameters.

    t_00: start of sequence
    t_01: first 90deg RF pulse (initial excitation)
    t_02: start of first gradient & start of first ramp-up
    t_03: end of first ramp-up & start of first flat-top
    t_04: end of first flat-top & start of first ramp-down
    t_05: end of first ramp-down & end of first gradient
    t_06: second 90deg RF pulse (store magnetisation)
    t_07: third 90deg RF pulse (recover magnetisation)
    t_08: start of second gradient & start of second ramp-up (= t_02 + Delta)
    t_09: end of second ramp-up & start of second flat-top
    t_10: end of second flat-top & start of second ramp-down
    t_11: end of second ramp-down & end of second gradient
    t_12: echo
    t_13: end of read-out & end of sequence

    notes:
        excitation pulse:
            t_00 is the start of the pulse
            t_02 is the end of the pulse
        read-out:
            t_11 is the start of the read-out
            t_13 is the end of the read-out
        90deg pulses:
            t_05 is not necessarily the start of the first 90deg pulse
            t_08 is not necessarily the end of the second 90deg pulse
    """

    # echo time calculation
    TE_half_1 = alpha90 + epsilon + eta + epsilon + alpha90
    TE_half_2 = alpha90 + epsilon + eta + epsilon + alphaRO
    TE_half = np.max([TE_half_1, TE_half_2]) # this is the theoretical minimum for TE, but not possible for alphaRO ~= alpha90
    #TODO: do we need this to be the max of the two? do the 90s have to be symmetric??
    TE = TE_half * 2

    # initialise
    t = [0.0] * 14

    # start
    t[0] = 0
    # wait for first pulse
    t[1] = t[0] + alpha90
    # first half of echo
    t[2] = t[1] + alpha90
    # gradient on
    t[3] = t[2] + epsilon
    t[4] = t[3] + eta
    t[5] = t[4] + epsilon
    # gradient off
    t[6] = t[1] + TE_half
    # diffusion ...
    t[7] = -1  # defined later
    # second half of echo
    t[8] = t[2] + Delta
    # gradient on
    t[9] = t[8] + epsilon
    t[10] = t[9] + eta
    t[11] = t[10] + epsilon
    # gradient off
    t[12] = t[11] + alphaRO
    # wait for signal
    t[13] = t[12] + alphaRO
    # end

    # post-calculate
    t[7] = t[12] - TE_half

    verify_STEAM(t, TE, alpha90)

    gradients = [0, 0, 0, +1, +1, 0, 0, 0, 0, -1, -1, 0, 0, 0]
    return t, gradients


def compute_STEAM(
        gamma: float,
        Gmax: float,
        Gdot: float,
        Delta: float,
        bvalue: float,
        alpha90: float,
        alphaRO: float,
):
    """
    The point of STEAM is to have a fixed diffusion time, so this is a simple sequence
    given the inputs, the only parameter that is not fixed is the gradient duration. This can be quickly calculated.
    """

    #TODO: guard against math error for Gdot->Inf (in which case episilon is zero)
    epsilon = Gmax / Gdot

    # compute flat-top time eta
    eta = solve_trapezoidal_sequence(D=Delta, b=bvalue, G=Gmax, g=gamma, e=epsilon)
    if eta is None:
        raise InvalidSequenceException("Did not find a valid eta for this sequence")

    #TODO: why no return? what does this do?
    solve_trapezoidal_sequence(
        D=Delta,
        b=bvalue,
        G=Gmax,
        g=gamma,
        e=epsilon,
        n=eta,
        rel_tol=1e-6,
        abs_tol=1e-9,
    )

    timepoints, gradients = define_STEAM(alpha90, epsilon, eta, Delta, alphaRO)
    gradients = [gradient * Gmax for gradient in gradients]

    #TODO: validate

    return timepoints, gradients
