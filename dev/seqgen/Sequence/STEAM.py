import math

import numpy as np

from DTpy.Sequence.helper.sequence_helper import solve_trapezoidal_sequence
from DTpy.Scan import Scanner, Spin

class STEAM():
    """STimulated Echo Acquisition Mode"""

    def __init__(self, bvalue, Delta, scanner=Scanner(), spin=Spin(), alpha90=0, alphaRO=0):

        # store inputs for reference
        self._spin = spin
        self._scanner = scanner
        self._bvalue = bvalue
        self._Delta = Delta

        # compute the sequence
        self._t, self._G = compute_STEAM(spin.gamma, scanner.gradientStrength, scanner.slewRate, Delta, bvalue, alpha90, alphaRO)

    @property
    def Delta(self, return_stored=True):
        if return_stored:
            return self._Delta
        else:
            return self._t[8] - self._t[2]

    @property
    def TE(self):
        return (self._t[6] - self._t[1]) + (self._t[12] - self._t[7])

    @property
    def TM(self):
        # relevant for T1
        return self._t[7] - self._t[6]

    @property
    def delta(self):
        return self._t[4] - self._t[2]

    @property
    def diffusion_window(self):
        # where particle motion matters
        return (self._t[1], self._t[12])  # (from, to)

    @property
    def bvalue(self, return_stored=True):
        if return_stored:
            return self._bvalue
        else:  # calculate based on timing
            return solve_trapezoidal_sequence(g=self.gamma, G=self.Gmax, D=self.Delta, n=self.eta, e=self.epsilon)

    @property
    def Gmax(self, return_stored=True):
        if return_stored:
            return self._scanner.gradientStrength
        else:  # find maximum from sequence
            return max(self._gradients)


class InvalidSequenceException(Exception):
    pass


def verify_STEAM(t, TE, alpha90, TE_max=100e-3):

    TE_half = TE/2
    # verify  #TODO: are tests necessary?
    if not math.isclose(t[6]-t[1], t[12]-t[7]):
        raise InvalidSequenceException("half echo times not matching")
    if not math.isclose(t[6]-t[1], TE_half):
        raise InvalidSequenceException("echo time inconsistent")
    if not np.all(np.diff(t) >= 0):
        raise InvalidSequenceException("sequence is invalid")
    if TE > TE_max:
        raise InvalidSequenceException("TE_max exceeded, expect problems with T2")
    if t[6]-t[5] < alpha90:
        raise InvalidSequenceException("conflict with first 90deg RF pulse")
    if t[8]-t[7] < alpha90:
        raise InvalidSequenceException("conflict with second 90deg RF pulse")
    if t[7]-t[6] < 2*alpha90:
        raise InvalidSequenceException("conflict with both 90deg RF pulses")


def define_STEAM(alpha90, epsilon, eta, Delta, alphaRO):
    """Create STEAM sequence based on its key parameters.

    timings:
    t00: start of sequence
    t01: first 90deg RF pulse (initial excitation)
    t02: start of first gradient & start of first ramp-up
    t03: end of first ramp-up & start of first flat-top
    t04: end of first flat-top & start of first ramp-down
    t05: end of first ramp-down & end of first gradient
    t06: second 90deg RF pulse (store magnetisation)
    t07: third 90deg RF pulse (recover magnetisation)
    t08: start of second gradient & start of second ramp-up (= t02 + \Delta)
    t09: end of second ramp-up & start of second flat-top
    t10: end of second flat-top & start of second ramp-down
    t11: end of second ramp-down & end of second gradient
    t12: echo
    t13: end of read-out & end of sequence

    notes:
    - excitation pulse:
        - t00 is the start of the pulse
        - t02 is the end of the pulse
    - read-out:
        - t11 is the start of the read-out
        - t13 is the end of the read-out
    - 90deg pulses:
        - t05 is not necessarily the start of the first 90deg pulse
        - t08 is not necessarily the end of the second 90deg pulse
    """

    # echo time
    TE_half_1 = alpha90 + epsilon+eta+epsilon + alpha90
    TE_half_2 = alpha90 + epsilon+eta+epsilon + alphaRO
    TE_half = np.max([TE_half_1, TE_half_2])  # this is the theoretical minimum for TE, but not possible for alphaRO ~= alpha90
    #TODO: do we need this to be the max of the two? do the 90s have to be symmetric??
    TE = TE_half*2

    # init
    t = [None]*14

    # start
    t[ 0] = 0

    # first half of echo
    t[ 1] = t[ 0] + alpha90
    t[ 2] = t[ 1] + alpha90
    # gradient on
    t[ 3] = t[ 2] + epsilon
    t[ 4] = t[ 3] + eta
    t[ 5] = t[ 4] + epsilon
    # gradient off
    t[ 6] = t[ 1] + TE_half

    # second half of echo
    t[ 7] = None  # defined later
    t[ 8] = t[ 2] + Delta
    # gradient on
    t[ 9] = t[ 8] + epsilon
    t[10] = t[ 9] + eta
    t[11] = t[10] + epsilon
    # gradient off
    t[12] = t[11] + alphaRO
    t[ 7] = t[12] - TE_half

    # end
    t[13] = t[12] + alphaRO

    # check
    verify_STEAM(t, TE, alpha90)

    # define gradient state
    G = [0, 0, 0, +1, +1, 0, 0, 0, 0, -1, -1, 0, 0, 0]

    return t, G


def compute_STEAM(gamma, Gmax, Gdot, Delta, bvalue, alpha90, alphaRO):
    """
    the point of STEAM is to have a fixed diffusion time, so this is a simple sequence

    given the inputs, the only parameter that is not fixed is the gradient duration. This can be quickly calculated.
    """

    epsilon = Gmax / Gdot  # = 0 if Gdot==inf

    # compute flat-top time eta
    eta = solve_trapezoidal_sequence(D=Delta, b=bvalue, G=Gmax, g=gamma, e=epsilon)  # no n=eta
    if eta is None:
        raise InvalidSequenceException('Did not find a valid eta for this sequence')
    solve_trapezoidal_sequence(D=Delta, b=bvalue, G=Gmax, g=gamma, e=epsilon, n=eta)
    delta = eta+epsilon  # compute gradient duration

    t, G = define_STEAM(alpha90, epsilon, eta, Delta, alphaRO)
    G = [Gi*Gmax for Gi in G]
    return t, G
