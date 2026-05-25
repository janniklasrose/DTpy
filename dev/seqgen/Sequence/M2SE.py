from Sequence import Sequence

class M2SE(Sequence):
    """Motion Compensated Spin Echo"""
    def set_sequence(self, bvalue, scanner):
        Gmax = scanner.Gmax
        Gdot = scanner.slew
        tau  = Gmax/Gdot
        del1 = delta1+2*tau
        del2 = delta2+2*tau

        gamma = self.spin.gamma


def make_M2SE():
    tau = G_peak/slewrate
    tau_val = tau


    TEmax = 100e-3  # limited by T2
    TEmin = alpha90 + 2*tau + alpha180 + 2*tau + alphaEcho  # this is the theoretical minimum for TE, but not possible for alphaEcho ~= alpha90

    # TE = 69.2e-3;
    # DELTA = 34.6e-3;
    # tau = 0.66e-3;
    # tau = 0; % zero (unrealistic for now, if nonzero need to adjust times/gradients below)
    #
    # delta1 = 9e-3; % 5.6+2*0.9
    # delta2 = delta1*(DELTA-tau)/(DELTA-2*delta1+tau);
    # gap = 2*tau + delta1
    # TM = DELTA-delta1-delta2;
    # alpha = (TE - (delta1+delta2+TM+delta2+delta1))/2;

    # we will go through all DELTA values and pick the optimal one
    for Delta in np.arange(TE_min, TE_max, 0.1e-3):

        # compute delta
        delta2_val = solx.evalf(subs={DELTA_sym: Delta, tau_sym: tau})  # use real part (?)

        valid = false(size(delta2_val))
        for j, delta2_val_j in enumerate(delta2_val):
            valid[j] = isreal(delta2_val_j) & delta2_val_j>=0
        del2 = del2[valid]  # has all positive real deltas
        del2 = min(del2)  # take smallest

        # did not find a positive real delta
        if del2 is None or isnan(del2) or isinf(del2):
            continue

        # checks
        del1 = del2*(Delta+tau)/(Delta+2*del2-tau)
        gap = Delta-del2-del1
        TE_i = alpha90 + Delta + (del1+del2) + alphaEcho
        t0 = 0
        t1 = t0 + alpha90
        t4 = t1 + del1  # end of first up gradient
        t8 = t4 + del2  # end of first down gradient
        t9 = t1 + Delta  # start of second down gradient
        t15 = t9 + del1+del2  # end of second up gradient
        if TE > maxTE or TE < minTE:
            raise Exception("TE out of bounds")
        if t9 < TE/2+alpha180/2 or t8 > TE_i/2-alpha180/2:
            raise Exception("180 at TE/2 overlaps with gradients")
        if t15 > TE-alphaEcho:
            raise Exception("second gradient overlaps with signal readout")

    eta1 = del1-2*tau
    eta2 = del2-2*tau
    TM = Delta-4*tau-eta1-eta2
    Delta_del1del2tau = (2*del1*del2 - del2*tau - del1*tau)/(del2-del1)
    TM = 2*tau+eta1
    TM2 = Delta_del1del2tau - del1 - del2
