from Sequence import Sequence

class PGSE(Sequence):
    """Pulse(D???) Gradient Spin Echo"""

    def bvalue(self):
        G = self.G_peak  # with or without sqrt(2)
        DELTA = self.Delta
        delta = self.delta + epsilon  # with or without?
        epsilon = self.epsilon
        
        gG2 = (self.spin.gamma*G)**2
        bvalue = gG2 * ( delta**2*(DELTA-delta/3) \
                       + epsilon**3/30 \
                       - delta*epsilon**2/6 \
                       )
        return bvalue
    def TE(self):
        return np.sum(self.dt)  # all distance
    def TM(self):
        return DELTA - (epsilon+delta+epsilon)

















def makePGSE(this, bfactor, G_peak, slewrate):

    epsilon = G_peak/slewrate

    maxTE = 100e-3  # limited by T2
    minTE = alpha90 + 2*epsilon + alpha180 + 2*epsilon + alphaEcho  # this is the theoretical minimum for TE, but not possible for alphaEcho ~= alpha90

    DELTA_values = linspace()
    # the reason we try Delta values is that we can calculate TE for an exact sequence.
    # otherwise, we would have to try TE values and calculate if the sequence is correct (with some error).
    # by varying TE, the only constraint is the resolution, but it will always lead a valid sequence, just not guaranteed to be the lowest TE.
    #TODO: express as function that can be minimised/root found

    # we will go through all DELTA values and pick the optimal one
    seqs = []  # empty initially
    for i, Delta in enumerate(DELTA_values):  # try all possible DELTA values

        # compute delta
        eta = solve_trapezoidal_sequence(b=bfactor, g=gamma, G=G_peak, D=Delta, e=epsilon)
        # did not find a positive real delta
        if eta is None or isempty(eta) or isnan(eta) or isinf(eta) or iscomplex(eta) or eta<0:
            continue  # delta_values(i) stays NaN

        delta = eta + epsilon

        # check if delta is valid
        graddur = epsilon + delta + epsilon
        TE = alpha90 + Delta + graddur + alphaEcho
        t0_i = 0
        t1_i = t0_i + alpha90
        t4_i = t1_i + graddur  # end of first gradient
        t5_i = t1_i + Delta  # start of second gradient
        t6_i = t5_i + graddur  # end of second gradient

        #... add more constraints
        if TE > maxTE or TE < minTE:  # TE in bounds
            continue
        if t5_i < TE/2 + alpha180/2 or t4_i > TE/2 - alpha180/2:  # 180 at TE/2 overlaps with gradients
            continue
        if t6_i > TE - alphaEcho:  # second gradient overlaps with signal readout
            continue

        seqs.append((delta, TE))  #NOTE: this is the delta in the bvalue notation, i.e. flat+ramp

    if len(seqs) == 0:
        raise Exception('No suitable sequence found')

    # pick the delta that yielded the lowest TE
    seqs = np.array(seqs)
    mindex = np.argmin(seqs[:, 0])  # index of minimum TE
    delta, TE = seqs[mindex]
    eta = delta - epsilon

    #TODO: refine delta around where it was low?
