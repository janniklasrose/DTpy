# iT  :  state, starts at zero so that first call to .next() initialises to 1




class DiscreteSequence():

    def __init(self, NT):
        self.NT = NT
        # we must now discretise
        dt, Gv = discretise(seq_parameters)
        self.dt = dt
        self.Gv = Gv
        #self.MAXDTCORSE = 1e-3
        #self.MAXDTFINE = 1e-4

    def has_next(self):
        if this.iT >= length(this.dt):  # at the end
            bool = false
            return
        this.iT = this.iT + 1
        bool = true
        return

    def next(self):
        iT = iT + 1
        return (dt[iT, :], grad[iT, :])


    def plot(self):
        #TODO: this can already be in Sequence, just gets overridden in DiscreteSequence

        t = np.cumsum(dt)  #TODO: add ZERO in beginning of dt. what about grad
        plt.plot(t, grad)

    @property
    def dt(self):
        # shape (NT,)
        return None

def discretise_sequence(t, ID, dt_target):
    dt = []
    gG = []
    for t0_i, t1_i in JOIN(t[0:-1], t[1:]):
        Dt_i = t1_i - t0_i
        NT_i = math.ceil(Dt_i / dt_target)
        if NT_i == 0:  # time points coincide
            pass
        dt_i = Dt_1/NT_1

        dt.append([dt_i]*NT_i)


        ####### gradients

        t_ms = cumsum(t_Dt)  # milestones
        # gradient 1
        gG(i) = +G_peak/(t_ms(2)-t_ms(1))  # gradient 1a - ramp up
        gG(i) = 0                          # gradient 1a - plateau
        gG(i) = -G_peak/(t_ms(4)-t_ms(3))  # gradient 1a - ramp down
        gG(i) = -G_peak/(t_ms(5)-t_ms(4))  # gradient 1b - ramp down
        gG(i) = 0                          # gradient 1b - plateau
        gG(i) = +G_peak/(t_ms(7)-t_ms(6))  # gradient 1b - ramp up
        # gradient 2
        gG(i) = +G_peak/(t_ms(7+2)-t_ms(7+1))  # gradient 2a - ramp up
        gG(i) = 0                              # gradient 2a - plateau
        gG(i) = -G_peak/(t_ms(7+4)-t_ms(7+3))  # gradient 2a - ramp down
        gG(i) = -G_peak/(t_ms(7+5)-t_ms(7+4))  # gradient 2b - ramp down
        gG(i) = 0                              # gradient 2b - plateau
        gG(i) = +G_peak/(t_ms(7+7)-t_ms(7+6))  # gradient 2b - ramp up



        gG_peak = sqrt(bfactor/bGterm)
        dG = [0, G_peak/epsilon, 0, -G_peak/epsilon, 0]
        dGdt[0] = +G_peak/(t2-t1)  # gradient 1 - ramp up
        dGdt[1] = 0                # gradient 1 - plateau
        dGdt[2] = +G_peak/(t4-t3)  # gradient 1 - ramp down
        dGdt[3] = -G_peak/(t6-t5)  # gradient 2 - ramp down
        dGdt[4] = 0                # gradient 2 - plateau
        dGdt[5] = -G_peak/(t8-t7)  # gradient 2 - ramp up




        gG = np.zeros(size(dt));
        gG_peak = this.gamma*G_peak;
        for i in range(numel(G)):
            if   t(i) >= t_ms(1) and t(i) < t_ms(2) : # gradient 1 - ramp up
                gG(i) = +gG_peak * (t(i)-t_ms(1))/(t_ms(2)-t_ms(1))
            elif t(i) >= t_ms(2) and t(i) < t_ms(3) : # gradient 1 - plateau
                gG(i) = +gG_peak
            elif t(i) >= t_ms(3) and t(i) < t_ms(4) : # gradient 1 - ramp down
                gG(i) = +gG_peak*(1 - (t(i)-t_ms(3))/(t_ms(4)-t_ms(3)))
            elif t(i) >= t_ms(5) and t(i) < t_ms(6) : # gradient 2 - ramp down
                gG(i) = -gG_peak * (t(i)-t_ms(5))/(t_ms(6)-t_ms(5))
            elif t(i) >= t_ms(6) and t(i) < t_ms(7) : # gradient 2 - plateau
                gG(i) = -gG_peak
            elif t(i) >= t_ms(7) and t(i) < t_ms(8) : # gradient 2 - ramp up
                gG(i) = -gG_peak * (1 - (t(i)-t_ms(7))/(t_ms(8)-t_ms(7)))




    t = 0
    for i=1:numel(t_Dt):
        t = [t(1:end-1), t(end)+linspace(0, t_Dt(i), numberofsteps(i))];
    t = unique(t)  # should already be unique but lets be safe
    dt = diff(t)
    gG = zeros(size(dt))
    G = np.cumsum(dG*dt, 0)
    #TODO: not guaranteed to return to zero


def refine():
    idFine = ID > 0
    idCoarse = ID == 0
    dt_target = idFine*dt_fine + idCoarse*dt_coarse

    durationFine = sum(durations[idFine])
    durationCorse = sum(durations[idCorse])

    dtCurrent = (durationCorse + durationFine)/NT
    dtCorse = min(dtCurrent, this.MAXDTCORSE)
    dtFine  = min(dtCurrent, this.MAXDTFINE)
    t_Nt = zeros(size(t_Dt))
    t_Nt(idCorse) = ceil(t_Dt(idCorse)/dtCorse)
    t_Nt(idFine) = ceil(t_Dt(idFine)/dtFine)
