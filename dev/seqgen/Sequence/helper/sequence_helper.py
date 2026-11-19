def solve_motion_compensated_sequence(b=None, G=None, g=None, d1=None, D=None, t=None):
    """
    Welsh 2015 IEEE

    VARIABLES
        b := bvalue
        G := Gradient
        g := gamma
        d := delta_1
        D := Delta
        t := tau
    """

    #del2, DELTA, tau = sympy.symbol(('del2', 'DELTA', 'tau'))
    # eqn = sympy.Eq(terms, bfactor/gamma**2*Grad**2)
    # solx = sympy.solve(eqn, del2)  #TODO: how many?

    if b is None:

        del1 = d1
        DELTA = D
        tau = t
        gamma = g
        Grad = G

        # eqn (15)
        del2 = del1*(DELTA-tau)/(DELTA-2*del1+tau)

        # eqn (16) BUT: first line, last char should be small closed parenthesis, third line is + not x, last line needs a closing big parenthesis
        terms = del2**2*(DELTA+tau) - del2*tau**2/6 + 49*tau**3/60 - del2**3/3
        term1_num = (tau**2 - DELTA*tau + 2*DELTA*del2)**3 - 12*del2**3*(DELTA+tau)**3
        term1_den = 12*(DELTA + 2*del2 - tau)**3
        term2_num = del2**2*(DELTA+tau)**2*(DELTA+3*del2)
        term2_den = (DELTA+2*del2-tau)**2  # + or - ???
        term3_num = -tau**2*(tau**2 - DELTA*tau + 2*DELTA*del2) + 3*del2**2*(DELTA+tau)*(2*DELTA+del2+2*tau)
        term3_den = 3*(DELTA+2*del2-tau)
        b = gamma**2*Grad**2 * ( terms + term1_num/term1_den + term2_num/term2_den + term3_num/term3_den )

        return b
