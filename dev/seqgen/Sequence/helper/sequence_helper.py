import numpy as np
import math

#TODO: theoretical value could be zero, but result is slightly negative, missing the result.

def find_roots(coeffs):
    verbose = False
    x = np.roots(coeffs)  # x has 3 values. either all real, or one real and two complex
    #TODO: could be that x[i] is complex but imaginary part is almost zero
    nearly_real = abs(x.imag) < 1e-9  #TODO: make parameter
    if verbose:
        print("x.C=", x)
        print("x.R=", x.real)
        print("x.I=", x.imag)
        print(nearly_real)
    x[nearly_real] = x.real[nearly_real]  # remove imag
    if verbose:
        print("x.C=", x)
        print("x.R=", x.real)
        print("x.I=", x.imag)

    #x.imag[nearly_real] = 0
    # now we only take real roots
    if np.any(~np.isreal(x)):  # complex solution
        x = x[np.isreal(x)] # real (not complex pair) root
    if verbose:
        print("x=", x)
    return x

def solve_trapezoidal_sequence(b=None, g=None, G=None, D=None, n=None, e=None, **tols):
    """
    McNab -> pulse sequence

    VARIABLES
    b := bvalue
    g := gamma_0
    G := gradient
    D := Delta
    n := eta
    e := epsilon
    """

    # d := delta (=epsilon+eta)
    if e is not None and n is not None:
        d = e+n

    # solve for the one that was left defaulted to None
    if b is None:  # bvalue
        b = G**2*g**2*(d**2*(D - d/3) - (d*e**2)/6 + e**3/30)  #TODO: sqrt(2) ???
        return b  # straightforward

    elif g is None:  # gamma_0
        g = (30**0.5*b**0.5)/(G*(30*D*d**2 - 5*d*e**2 - 10*d**3 + e**3)**0.5)
        return g  # positive root

    elif G is None:  # gradient
        G = (30**0.5*b**0.5)/(g*(30*D*d**2 - 5*d*e**2 - 10*d**3 + e**3)**0.5)
        return G  # positive root

    elif D is None:  # Delta
        D = ((d**3/3 + (d*e**2)/6 - e**3/30)*G**2*g**2 + b)/(G**2*d**2*g**2)
        return D  # straightforward

    elif n is None:  # eta
        verbose = False
        # solve the equation for d, then extract d
        d = find_roots([1/3, -D, 1/6*e**2, b/(g*G)**2-e**3/30])  # 0 == a*x^3, b*x^2, c*x
        if verbose:
            print("d=",d)
        n = d - e
        if verbose:
            print("n=",n)
            print("D=",D)
            print("e=",e)
            print("D-2*e=",D-2*e)
        valid_range = np.all((n>=0, n<=D-2*e), 0)  # positive and fits definition of D
        if verbose:
            print("valid_range=",valid_range)
        # has to be real too?!
        return np.min(n[valid_range]) if np.any(valid_range) else None  # take the minimum

    elif e is None:  # epsilon
        # substitute d=n+e, rearrange, and solve for e
        e = find_roots([14/30, 5/6*n-D, n**2-2*D*n, 1/3*n**3+b/(g*G)**2-D*n**2])
        valid_range = np.all((e>=0, 2*e<=D-n), 0)  # positive and fits definition of D
        if np.isnan(e) or np.isinf(e):
            return None
        return np.min(e[valid_range]) if np.any(valid_range) else None  # take the minimum

    else:  # verify
        b_calc = solve_trapezoidal_sequence(b=None, g=g, G=G, D=D, n=n, e=e)
        if not math.isclose(b, b_calc, **tols):
            raise Exception("bvalue differs from nominal")  #TODO: add tols here
        return b - b_calc  # error

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
