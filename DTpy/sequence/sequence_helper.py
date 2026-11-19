import math

import numpy as np


#TODO: theoretical value could be zero, but result is slightly negative, missing the result.


def find_roots(coeffs, thresh = 1e-9):
    """Find roots.
    Expect 3 values. either all real, or one real and two complex.
    We remove the imaginary part if its magnitude is too small.
    We return real roots if there are any, otherwise we return the complex roots.
    """
    roots = np.roots(coeffs) # 3 values. either all real, or one real and two complex

    # remove imaginary part if it's too small
    #TODO: could be that x[i] is complex but imaginary part is almost zero
    nearly_real = abs(roots.imag) < thresh
    roots[nearly_real] = roots.real[nearly_real]

    return roots[np.isreal(roots)].real


def solve_trapezoidal_sequence(
        b: float | None = None,
        g: float | None = None,
        G: float | None = None,
        D: float | None = None,
        n: float | None = None,
        e: float | None = None,
        **tols,
) -> float | None:
    """
    Solve one parameter of the trapezoidal diffusion gradient equation.

    McNab -> pulse sequence

    Parameters use the notation from the seqgen notes:
    b: b-value
    g: gyromagnetic ratio
    G: gradient strength
    D: diffusion time Delta
    n: flat-top time eta
    e: ramp time epsilon
    """

    # solve for the one that was left defaulted to None

    if e is not None and n is not None:
        d = e + n  # d := delta = epsilon + eta

    if b is None:
        #TODO: sqrt(2) ???
        return G**2 * g**2 * (d**2 * (D - d / 3) - (d * e**2) / 6 + e**3 / 30)

    if g is None:
        # positive root
        return (30**0.5 * b**0.5) / (
            G * (30 * D * d**2 - 5 * d * e**2 - 10 * d**3 + e**3) ** 0.5
        )

    if G is None:
        # positive root
        return (30**0.5 * b**0.5) / (
            g * (30 * D * d**2 - 5 * d * e**2 - 10 * d**3 + e**3) ** 0.5
        )

    if D is None:
        return ((d**3 / 3 + (d * e**2) / 6 - e**3 / 30) * G**2 * g**2 + b) / (
            G**2 * d**2 * g**2
        )

    if n is None:
        # solve the equation for d, then extract d
        d = find_roots([1 / 3, -D, 1 / 6 * e**2, b / (g * G) ** 2 - e**3 / 30]) # 0 == a*x^3, b*x^2, c*x
        n = d - e
        valid_range = np.logical_and(n >= 0, n <= D - 2 * e) # positive and fits definition of D
        #TODO: has to be real too?!
        return np.min(n[valid_range]) if np.any(valid_range) else None

    if e is None:
        # substitute d=n+e, rearrange, and solve for e
        e = find_roots(
            [
                14 / 30,
                5 / 6 * n - D,
                n**2 - 2 * D * n,
                1 / 3 * n**3 + b / (g * G) ** 2 - D * n**2,
            ]
        )
        valid_range = np.logical_and(e >= 0, 2 * e <= D - n) # positive and fits definition of D
        finite = np.logical_and(~np.isnan(e), ~np.isinf(e))
        valid = np.logical_and(valid_range, finite)
        if np.isnan(e) or np.isinf(e):
            #TODO: is this already covered?
            return None
        return np.min(e[valid]) if np.any(valid) else None

    # all provided, so we simply verify
    b_calc = solve_trapezoidal_sequence(b=None, g=g, G=G, D=D, n=n, e=e)
    if not math.isclose(b, b_calc, **tols):
        raise ValueError("bvalue differs from nominal")  #include tols in error message
    return b - b_calc # return the error


def solve_motion_compensated_sequence(
        b: float | None = None,
        G: float | None = None,
        g: float | None = None,
        d1: float | None = None,
        D: float | None = None,
        t: float | None = None,
) -> float | None:
    """
    Solve the Welsh 2015 motion-compensated sequence equation where implemented.
    Welsh 2015 IEEE

    VARIABLES
        b := bvalue
        G := Gradient
        g := gamma
        d := delta_1
        D := Delta
        t := tau
    """

    if b is not None:
        raise NotImplementedError("Only b-value calculation is implemented")

    del1 = d1
    DELTA = D
    tau = t
    gamma = g
    Grad = G

    del2 = del1 * (DELTA - tau) / (DELTA - 2 * del1 + tau)

    terms = del2**2 * (DELTA + tau) - del2 * tau**2 / 6 + 49 * tau**3 / 60 - del2**3 / 3
    term1_num = (tau**2 - DELTA * tau + 2 * DELTA * del2) ** 3 - 12 * del2**3 * (DELTA + tau) ** 3
    term1_den = 12 * (DELTA + 2 * del2 - tau) ** 3
    term2_num = del2**2 * (DELTA + tau) ** 2 * (DELTA + 3 * del2)
    term2_den = (DELTA + 2 * del2 - tau) ** 2
    term3_num = -tau**2 * (tau**2 - DELTA * tau + 2 * DELTA * del2) + 3 * del2**2 * (
        DELTA + tau
    ) * (2 * DELTA + del2 + 2 * tau)
    term3_den = 3 * (DELTA + 2 * del2 - tau)
    return gamma**2 * Grad**2 * (
        terms + term1_num / term1_den + term2_num / term2_den + term3_num / term3_den
    )
