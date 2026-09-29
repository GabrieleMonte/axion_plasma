"""Independent cross-check for tmatrix.py: numerical ODE integration of the
homogeneous solutions on a RESOLVED tanh profile (p.omega_p, half-width p.ell).
tmatrix (sharp steps) and this (tanh) must agree linearly as ell -> 0.

    [ -d_x^2 + V(x) - omega^2 ] psi = 0 ,   V(x) = wp^2(x)

psi_- : decaying/outgoing as x -> -inf, psi_+ : outgoing as x -> +inf.
"""
import numpy as np
from scipy.integrate import solve_ivp


def kappa(omega, V):
    """sqrt(omega^2 - V) with Im >= 0 (retarded branch)."""
    k = np.sqrt(complex(omega) ** 2 - V + 0j)
    return k if k.imag >= 0 else -k


def _integrate(V, omega, xa, xb, y0, n=4001):
    """Integrate psi'' = (V - omega^2) psi from xa to xb, renormalising to
    avoid overflow through evanescent layers. Returns x, psi, psi', and the
    accumulated log-scale so the two solutions can be put on a common footing."""
    xs = np.linspace(xa, xb, n)

    def rhs(x, y):
        return [y[1], (V(x) - omega ** 2) * y[0]]

    psi = np.empty(n, complex)
    dpsi = np.empty(n, complex)
    logs = np.zeros(n)
    y = np.array(y0, dtype=complex)
    psi[0], dpsi[0] = y
    acc = 0.0
    for i in range(n - 1):
        sol = solve_ivp(rhs, (xs[i], xs[i + 1]), y, rtol=1e-10, atol=1e-30,
                        method="DOP853")
        y = sol.y[:, -1]
        m = max(abs(y[0]), abs(y[1]), 1e-300)
        if m > 1e6 or m < 1e-6:          # renormalise, remember the factor
            y = y / m
            acc += np.log(m)
        psi[i + 1], dpsi[i + 1] = y
        logs[i + 1] = acc
    return xs, psi, dpsi, logs


def solutions(p, omega, x_lo=-40.0, x_hi=60.0, n=4001):
    V = lambda x: float(p.omega_p(np.array([x]))[0] ** 2)
    V_L = (p.w[0] * p.m_a) ** 2
    V_R = (p.w[-1] * p.m_a) ** 2
    kL, kR = kappa(omega, V_L), kappa(omega, V_R)

    # psi_- : e^{-i kL x} at the far left (Im kL >= 0 -> decays as x -> -inf)
    y0 = [1.0 + 0j, -1j * kL]
    xs, pm, dpm, lm = _integrate(V, omega, x_lo, x_hi, y0, n)

    # psi_+ : e^{+i kR x} at the far right, integrated leftwards
    y0 = [1.0 + 0j, 1j * kR]
    xs2, pp, dpp, lp = _integrate(V, omega, x_hi, x_lo, y0, n)
    # put on the same increasing-x grid
    xs2, pp, dpp, lp = xs2[::-1], pp[::-1], dpp[::-1], lp[::-1]

    return dict(x=xs, pm=pm, dpm=dpm, logm=lm, pp=pp, dpp=dpp, logp=lp,
                kL=kL, kR=kR)


def wronskian(S):
    """W(x) = psi_-' psi_+ - psi_- psi_+', up to the common log scale.
    Should be constant in x -- that is the test."""
    return (S["dpm"] * S["pp"] - S["pm"] * S["dpp"])
