"""Analytic solutions for a single uniform slab (CLAUDE.md section 3).

E_slab     exact E_y(x,t), one numerical k-integral      (3.1)
E_steady   the driven, packet-co-moving part in closed form, mu > m_a only (3.3)
E_slab_ss  the same driven part as a k-integral, kept to test E_steady

All take the parameter object p from axion_solver.Params; mu is the slab's
plasma frequency.
"""
import numpy as np
from scipy.special import wofz   # Faddeeva function w(z) = exp(-z^2) erfc(-iz)

# np.trapezoid is the numpy>=2.0 name; cosmo_env has numpy 1.25 (np.trapz)
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def E_slab(x, t, mu, p, K=12.0, NK=60001, chunk=64):
    """Exact E_y(x,t) in a uniform slab on the infinite line, starting from
    E = 0 with the axion already oscillating. Solves each Fourier mode k as a
    driven oscillator and integrates back to x; the integrand is regular
    everywhere (the resonance is a removable 0/0). K, NK set the k grid;
    x is processed `chunk` points at a time to bound the (x, k) array."""
    k = np.linspace(-K, K, NK)
    Wk = np.sqrt(k**2 + mu**2)                    # photon frequency of mode k
    pref = p.gB0a0 * np.sqrt(2*np.pi) * p.sigma_x / 2.0 / (2*np.pi)
    ker = np.zeros_like(k, dtype=complex)
    for s in (+1, -1):                    # the two halves e^{-+ i omega_a t} of cos
        d  = k - s*p.k_a                  # k measured from the packet's centre
        Om = s*p.omega_a + p.v_a*d        # frequency the axion drives mode k at
        A  = np.exp(-0.5*p.sigma_x**2 * d**2)   # packet's Gaussian k-spectrum
        D  = Wk**2 - Om**2                # detuning; D = 0 is the resonance
        N  = Om**2*(np.exp(-1j*Om*t) - np.cos(Wk*t)) + 1j*Om*Wk*np.sin(Wk*t)
        # N/D stays finite as D -> 0, but np.where evaluates BOTH branches,
        # so the division itself must also be guarded (CLAUDE.md section 8)
        reg = np.abs(D) > 1e-8
        Kf  = np.where(reg, N/np.where(reg, D, 1.0),
                       0.5j*(Om*t*np.exp(-1j*Om*t) + np.sin(Om*t)))   # D->0 limit
        ker += A*Kf
    # back to x: E(x) = pref * integral dk e^{ik(x-x0)} ker(k)
    xa = np.atleast_1d(np.asarray(x, float))
    out = np.empty(len(xa))
    for i in range(0, len(xa), chunk):
        u = xa[i:i+chunk][:, None] - p.x0
        out[i:i+chunk] = pref*trapezoid(np.exp(1j*k*u)*ker, k, axis=-1).real
    return out


def E_steady(x, t, mu, p):
    """The driven field that rides along with the packet, in closed form —
    no integral (CLAUDE.md 3.3). Only valid for mu > m_a: below that the slab
    radiates and neither branch of this expression is right (see 6.4)."""
    if mu <= p.m_a:
        raise ValueError("E_steady requires mu > m_a; see CLAUDE.md 6.4")
    g2 = p.gamma_a**2
    D2 = g2*(mu**2 - p.m_a**2)
    Dl = np.sqrt(D2 + 0j)                 # Delta: how far below cutoff we are
    u  = np.asarray(x) - p.x0
    xi = u - p.v_a*t                      # position relative to the packet centre
    ph = p.k_a*u - p.omega_a*t            # carrier phase
    s  = p.sigma_x

    def I0(xi):
        zp = s*Dl/np.sqrt(2) + xi/(s*np.sqrt(2))
        zm = s*Dl/np.sqrt(2) - xi/(s*np.sqrt(2))
        return np.pi/(2*Dl)*np.exp(-xi**2/(2*s**2))*(wofz(1j*zp) + wofz(1j*zm))

    # I1 = -i dI0/dxi, taken numerically: error ~h^2 = 1e-8, and differentiating
    # the Faddeeva functions by hand buys nothing
    h  = 1e-4
    I1 = -1j*(I0(xi+h) - I0(xi-h))/(2*h)
    G  = np.sqrt(2*np.pi)/s*np.exp(-xi**2/(2*s**2))   # the pure-Gaussian piece
    v, wa = p.v_a, p.omega_a
    tot = (wa**2 - v**2*Dl**2)*I0(xi) + 2*wa*v*I1 + v**2*G
    pre = p.gB0a0*s*g2/(2*np.sqrt(2*np.pi))
    return 2.0*np.real(pre*np.exp(1j*ph)*tot)         # 2 Re = both +/- halves


def E_slab_ss(x, t, mu, p, K=12.0, NK=60001, chunk=64):
    """Only the driven (co-moving, e^{-i Om t}) part of E_slab, as a k-integral.
    Exists to check E_steady; agrees with it to ~1e-10 for mu > m_a."""
    k = np.linspace(-K, K, NK)
    Wk = np.sqrt(k**2 + mu**2)
    pref = p.gB0a0*np.sqrt(2*np.pi)*p.sigma_x/2.0/(2*np.pi)
    ker = np.zeros_like(k, dtype=complex)
    for s in (+1, -1):
        d = k - s*p.k_a
        Om = s*p.omega_a + p.v_a*d
        ker += np.exp(-0.5*p.sigma_x**2*d**2)*Om**2*np.exp(-1j*Om*t)/(Wk**2 - Om**2)
    xa = np.atleast_1d(np.asarray(x, float))
    out = np.empty(len(xa))
    for i in range(0, len(xa), chunk):
        u = xa[i:i+chunk][:, None] - p.x0
        out[i:i+chunk] = pref*trapezoid(np.exp(1j*k*u)*ker, k, axis=-1).real
    return out
