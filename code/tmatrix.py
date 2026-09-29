"""
Exact psi_pm for the piecewise-constant profile, by transfer matrix
(greens_derivation section 4).

Regions (four interfaces at X_i = i*L, i=1..4):
    1 : (-inf, L]   wp = 1.2      2 : [L,2L]    wp = 1.1
    3 : [2L,3L]     wp = 1.0      4 : [3L,4L]   wp = 0.9
    5 : [4L, +inf)  wp = 0.8

Carry the state vector (psi, psi'), which is continuous across every interface,
so the only thing that changes at an interface is kappa. Within a slab of width
d the propagator is the elementary

    M(k,d) = [[ cos(kd), sin(kd)/k ], [ -k sin(kd), cos(kd) ]]

written below as sin(kd)/k = d*sinc(kd) and -k sin(kd) = -k^2 d * sinc(kd),
so nothing ever divides by k: the k -> 0 point (omega = 1.0 exactly, inside
the band of interest) is regular by construction.

psi_states  works on one omega and returns dicts (readable, used in tests).
psi_states_vec  does the same for a whole array of omega at once; that is what
the solver calls, since the per-omega Python overhead otherwise dominates.
"""
import numpy as np


def kap(omega, V):
    """kappa = sqrt(omega^2 - V) on the retarded branch, Im >= 0."""
    k = np.sqrt(complex(omega) ** 2 - V + 0j)
    return k if k.imag >= 0 else -k


def kap_vec(omega, V):
    """kap for an array of omega."""
    k = np.sqrt(np.asarray(omega, complex) ** 2 - V)
    return np.where(k.imag >= 0, k, -k)


def _sinc(z):
    """sin(z)/z, series near 0. Works for scalars and arrays."""
    z = np.asarray(z, complex)
    small = np.abs(z) < 1e-4
    safe = np.where(small, 1.0, z)          # dummy, avoids 0/0 in the unused branch
    z2 = z * z
    return np.where(small, 1.0 - z2 / 6.0 + z2 * z2 / 120.0, np.sin(safe) / safe)


def M(k, d):
    """Propagator of (psi, psi') across a uniform slab of width d."""
    c = np.cos(k * d)
    s = d * _sinc(k * d)          # = sin(kd)/k, finite at k = 0
    return np.array([[c, s], [-k * k * s, c]], dtype=complex)


def _step(k, d, psi, dpsi):
    """Apply M(k,d) to (psi, dpsi) elementwise -- the vectorised form of M."""
    c = np.cos(k * d)
    s = d * _sinc(k * d)
    return c * psi + s * dpsi, -k * k * s * psi + c * dpsi


def psi_states_vec(p, omega):
    """psi_- and psi_+ for an array of omega.

    Returns (sm, sp, W, k) with sm[i], sp[i] of shape (Nw, 2) holding
    (psi, psi') at interface X_{i+1} for i = 0..3, W of shape (Nw,), and
    k of shape (5, Nw)."""
    om = np.atleast_1d(np.asarray(omega, complex))
    L = p.L_region
    k = np.stack([kap_vec(om, (w * p.m_a) ** 2) for w in p.w])     # (5, Nw)

    # psi_- : e^{-i k1 (x-L)} in region 1, so at X1 the state is (1, -i k1)
    psi = np.ones_like(om)
    dpsi = -1j * k[0]
    sm = [np.stack([psi, dpsi], axis=-1)]
    for j in (1, 2, 3):                       # cross slabs 2,3,4 rightwards
        psi, dpsi = _step(k[j], L, psi, dpsi)
        sm.append(np.stack([psi, dpsi], axis=-1))

    # psi_+ : e^{+i k5 (x-4L)} in region 5, so at X4 the state is (1, +i k5)
    psi = np.ones_like(om)
    dpsi = 1j * k[4]
    sp = [None, None, None, np.stack([psi, dpsi], axis=-1)]
    for i, j in zip((2, 1, 0), (3, 2, 1)):    # cross slabs 4,3,2 leftwards
        psi, dpsi = _step(k[j], -L, psi, dpsi)
        sp[i] = np.stack([psi, dpsi], axis=-1)

    W = sm[3][:, 1] * sp[3][:, 0] - sm[3][:, 0] * sp[3][:, 1]
    return sm, sp, W, k


def psi_states(p, omega):
    """Single-omega version, returning the state vectors as dicts keyed by
    interface position (kept for readability in tests and figures)."""
    L = p.L_region
    sm, sp, W, k = psi_states_vec(p, omega)
    X = [L, 2 * L, 3 * L, 4 * L]
    states_m = {X[i]: sm[i][0] for i in range(4)}
    states_p = {X[i]: sp[i][0] for i in range(4)}
    return states_m, states_p, W[0], [kk[0] for kk in k]


def wronskian(p, omega):
    return psi_states(p, omega)[2]
