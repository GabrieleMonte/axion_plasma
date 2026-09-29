"""Frequency-domain Green's-function solution for the axion wavepacket in the
piecewise-constant five-slab plasma (greens_derivation, all steps; equation
numbers refer to the PDF).

    [ -d_x^2 + V(x) - omega^2 ] E~(x,omega) = S(x,omega)          (9)
    S = gB0 [ omega^2 a~(x,omega) - i omega a(x,0) ]              (12)
    E~ = [ psi_+(x) int_<x psi_- S + psi_-(x) int_>x psi_+ S ]/W  (32)
    E_y(x,t) = (1/pi) Re int_0^inf e^{-i omega t} E~ domega       (38)

The only numerics is the final omega integral; everything else is closed form.
psi_pm and W come from tmatrix; the spatial integrals reduce to the two
antiderivatives (36)-(37), evaluated at region edges.

Everything is vectorised over omega: E_tilde takes the whole omega array and
returns (N_omega, N_x). The work per omega is ~300 small special-function
calls, so evaluating them one omega at a time is all Python overhead -- doing
the array at once is ~50x faster for the same arithmetic.

Numerical-stability rule used throughout: an exponential prefactor and an
erfc/erf that separately overflow and underflow are always combined into a
single exp(<summed exponent>) * w(<argument in the bounded half-plane>) call,
where w(z) = exp(-z^2) erfc(-iz) is the Faddeeva function (scipy wofz).
"""
import numpy as np
from scipy.special import wofz

from axion_solver import Params
from tmatrix import psi_states_vec

SQ2 = np.sqrt(2.0)


class StepParams(Params):
    """Params with the piecewise-constant density profile of the derivation
    (the tanh in the original runs was a numerical convenience, not the model)."""

    def omega_p(self, x):
        x = np.asarray(x, float)
        L = self.L_region
        idx = np.searchsorted([L, 2*L, 3*L, 4*L], x, side="right")
        return self.m_a * np.asarray(self.w)[idx]


# ----------------------------------------------------------------------------
# stable scaled error functions
# ----------------------------------------------------------------------------

def scaled_erfc(E, z):
    """e^E * erfc(z), without ever forming either factor alone.
    Uses erfc(z) = e^{-z^2} w(iz) where Re z >= 0 (w is bounded there) and
    erfc(z) = 2 - erfc(-z) otherwise. Masked assignment, not np.where: the
    rejected branch really does overflow."""
    E, z = np.broadcast_arrays(np.asarray(E, complex), np.asarray(z, complex))
    out = np.empty(E.shape, complex)
    m = z.real >= 0
    n = ~m
    out[m] = np.exp(E[m] - z[m]**2) * wofz(1j*z[m])
    out[n] = 2.0*np.exp(E[n]) - np.exp(E[n] - z[n]**2) * wofz(-1j*z[n])
    return out


def scaled_erf(E, z):
    """e^E * erf(z) = e^E - e^E erfc(z), stable in the same sense."""
    return np.exp(np.asarray(E, complex)) - scaled_erfc(E, z)


# ----------------------------------------------------------------------------
# Piece A -- the source (Eqs. 12, 17-21)
# ----------------------------------------------------------------------------

def a_tilde(x, omega, p):
    """One-sided time transform of the packet, a0 = 1 (Eq. 20). x and omega
    broadcast against each other; pass omega as a column to get (Nw, Nx)."""
    u = np.asarray(x, float) - p.x0
    om = np.asarray(omega, complex)
    s_x, v = p.sigma_x, p.v_a
    tot = 0.0
    for s in (1, -1):
        q = (om - s*p.omega_a) / v
        qb, ub = np.broadcast_arrays(q, u)
        Z = (qb*s_x - 1j*ub/s_x) / SQ2
        term = np.empty(Z.shape, complex)
        m = Z.imag >= 0                       # w(Z) bounded: evaluate directly
        n = ~m                                # w(Z) = 2 e^{-Z^2} - w(-Z)
        term[m] = np.exp(-0.5*(ub[m]/s_x)**2) * wofz(Z[m])
        term[n] = (2.0*np.exp(-0.5*(qb[n]*s_x)**2 + 1j*qb[n]*ub[n])
                   - np.exp(-0.5*(ub[n]/s_x)**2) * wofz(-Z[n]))
        tot = tot + np.exp(1j*s*p.k_a*ub) * term
    return (s_x/(2.0*v)) * np.sqrt(np.pi/2.0) * tot


def a_init(x, p):
    """a(x, 0), a0 = 1 (Eq. 21)."""
    u = np.asarray(x, float) - p.x0
    return np.exp(-0.5*(u/p.sigma_x)**2) * np.cos(p.k_a*u)


def S_tilde(x, omega, p):
    """Transformed source including the initial data (Eq. 12)."""
    om = np.asarray(omega, complex)
    return p.gB0a0 * (om**2 * a_tilde(x, om, p) - 1j*om * a_init(x, p))


# ----------------------------------------------------------------------------
# Piece B -- the two spatial antiderivatives (Eqs. 36-37), scaled
# ----------------------------------------------------------------------------

def _F1(u, lam, alpha, beta, C):
    """e^C * antiderivative of e^{lam u} erfc(alpha u + beta)  (Eq. 36).
    C carries the external Gaussian -q^2 sigma^2/2 so every exponent is tame."""
    z = alpha*u + beta
    D = (lam/alpha) * (lam/(4.0*alpha) - beta)
    return (scaled_erfc(C + lam*u, z)
            + scaled_erf(C + D, z - lam/(2.0*alpha))) / lam


def _gauss_legendre(f, a, b, n_panels, order=10):
    """Composite Gauss-Legendre quadrature of f on [a, b] (fallback path)."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_panels + 1)
    mid = 0.5*(edges[1:] + edges[:-1])
    half = 0.5*(edges[1:] - edges[:-1])
    pts = (mid[:, None] + half[:, None]*xg[None, :]).ravel()
    wts = (half[:, None]*wg[None, :]).ravel()
    return np.sum(f(pts) * wts)


def dF1(u1, u2, lam, alpha, beta, C):
    """e^C * int_{u1}^{u2} e^{lam u} erfc(alpha u + beta) du. All arguments
    broadcast. Where lam*(u2-u1) is too small the antiderivative difference
    loses every digit to cancellation, so those entries are integrated
    directly instead; lam does pass through 0 (at omega = omega_a in the
    resonant slab, exactly), so this branch is not decorative."""
    u1 = np.asarray(u1, float)
    u2 = np.asarray(u2, float)
    lam = np.asarray(lam, complex)
    out = _F1(u2, lam, alpha, beta, C) - _F1(u1, lam, alpha, beta, C)
    span = np.abs(u2 - u1)
    tiny = np.broadcast_to(np.abs(lam)*span <= 1e-4, out.shape)
    if tiny.any():
        b = np.broadcast_to
        lamb, betab, Cb = b(lam, out.shape), b(beta, out.shape), b(C, out.shape)
        u1b, u2b = b(u1, out.shape), b(u2, out.shape)
        for idx in zip(*np.nonzero(tiny)):
            lo, hi, lm = u1b[idx], u2b[idx], lamb[idx]
            bt, Cc = betab[idx], Cb[idx]
            rate = abs(lm) + 2.0*alpha*alpha*abs(hi - lo) + 2.0*abs(alpha*bt.imag)
            n = min(max(16, int(abs(hi - lo)*rate/3.0) + 1), 4000)
            out[idx] = _gauss_legendre(
                lambda u: scaled_erfc(Cc + lm*u, alpha*u + bt), lo, hi, n)
    return out


def dF2(u1, u2, lam, sigma_x):
    """int_{u1}^{u2} e^{-u^2/2 sigma^2} e^{lam u} du   (Eq. 37)."""
    lam = np.asarray(lam, complex)
    E = 0.5*(lam*sigma_x)**2
    z1 = (np.asarray(u1, float) - lam*sigma_x**2) / (sigma_x*SQ2)
    z2 = (np.asarray(u2, float) - lam*sigma_x**2) / (sigma_x*SQ2)
    return sigma_x*np.sqrt(np.pi/2.0)*(scaled_erf(E, z2) - scaled_erf(E, z1))


# ----------------------------------------------------------------------------
# Piece C -- assemble E~(x, omega)  (Eqs. 30-35)
# ----------------------------------------------------------------------------

def _region_integral(p, om, kj, Aj, P, Q, ua, ub):
    """int psi(x') S(x') dx' over u in [ua, ub] inside one region, where
    psi = P e^{i k zeta} + Q e^{-i k zeta}, zeta = u + x0 - Aj (Eq. 33).

    om, kj, P, Q are columns of shape (Nw, 1); ua, ub are scalars (a whole
    region) or rows of shape (Nu,) (a partial region, one edge per output
    point). The result has shape (Nw, Nu)."""
    s_x, v = p.sigma_x, p.v_a
    alpha = -1.0/(s_x*SQ2)
    N = (s_x/(2.0*v)) * np.sqrt(np.pi/2.0)
    cp = P * np.exp(+1j*kj*(p.x0 - Aj))       # coefficient of e^{+i k u}
    cm = Q * np.exp(-1j*kj*(p.x0 - Aj))       # coefficient of e^{-i k u}
    tot = 0.0
    for s in (1, -1):
        q = (om - s*p.omega_a) / v
        C = -0.5*(q*s_x)**2
        beta = -1j*q*s_x/SQ2
        for coef, sgn in ((cp, +1.0), (cm, -1.0)):
            # In the outer regions one of P, Q is identically zero (the
            # solution is a single exponential there). Its partner integral
            # can overflow when the limit is far out, and 0*inf = nan would
            # poison the sum, so skip the branch rather than multiply it away.
            if not np.any(coef):
                continue
            lam1 = 1j*(s*p.k_a + q + sgn*kj)          # a~ part      -> Eq. 36
            tot = tot + om**2 * N * coef * dF1(ua, ub, lam1, alpha, beta, C)
            lam2 = 1j*(s*p.k_a + sgn*kj)              # a(x,0) part  -> Eq. 37
            tot = tot - 0.5j*om * coef * dF2(ua, ub, lam2, s_x)
    return p.gB0a0 * tot


def _E_tilde_block(x, om, p, x_L, x_R):
    """E~ for one block of omega values (no chunking); shape (Nw, Nx)."""
    Nw, Nx = len(om), len(x)
    L = p.L_region
    sm, sp, W, k = psi_states_vec(p, om)
    anchors = [L, L, 2*L, 3*L, 4*L]           # region 1 is anchored at X1 too
    key = [0, 0, 1, 2, 3]                     # which interface each region uses

    Pm = np.empty((5, Nw), complex); Qm = np.empty((5, Nw), complex)
    Pp = np.empty((5, Nw), complex); Qp = np.empty((5, Nw), complex)
    for j in range(5):
        ik = 1j*k[j]
        ym, yp = sm[key[j]], sp[key[j]]
        Pm[j] = 0.5*(ym[:, 0] + ym[:, 1]/ik); Qm[j] = 0.5*(ym[:, 0] - ym[:, 1]/ik)
        Pp[j] = 0.5*(yp[:, 0] + yp[:, 1]/ik); Qp[j] = 0.5*(yp[:, 0] - yp[:, 1]/ik)
    # exact endpoint forms: any epsilon in the growing coefficient would be
    # multiplied by e^{+|kappa| * 90} in the outer regions
    Pm[0], Qm[0] = 0.0, 1.0
    Pp[4], Qp[4] = 1.0, 0.0

    ua = np.array([x_L, L, 2*L, 3*L, 4*L]) - p.x0     # region lower u-edges
    ub = np.array([L, 2*L, 3*L, 4*L, x_R]) - p.x0     # region upper u-edges

    omc = om[:, None]
    # Only regions strictly LEFT of the output point need the full psi_-
    # integral, and only those strictly RIGHT need the full psi_+ one. So
    # Im_full[4] and Ip_full[0] are never used -- and must not be computed:
    # psi_- grows to the right, so int_{X4}^{x_R} psi_- S truly diverges.
    Im_full = np.zeros((5, Nw), complex)
    Ip_full = np.zeros((5, Nw), complex)
    for j in range(5):
        kc = k[j][:, None]
        if j <= 3:
            Im_full[j] = _region_integral(p, omc, kc, anchors[j], Pm[j][:, None],
                                          Qm[j][:, None], ua[j], ub[j])[:, 0]
        if j >= 1:
            Ip_full[j] = _region_integral(p, omc, kc, anchors[j], Pp[j][:, None],
                                          Qp[j][:, None], ua[j], ub[j])[:, 0]
    Lcum = np.zeros((5, Nw), complex)         # sum over regions strictly left
    Lcum[1:] = np.cumsum(Im_full[:-1], axis=0)
    Rcum = np.zeros((5, Nw), complex)         # sum over regions strictly right
    Rcum[:-1] = np.cumsum(Ip_full[:0:-1], axis=0)[::-1]

    out = np.empty((Nw, Nx), complex)
    ridx = np.searchsorted(np.array([L, 2*L, 3*L, 4*L]), x, side="left")
    for J in range(5):
        m = ridx == J
        if not m.any():
            continue
        us = x[m] - p.x0
        zeta = x[m] - anchors[J]
        kc = k[J][:, None]
        left = Lcum[J][:, None] + _region_integral(
            p, omc, kc, anchors[J], Pm[J][:, None], Qm[J][:, None], ua[J], us)
        right = Rcum[J][:, None] + _region_integral(
            p, omc, kc, anchors[J], Pp[J][:, None], Qp[J][:, None], us, ub[J])
        ep = np.exp(+1j*kc*zeta)
        em = np.exp(-1j*kc*zeta)
        psim = Pm[J][:, None]*ep + Qm[J][:, None]*em
        psip = Pp[J][:, None]*ep + Qp[J][:, None]*em
        out[:, m] = (psip*left + psim*right) / W[:, None]
    return out


def E_tilde(x, omega, p, x_L=-90.0, x_R=140.0, block=400000):
    """E~(x, omega) via the Green's function (Eq. 32). omega may be a scalar
    or an array; the result is (Nx,) or (N_omega, Nx). x must lie inside
    [x_L, x_R]; that truncation is exact by causality for the times of
    interest (derivation section 5).

    `block` caps N_omega * N_x per internal batch, bounding peak memory."""
    x = np.atleast_1d(np.asarray(x, float))
    scalar = np.asarray(omega).ndim == 0
    om = np.atleast_1d(np.asarray(omega, complex)).copy()
    # the (P, Q) split degenerates at kappa_j = 0; a 1e-6 nudge changes E~ by
    # ~1e-3 right at the cusp and nothing measurable once integrated
    for V in np.unique(np.asarray([(w*p.m_a)**2 for w in p.w])):
        bad = np.abs(om**2 - V) < 1e-11
        if bad.any():
            om[bad] += 1e-6
    nb = max(1, int(block // max(len(x), 1)))
    out = np.empty((len(om), len(x)), complex)
    for a in range(0, len(om), nb):
        b = min(a + nb, len(om))
        out[a:b] = _E_tilde_block(x, om[a:b], p, x_L, x_R)
    return out[0] if scalar else out


# ----------------------------------------------------------------------------
# Piece D -- back to the time domain  (Eq. 38)
# ----------------------------------------------------------------------------

def omega_grid(p, d_base=1e-3, w_fine=3.0, w_max=150.0, growth=1.02,
               win_cut=0.35, d_cut=2.5e-4, n_sqrt=600, d_kap=2e-3,
               win_a=0.25, d_a=5e-5):
    """Quadrature nodes for Eq. (38) on the shifted contour. Three kinds of
    fast structure set the local density (all measured; section 7.3 of the
    derivation anticipated only the first and the last):

    1. Propagation phases e^{i kappa(omega) D}, D up to ~200. Near a cutoff
       omega = w_j, kappa = sqrt(omega^2 - w_j^2) has infinite slope, so a
       mesh uniform in omega cannot resolve the phase however fine it is.
       Nodes are placed uniform in KAPPA instead, omega = sqrt(w_j^2 + kap^2)
       on both sides, n_sqrt of them at spacing d_kap -- this is the only
       grading that works at a square-root branch point.
    2. The driven term carries e^{i(omega-omega_a)u/v_a}, rate u/v_a up to
       ~1400 at the far edge, weighted by the drive Gaussian of width
       v_a/sigma_x = 0.067 -> d_a within +-win_a of omega_a.
    3. The switch-on tail, |E~| ~ 1/omega^3 and smooth -> a geometrically
       stretched grid from w_fine out to w_max.
    """
    nodes = [np.arange(0.0, w_fine, d_base)]
    pt, h = w_fine, d_base
    tail = [pt]
    while pt < w_max:
        h = min(h*growth, 0.25)
        pt += h
        tail.append(pt)
    nodes.append(np.array(tail))

    kapg = d_kap*np.arange(1, n_sqrt + 1)              # uniform in kappa
    for wc in np.unique(np.asarray(p.w, float))*p.m_a:
        nodes.append(np.sqrt(wc**2 + kapg**2))         # above cutoff
        below = wc**2 - kapg**2
        nodes.append(np.sqrt(below[below > 0.0]))      # below cutoff
        nodes.append(wc + np.arange(-win_cut, win_cut + d_cut/2, d_cut))
    nodes.append(p.omega_a + np.arange(-win_a, win_a + d_a/2, d_a))

    om = np.unique(np.concatenate(nodes))
    return om[om >= 0.0]


def _phi12(theta):
    """phi1 = int_0^1 e^{-i theta tau} dtau, phi2 = int_0^1 tau e^{-i ...},
    stable for small theta (series) and large (closed form)."""
    theta = np.asarray(theta, float)
    small = np.abs(theta) < 1e-3
    th = np.where(small, 1.0, theta)          # dummy to avoid 0/0
    e = np.exp(-1j*theta)
    phi1 = np.where(small,
                    1.0 - 0.5j*theta - theta**2/6.0 + 1j*theta**3/24.0,
                    (1.0 - e) / (1j*th))
    phi2 = np.where(small,
                    0.5 - 1j*theta/3.0 - theta**2/8.0 + 1j*theta**3/30.0,
                    (phi1 - e) / (1j*th))
    return phi1, phi2


def filon_weights(om, t):
    """Complex weights w_n(t) with  int F(omega) e^{-i omega t} domega
    ~= sum_n w_n(t) F(om_n):  piecewise-linear-in-F Filon rule, so the
    e^{-i omega t} oscillation is integrated exactly and accuracy does not
    degrade at late t. Returns shape (len(t), len(om))."""
    om = np.asarray(om, float)
    t = np.atleast_1d(np.asarray(t, float))
    h = np.diff(om)
    theta = t[:, None] * h[None, :]
    phi1, phi2 = _phi12(theta)
    base = np.exp(-1j*t[:, None]*om[None, :-1]) * h[None, :]
    wts = np.zeros((len(t), len(om)), complex)
    wts[:, :-1] += base * (phi1 - phi2)
    wts[:, 1:] += base * phi2
    return wts


def E_y(x, t, p, om=None, x_L=-90.0, x_R=140.0, Et=None, eps=0.015,
        progress=False, block=400000):
    """E_y(x, t) by Eq. (38), integrated along the SHIFTED contour
    Im omega = eps rather than the real axis. On the real axis the truncated
    source puts terms e^{i q_s u_R} into E~ that oscillate in omega with
    period 2 pi v_a/(x_R - x0) ~ 0.005 -- unresolvable. At Im omega = eps
    every such endpoint term carries e^{-eps u_R/v_a} (~1e-9 at eps=0.015)
    and E~ is smooth on the physical scale v_a/sigma_x. The price is the
    factor e^{eps t} restoring the contour shift, which multiplies the
    quadrature error by ~50 at t = 260 -- still inside budget.

    Builds E~ once on the grid (or reuses a precomputed Et of shape
    (len(om), len(x))), then every output time is one weighted sum.
    Returns (E, om, Et) so Et can be cached."""
    x = np.atleast_1d(np.asarray(x, float))
    t = np.atleast_1d(np.asarray(t, float))
    if om is None:
        om = omega_grid(p)
    if Et is None:
        if progress:
            print(f"  E_tilde: {len(om)} omega x {len(x)} x", flush=True)
        Et = E_tilde(x, om + 1j*eps, p, x_L, x_R, block=block)
    # chunk over t: the weight matrix is Nt x Nomega and would otherwise
    # allocate ~1 GB of temporaries for a 601 x 34000 run
    E = np.empty((len(t), len(x)))
    for a in range(0, len(t), 64):
        b = min(a + 64, len(t))
        wts = filon_weights(om, t[a:b])
        E[a:b] = (wts @ Et).real / np.pi * np.exp(eps*t[a:b])[:, None]
    return E, om, Et


def wronskian_scan(p, om_lo=0.0, om_hi=10.0, d_om=1e-3):
    """|W(omega)| on the real axis -- run BEFORE trusting the inversion:
    a near-zero would mean a quasinormal mode sitting on the contour."""
    oms = np.arange(om_lo, om_hi + d_om/2, d_om)
    return oms, np.abs(psi_states_vec(p, oms)[2])
