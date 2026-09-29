"""Validation suite for the frequency-domain Green's-function solver
(code/greens_solver.py), following the gates in BUILD_greens.md.

Usage:
    python scripts/validate_greens.py          # everything (~20 min)
    python scripts/validate_greens.py --fast   # skip the time-domain gates (~3 min)

Every gate prints PASS/FAIL with its measured number and its target.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
DATA = ROOT / "data"

from axion_solver import Params, solve                     # noqa: E402
from tmatrix import psi_states, wronskian, kap, M          # noqa: E402
from greens_solver import (a_tilde, dF1, dF2, scaled_erfc, E_tilde, S_tilde,
                           StepParams, omega_grid, filon_weights, E_y,
                           wronskian_scan, SQ2)            # noqa: E402
from analytic import E_slab                                # noqa: E402

FAST = "--fast" in sys.argv
p5 = Params()
L = p5.L_region
RESULTS = []


def gate(name, value, target, larger_ok=False):
    ok = value <= target if not larger_ok else value >= target
    RESULTS.append((name, value, target, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2e}  (target {target:.0e})")
    return ok


def gl_panels(f, a, b, n_panels, order=16):
    xg, wg = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_panels + 1)
    mid = 0.5*(edges[1:] + edges[:-1])[:, None]
    half = 0.5*(edges[1:] - edges[:-1])[:, None]
    return np.sum(f((mid + half*xg[None, :]).ravel())*(half*wg[None, :]).ravel())


# ===================== Step 3: transfer matrix =====================
print("-- tmatrix (homogeneous solutions) --")
worst = 0.0
for w0 in (0.8, 1.2):
    for om in (0.3, 1.05, 2.7):
        k = kap(om, (w0*p5.m_a)**2)
        W = wronskian(Params(w=(w0,)*5), om)
        Wex = -2j*k*np.exp(-3j*k*L)        # exact for a uniform medium
        worst = max(worst, abs(W - Wex)/abs(Wex))
gate("uniform-medium Wronskian", worst, 1e-13)
gate("M regular at kappa=0 (|M01 - L|)", abs(M(0.0, L)[0, 1] - L), 1e-14)

# ===================== Piece A: the source =========================
print("-- Piece A: a_tilde --")


def a_of_t(x, t):
    u = x - p5.x0
    xi = u - p5.v_a*t
    return np.exp(-0.5*(xi/p5.sigma_x)**2)*np.cos(p5.k_a*u - p5.omega_a*t)


worst = 0.0
for x in (-5.0, 0.5, 5.0, 20.0, 100.0):
    for om in (0.3, p5.omega_a, 7.3, 25.0):
        T = max(0.0, (x - p5.x0)/p5.v_a) + 14*p5.sigma_x/p5.v_a
        n = max(32, int(T*(abs(om) + 2.1)))
        ref = gl_panels(lambda t: np.exp(1j*om*t)*a_of_t(x, t), 0.0, T, n)
        worst = max(worst, abs(a_tilde(np.array([x]), om, p5)[0] - ref)
                    / max(abs(ref), 1e-3))
gate("a_tilde vs time quadrature", worst, 2e-10)

# ===================== Piece B: the primitives =====================
print("-- Piece B: spatial antiderivatives --")
s_x = p5.sigma_x
alpha = -1.0/(s_x*SQ2)
worst = 0.0
for om in (0.35, 0.97, 3.0):
    for s in (1, -1):
        q = (om - s*p5.omega_a)/p5.v_a
        C = -0.5*(q*s_x)**2
        beta = -1j*q*s_x/SQ2
        for kv in (0.61, 1j*0.66):
            for sgn in (+1, -1):
                lam = 1j*(s*p5.k_a + q + sgn*kv)
                for (u1, u2) in ((-91.5, 3.4), (-8.0, 8.0)):
                    val = dF1(u1, u2, lam, alpha, beta, C)
                    f = lambda u: scaled_erfc(C + lam*u, alpha*u + beta)
                    rate = abs(lam.imag) + 2*abs(alpha*np.imag(beta)) + 1.0
                    ref = gl_panels(f, u1, u2, max(64, int((u2 - u1)*rate/2.0)))
                    worst = max(worst, abs(val - ref)/max(abs(ref), 1e-6))
gate("dF1 vs quadrature", worst, 1e-12)

worst = 0.0
for lam in (1j*0.7, 1j*0.1 - 0.66, 0.3j + 0.02):
    for (u1, u2) in ((-91.5, 3.4), (-8.0, 8.0)):
        val = dF2(u1, u2, lam, s_x)
        f = lambda u: np.exp(-0.5*(u/s_x)**2 + lam*u)
        ref = gl_panels(f, u1, u2, max(64, int((u2 - u1)*(abs(lam) + 1))))
        worst = max(worst, abs(val - ref)/max(abs(ref), 1e-8))
gate("dF2 vs quadrature", worst, 1e-12)

ok = True
u = np.linspace(-91.5, 138.5, 401)
for om in (30.0, 120.0, 400.0):
    va = a_tilde(u + p5.x0, om, p5)
    ok = ok and bool(np.all(np.isfinite(va)))
    ok = ok and bool(np.all(np.isfinite(E_tilde(np.linspace(-3, 55, 30), om, p5))))
gate("finite at omega = 30/120/400", 0.0 if ok else 1.0, 0.5)

# ===================== Piece C: assembly ===========================
print("-- Piece C: E_tilde --")


def E_uniform_kspace(x, omega, w0, p, K=14.0, NK=120001):
    u = np.atleast_1d(np.asarray(x, float)) - p.x0
    k = np.linspace(-K, K, NK)
    Shat = np.zeros(len(k), complex)
    for s in (1, -1):
        A = (p.sigma_x*np.sqrt(2*np.pi)/2.0)*np.exp(-0.5*p.sigma_x**2*(k - s*p.k_a)**2)
        Om = s*p.omega_a + p.v_a*(k - s*p.k_a)
        Shat += A*omega*Om/(omega - Om)
    ker = 1j*p.gB0a0*Shat/(k**2 + (w0*p.m_a)**2 - omega**2)
    return np.array([np.trapz(np.exp(1j*k*uu)*ker, k) for uu in u])/(2*np.pi)


xs7 = np.array([-20.0, -3.0, 1.5, 7.0, 12.0, 22.0, 40.0])
worst = 0.0
for w0 in (1.0, 1.15, 0.85):
    for om in (1.35 + 0.3j, 0.6 + 0.3j):
        pu = Params(w=(w0,)*5)
        ref = E_uniform_kspace(xs7, om, w0, pu)
        worst = max(worst, np.max(np.abs(E_tilde(xs7, om, pu) - ref))
                    / np.max(np.abs(ref)))
gate("uniform E_tilde vs k-space (complex omega)", worst, 1e-10)

# The same check on the contour the inversion actually uses. E_tilde loses
# accuracy in a DEEPLY EVANESCENT medium at small Im(omega): psi_- grows like
# e^{|kappa|x} while the source only decays like e^{-eps x/v_a}, so
# int psi_- S reaches ~1e7 while E~ itself is ~1e-5, and psi_+(x), that
# integral and W are formed as separate factors. See CLAUDE.md section 7.
print("-- E_tilde on the working contour (evanescent stress test) --")
for w0, tgt in ((0.8, 1e-9), (1.0, 1e-9), (1.2, 1e-4), (1.3, 5e-3)):
    pu = Params(w=(w0,)*5)
    ref = E_uniform_kspace(xs7, 1.0 + 0.015j, w0, pu)
    err = np.max(np.abs(E_tilde(xs7, 1.0 + 0.015j, pu) - ref))/np.max(np.abs(ref))
    gate(f"  uniform w0={w0}, Im(omega)=0.015", err, tgt)

h = 5e-3
worst = 0.0
for om in (0.5, 0.85, 1.0, 1.35, 2.5):
    pts = np.array([-5.0, 0.5, 1.5, 3.0, 1.5*L, 2.5*L, 3.5*L, 4.5*L, 30.0, 110.0])
    st = pts[None, :] + h*np.array([-2, -1, 0, 1, 2])[:, None]
    E = np.array([E_tilde(row, om, p5) for row in st])
    om_eff = om + 1e-6 if min(abs(om**2 - np.array(p5.w)**2)) < 1e-11 else om
    S = S_tilde(pts, om_eff, p5)
    V = p5.m_a**2*np.select([pts <= L, pts <= 2*L, pts <= 3*L, pts <= 4*L, True],
                            [1.44, 1.21, 1.0, 0.81, 0.64])
    d2E = (-E[0] + 16*E[1] - 30*E[2] + 16*E[3] - E[4])/(12*h**2)
    resid = -d2E + (V - om_eff**2)*E[2] - S
    scale = np.abs((V - om_eff**2)*E[2]) + np.abs(d2E) + np.abs(S)
    keep = np.abs(E[2]) > 1e-6*np.abs(E[2]).max()     # below that, d2E is noise
    worst = max(worst, np.max(np.abs(resid[keep])/scale[keep]))
    # 4th-order FD at h=5e-3 against a field spanning ~10 decades in x:
    # 2e-6 is the stencil's own noise floor, not the solution's error
gate("ODE residual (FD, where field resolvable)", worst, 1e-5)

worst = 0.0
for om in (0.5, 0.97, 1.35):
    for Xi in (L, 2*L, 3*L, 4*L):
        e = E_tilde(np.array([Xi - 1e-9, Xi, Xi + 1e-9]), om, p5)
        worst = max(worst, max(abs(e[1] - e[0]), abs(e[2] - e[1]))/abs(e[1]))
gate("interface continuity", worst, 1e-7)

# ===================== Hazard 3: Wronskian scan ====================
print("-- Wronskian real-axis scan --")
oms, absW = wronskian_scan(p5, 0.0, 10.0, 1e-3)
k1 = np.array([abs(kap(w, 1.44)) for w in oms])
k5 = np.array([abs(kap(w, 0.64)) for w in oms])
r = absW/np.maximum(k1 + k5, 0.3)
print(f"  min |W|/(k1+k5) = {r.min():.3f} at omega = {oms[r.argmin()]:.3f}")
gate("no real-axis Wronskian zeros (min ratio)", r.min(), 0.2, larger_ok=True)

if FAST:
    print("\n--fast: skipping time-domain gates D0 and D")
else:
    # ===================== D0: uniform end-to-end ==================
    print("-- D0: uniform medium, E_y(x,t) vs analytic.E_slab --")
    xs = np.arange(-3.0, 55.0 + 1e-9, 1.0)
    ts = np.array([0.0, 40.0, 100.0, 250.0])
    # the looser targets at 1.2/1.3 are the evanescent-cancellation limit
    # measured above, not a quadrature tolerance -- refining the omega grid
    # does not move them (verified over 5 grid parameters spanning 16x)
    for w0, tgt in ((0.8, 1e-4), (1.0, 1e-4), (1.2, 1e-3), (1.3, 3e-3)):
        pu = Params(w=(w0,)*5)
        t0 = time.time()
        E, om, _ = E_y(xs, ts, pu, om=omega_grid(pu, w_max=2000.0))
        ref = np.array([E_slab(xs, tt, w0*pu.m_a, pu, K=14.0, NK=240001)
                        for tt in ts])
        err = np.abs(E - ref).max()/np.abs(ref).max()
        print(f"  (w0={w0}: N_omega={len(om)}, build {time.time()-t0:.0f}s)")
        gate(f"uniform w0={w0} vs E_slab, t<=250", err, tgt)

    # ===================== D: the five-slab problem ================
    print("-- D: full profile vs step-profile time-domain run --")
    ref_file = DATA / "run_step_dx0.005.npz"
    if not ref_file.exists():
        print("  reference data/run_step_dx0.005.npz missing -- run the")
        print("  step-profile solve first (see CLAUDE.md section 7). SKIPPED.")
    else:
        d = np.load(ref_file)
        xn, tn, En = d["x"], d["t"], d["E"]
        m = (xn >= -3.0) & (xn <= 55.0)
        xs = xn[m][::4]                       # every 0.08
        t0 = time.time()
        E, om, Et = E_y(xs, tn, p5)
        print(f"  (N_omega={len(om)}, Nx={len(xs)}, build {time.time()-t0:.0f}s)")
        ref = En[:, m][:, ::4]
        scale = np.abs(ref).max()
        gate("full profile vs TD (max over x,t)", np.abs(E - ref).max()/scale, 1e-3)

        # save the field (and spectra at three x) for fig_greens_compare.py
        cols = [np.argmin(np.abs(xs - xv)) for xv in (7.4, 12.3, 30.0)]
        np.savez(DATA / "greens_run.npz", x=xs, t=tn, E=E, om=om,
                 Et_cols=Et[:, cols], Et_x=xs[cols], E_td=ref)

        # secondary: against the tanh-profile production run (ell = 0.2);
        # BUILD section 6 predicts ~5e-3 differences, not a gate
        d2 = np.load(DATA / "run_main.npz")
        m2 = (d2["x"] >= -3.0) & (d2["x"] <= 55.0)
        x2 = d2["x"][m2][::2]                 # 0.08 spacing on the 0.04 grid
        if len(x2) == len(xs) and np.allclose(x2, xs):
            dtanh = np.abs(E - d2["E"][:, m2][:, ::2]).max()/scale
            print(f"  vs tanh-profile run_main.npz: {dtanh:.2e} "
                  f"(expected ~5e-3 from ell=0.2)")

        # per-slab peaks against BUILD_greens.md section 6
        print("  per-slab peaks, |E|/gB0a0 (Greens vs TD-step vs BUILD table):")
        table = {1: 2.40, 2: 3.71, 3: 6.80, 4: 8.80, 5: 6.85}
        edges = [(-1e9, L), (L, 2*L), (2*L, 3*L), (3*L, 4*L), (4*L, 1e9)]
        for j, (a, b) in enumerate(edges, 1):
            mm = (xs >= a) & (xs < b)
            pk_g = np.abs(E[:, mm]).max()/p5.gB0a0
            pk_t = np.abs(ref[:, mm]).max()/p5.gB0a0
            print(f"    slab {j}: {pk_g:7.3f}  {pk_t:7.3f}  {table[j]:5.2f}")

print("\n================ summary ================")
npass = sum(1 for r in RESULTS if r[3])
for name, v, tgt, ok in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"{npass}/{len(RESULTS)} gates passed")
sys.exit(0 if npass == len(RESULTS) else 1)
