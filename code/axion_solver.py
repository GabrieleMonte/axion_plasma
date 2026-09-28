"""
Axion wavepacket crossing a density-graded magnetized plasma, 1D.

Solves the first-order system (code units, c = 1, Heaviside-Lorentz):

    d_t E_y = -d_x B_z - J_y - g B0 d_t a        (Ampere)
    d_t B_z = -d_x E_y                            (Faraday)
    d_t J_y =  wp^2(x) E_y                        (cold linear plasma response)

with  E_y = B_z = J_y = 0  at t = 0, the axion prescribed (no backreaction),
and  B0 y-hat  a fixed external field.

Grid: E_y and J_y live on the points x_i, B_z halfway between them; E is
updated at integer time steps, B and J at half-integer steps ("leapfrog").
Second order in dx and dt, and does not damp waves. This is the same field
scheme a PIC code uses.

Near both ends an artificial conductivity ramps up smoothly and absorbs
outgoing waves before they can reflect back (the "sponge" layers).
"""

import numpy as np


# ----------------------------------------------------------------------------
# physical parameters
# ----------------------------------------------------------------------------

class Params:
    def __init__(self,
                 m_a=1.0, v_a=0.1,
                 sigma_x=1.5, x0=1.5,
                 gB0a0=3.0e-4,
                 L_region=24.576 / 5.0,
                 w=(1.2, 1.1, 1.0, 0.9, 0.8),
                 ell=0.2):
        self.m_a = m_a
        self.v_a = v_a
        self.gamma_a = 1.0 / np.sqrt(1.0 - v_a ** 2)
        self.omega_a = self.gamma_a * m_a
        self.k_a = self.gamma_a * m_a * v_a
        self.sigma_x = sigma_x
        self.x0 = x0
        self.gB0a0 = gB0a0          # only the product g_agg * B0 * a0 matters
        self.L_region = L_region    # width of each of the 5 density regions
        self.w = np.asarray(w)      # wp/m_a in each region, left to right
        self.ell = ell              # tanh transition half-width
        self.L_box = 5 * L_region   # the original 5-region box

    def omega_p(self, x):
        """wp(x). Automatically continues as 1.2 m_a for x<0 and 0.8 m_a for
        x>L_box, which is exactly the padding we want."""
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, self.w[0])
        for i in range(4):
            xi = (i + 1) * self.L_region
            out += 0.5 * (self.w[i + 1] - self.w[i]) * (1.0 + np.tanh((x - xi) / self.ell))
        return out * self.m_a


# ----------------------------------------------------------------------------
# axion source
# ----------------------------------------------------------------------------

def adot_packet(x, t, p):
    """g*B0 * d_t a  for the moving packet
        a(x,t) = a0 exp[-(x-x0-v_a t)^2 / 2 sigma^2] cos[k_a(x-x0) - omega_a t]
    (a0 enters through the single product gB0a0)."""
    u = x - p.x0
    xi = u - p.v_a * t
    env = np.exp(-0.5 * (xi / p.sigma_x) ** 2)
    phase = p.k_a * u - p.omega_a * t
    return p.gB0a0 * env * ((p.v_a * xi / p.sigma_x ** 2) * np.cos(phase)
                            + p.omega_a * np.sin(phase))


def adot_homogeneous(x, t, p):
    """d_t a for the validation case a(t) = a0 sin(w_a t)."""
    return p.gB0a0 * p.omega_a * np.cos(p.omega_a * t) * np.ones_like(x)


# ----------------------------------------------------------------------------
# sponge
# ----------------------------------------------------------------------------

def sponge_profile(x, x_lo, x_hi, width, sigma_max):
    """Cubic ramp-up conductivity in the outer `width` of each end."""
    s = np.zeros_like(x)
    left = (x_lo + width - x) / width
    right = (x - (x_hi - width)) / width
    s += sigma_max * np.clip(left, 0.0, 1.0) ** 3
    s += sigma_max * np.clip(right, 0.0, 1.0) ** 3
    return s


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------

def solve(p, x_lo=-30.0, x_hi=90.0, dx=0.02, cfl=0.5, t_final=260.0,
          sponge_width=15.0, sigma_max=2.0, n_snap=520,
          source=adot_packet, uniform_wp=None, x_sub=4, x_monitor=None):
    """
    Returns dict with x, t, E (shape n_snap x len(x)), and diagnostics.
    `uniform_wp`: if not None, override wp(x) by this constant (validation).
    `x_monitor`: if given, also time-integrate the energy flux E_y*B_z there.
    """
    nx = int(round((x_hi - x_lo) / dx)) + 1
    x = x_lo + dx * np.arange(nx)                 # E_y, J_y nodes
    xh = x[:-1] + 0.5 * dx                        # B_z nodes

    if uniform_wp is None:
        wp2 = p.omega_p(x) ** 2
    else:
        wp2 = np.full(nx, uniform_wp ** 2)

    sig = sponge_profile(x, x_lo, x_hi, sponge_width, sigma_max)
    sigh = sponge_profile(xh, x_lo, x_hi, sponge_width, sigma_max)

    dt = cfl * dx
    nt = int(round(t_final / dt))

    # update coefficients: each step does  y -> c1*y + c2*(the rest of dy/dt),
    # where c1 < 1 only inside the sponge (that inequality is what absorbs).
    # E and J sit on the same points so they share coefficients; B has its own.
    cE1 = (1.0 - 0.5 * sig * dt) / (1.0 + 0.5 * sig * dt)
    cE2 = dt / (1.0 + 0.5 * sig * dt)
    cB1 = (1.0 - 0.5 * sigh * dt) / (1.0 + 0.5 * sigh * dt)
    cB2 = dt / (1.0 + 0.5 * sigh * dt)

    E = np.zeros(nx)
    B = np.zeros(nx - 1)
    J = np.zeros(nx)

    snap_idx = np.unique(np.linspace(0, nt, n_snap).astype(int))
    xs = x[::x_sub]
    E_out = np.zeros((len(snap_idx), len(xs)))
    t_out = np.zeros(len(snap_idx))
    ptr = 0

    rhs = np.empty(nx)      # holds dB/dx + J + source = -d_t E, rebuilt each step

    # flux monitor: energy flux S_x = E_y B_z through x_monitor (Poynting)
    if x_monitor is not None:
        i_mon = int(round((x_monitor - x_lo) / dx))
        flux = 0.0
        E_mon_prev = 0.0
    else:
        i_mon = None
        flux = None

    for n in range(nt + 1):
        t = n * dt

        if ptr < len(snap_idx) and n == snap_idx[ptr]:
            E_out[ptr] = E[::x_sub]
            t_out[ptr] = t
            ptr += 1
        if n == nt:
            break

        # B_z^{n+1/2} from E_y^n                        (Faraday)
        B[:] = cB1 * B - cB2 * ((E[1:] - E[:-1]) / dx)

        # J_y^{n+1/2} from E_y^n                        (cold-plasma current)
        J[:] = cE1 * J + cE2 * (wp2 * E)

        # E_y^{n+1} from B_z, J_y, source at t+dt/2     (Ampere)
        rhs[1:-1] = (B[1:] - B[:-1]) / dx
        rhs[0] = 0.0
        rhs[-1] = 0.0
        rhs += J + source(x, t + 0.5 * dt, p)
        E[:] = cE1 * E - cE2 * rhs
        E[0] = 0.0                # fixed ends, far behind the sponges
        E[-1] = 0.0

        if i_mon is not None:
            # E is now at t=(n+1)dt, B at (n+1/2)dt: average E over the two
            # times and B over the two neighbouring half-points, so both sit
            # at (x_monitor, (n+1/2)dt) before multiplying
            flux += dt * (0.5 * (E[i_mon] + E_mon_prev) * 0.5 * (B[i_mon] + B[i_mon - 1]))
            E_mon_prev = E[i_mon]

    return dict(x=xs, t=t_out, E=E_out, dx=dx, dt=dt, flux=flux,
                wp=(p.omega_p(xs) if uniform_wp is None
                    else np.full_like(xs, uniform_wp)), p=p)
