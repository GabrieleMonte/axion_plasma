"""Restart a uniform-slab solution from numerical (E, dE/dt) data.

Kept as the record of a ruled-out idea (CLAUDE.md 6.3): the implementation is
correct (self-consistent to 5e-16 at tau = 0) but treating one slab as
isolated fails once waves have crossed it — useless for these parameters,
where the packet takes ~10 light-crossings per slab.
"""
import numpy as np


def propagate(dump, mu, p, tq):
    """E_y at time tq in a uniform slab of plasma frequency mu, evolved
    exactly (per Fourier mode) from the fields in `dump`: grid x, field E,
    time derivative Edot, all taken at time dump['t']."""
    x, E0, V0, tj = dump['x'], dump['E'], dump['Edot'], dump['t']
    N = len(x)
    dx = x[1] - x[0]
    k = 2*np.pi*np.fft.fftfreq(N, dx)
    Wk = np.sqrt(k**2 + mu**2)                    # photon frequency of mode k
    Eh, Vh = np.fft.fft(E0), np.fft.fft(V0)
    tau = tq - tj

    # free evolution of the initial data (each mode is a plain oscillator)
    out = np.cos(Wk*tau)*Eh + np.sin(Wk*tau)/Wk*Vh

    # plus the axion drive over [tj, tq]; same kernel as analytic.E_slab.
    # ph corrects for the grid starting at x[0] != 0: numpy's fft phases are
    # relative to index 0, the analytic transform is relative to x = 0
    ph = np.exp(-1j*k*(-x[0]))
    for s in (+1, -1):
        d  = k - s*p.k_a
        Om = s*p.omega_a + p.v_a*d
        A  = (p.gB0a0*np.sqrt(2*np.pi)*p.sigma_x/2.0)*np.exp(-0.5*p.sigma_x**2*d**2) \
             * np.exp(-1j*k*p.x0)
        D  = Wk**2 - Om**2
        num = Om**2*(np.exp(-1j*Om*tau) - np.cos(Wk*tau)) + 1j*Om*Wk*np.sin(Wk*tau)
        reg = np.abs(D) > 1e-8
        Kf = np.where(reg, num/np.where(reg, D, 1.0),
                      0.5j*(Om*tau*np.exp(-1j*Om*tau) + np.sin(Om*tau)))
        out += ph * A * np.exp(-1j*Om*tj) * Kf / dx
    return np.fft.ifft(out).real
