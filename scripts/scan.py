"""Resonance scan (CLAUDE.md section 5): one slab of adjustable density in a
uniform wp = 0.8 background. For each slab density, run the solver and record
the energy the generated wave carries past x = 45 (time-integrated E_y*B_z),
for two packet widths sigma_x.

Writes data/scan.json, plotted by fig_scan.py. 16 solver runs, a few seconds
each.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent    # the project directory
sys.path.insert(0, str(ROOT / "code"))           # so the solver imports work
DATA = ROOT / "data"

from axion_solver import Params, solve


class OneSlab(Params):
    """wp/m_a = 0.8 everywhere except a single slab at w_slab on
    [2*L_region, 3*L_region], with the same tanh edges as the main profile."""

    def __init__(self, w_slab, **kw):
        super().__init__(**kw)
        self.w_slab = w_slab

    def omega_p(self, x):
        x = np.asarray(x, float)
        a, b = 2*self.L_region, 3*self.L_region      # slab edges
        # 0 outside the slab, 1 inside, smooth tanh walls
        inside = 0.25*(1 + np.tanh((x - a)/self.ell))*(1 + np.tanh((b - x)/self.ell))
        return self.m_a*(0.8 + (self.w_slab - 0.8)*inside)


# run the scan only when this file is executed directly ("python scan.py"),
# not when OneSlab is imported from elsewhere
if __name__ == "__main__":
    w_scan = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    results = {}
    for sigma in [1.5, 4.0]:
        results[sigma] = []
        for w in w_scan:
            p = OneSlab(w, sigma_x=sigma)
            # left pad grows with sigma so the packet tail never touches the edge
            r = solve(p, x_lo=-40 - 8*sigma, x_hi=95, dx=0.04, t_final=300.0,
                      n_snap=3, sponge_width=25.0, sigma_max=2.0, x_sub=4,
                      x_monitor=45.0)
            results[sigma].append(r["flux"])
            print(sigma, w, r["flux"], flush=True)

    json.dump({str(k): v for k, v in results.items()}, open(DATA / "scan.json", "w"))
