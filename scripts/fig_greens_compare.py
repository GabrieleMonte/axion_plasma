"""Overlay of the frequency-domain Green's-function solution on the
time-domain reference (step profile): snapshots with residuals, per-slab
peak history, the omega-spectrum |E~|, and the Wronskian scan.

Needs data/greens_run.npz (written by validate_greens.py, full run) and
data/run_step_dx0.005.npz. Writes figures/fig_greens_compare.png."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
DATA, FIGS = ROOT / "data", ROOT / "figures"

from axion_solver import Params                  # noqa: E402
from tmatrix import psi_states, kap              # noqa: E402

plt.rcParams.update({
    "font.size": 10, "axes.linewidth": 0.8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "cm", "font.family": "serif",
})
C_TD, C_GR = "#1f4e79", "#c0392b"        # time-domain solid, Green's dashed

p = Params()
L, Lb = p.L_region, p.L_box
edges = [i*L for i in range(6)]
g = np.load(DATA / "greens_run.npz")
x, t, Eg, E_td = g["x"], g["t"], g["E"]/p.gB0a0, g["E_td"]/p.gB0a0
om, Et_cols, Et_x = g["om"], g["Et_cols"], g["Et_x"]

fig = plt.figure(figsize=(11.5, 9.2))
gs = fig.add_gridspec(3, 4, height_ratios=[2.1, 0.7, 1.6], hspace=0.35, wspace=0.3)

# ---- row 1+2: snapshots with residual strips ---------------------------
snaps = [40.0, 100.0, 147.0, 250.0]
for i, ts in enumerate(snaps):
    j = np.argmin(np.abs(t - ts))
    ax = fig.add_subplot(gs[0, i])
    axr = fig.add_subplot(gs[1, i], sharex=ax)
    for e in edges:
        ax.axvline(e, color="0.85", lw=0.6, ls="--")
    ax.plot(x, E_td[j], color=C_TD, lw=1.1, label="time domain")
    ax.plot(x, Eg[j], color=C_GR, lw=1.0, ls="--", label="Green's fn")
    ax.set_xlim(-3, 55)
    ax.set_title(rf"$t={t[j]:.0f}$", loc="left", fontsize=9)
    if i == 0:
        ax.set_ylabel(r"$E_y/(g B_0 a_0)$")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axr.plot(x, (Eg[j] - E_td[j])*1e3, color="0.35", lw=0.7)
    axr.axhline(0, color="0.85", lw=0.5)
    axr.set_xlabel(r"$x$")
    if i == 0:
        axr.set_ylabel(r"$\Delta\times10^{3}$", fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)

# ---- row 3 left: per-slab peak history ---------------------------------
ax = fig.add_subplot(gs[2, 0:2])
cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
for i in range(5):
    m = (x >= edges[i]) & (x < edges[i + 1])
    ax.plot(t, np.abs(E_td[:, m]).max(axis=1), color=cols[i], lw=1.3,
            label=rf"$\omega_p/m_a={p.w[i]}$")
    ax.plot(t, np.abs(Eg[:, m]).max(axis=1), color=cols[i], lw=1.0, ls="--")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\max_x|E_y|/(gB_0a_0)$")
ax.set_xlim(0, 260)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(b)  peak per slab: solid = time domain, dashed = Green's",
             loc="left", fontsize=9.5)

# ---- row 3 middle: |E~(x*, omega)| -------------------------------------
ax = fig.add_subplot(gs[2, 2])
for i, (xv, c) in enumerate(zip(Et_x, ["#1f4e79", "#c0392b", "#6d8f3c"])):
    ax.semilogy(om, np.abs(Et_cols[:, i])/p.gB0a0, color=c, lw=0.9,
                label=rf"$x={xv:.1f}$")
for wc in np.unique(np.asarray(p.w))*p.m_a:
    ax.axvline(wc, color="0.8", lw=0.6, ls=":")
ax.axvline(p.omega_a, color="0.4", lw=0.7, ls="--")
ax.text(p.omega_a + 0.04, 3e-6, r"$\omega_a$", fontsize=8, color="0.3")
ax.set_xlim(0, 2.5)
ax.set_ylim(1e-7, 3e2)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$|\tilde{E}_y|/(gB_0a_0)$")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("(c)  spectrum on the contour", loc="left", fontsize=9.5)

# ---- row 3 right: Wronskian scan ---------------------------------------
ax = fig.add_subplot(gs[2, 3])
oms = np.arange(0.02, 5.0, 2e-3)
r = np.array([abs(psi_states(p, w)[2])
              / max(abs(kap(w, 1.44)) + abs(kap(w, 0.64)), 0.3) for w in oms])
ax.semilogy(oms, r, color="0.25", lw=0.9)
for wc in np.unique(np.asarray(p.w))*p.m_a:
    ax.axvline(wc, color="0.8", lw=0.6, ls=":")
ax.axhline(r.min(), color="#c0392b", lw=0.8, ls="--")
ax.text(2.6, r.min()*1.25, f"min = {r.min():.2f}", fontsize=7.5, color="#c0392b")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$|W|/(\kappa_1+\kappa_5)$")
ax.set_title("(d)  no real-axis zeros of $W$", loc="left", fontsize=9.5)

fig.suptitle("Frequency-domain Green's function vs time-domain solver "
             "(piecewise-constant profile)", fontsize=11, y=0.995)
fig.savefig(FIGS / "fig_greens_compare.png", bbox_inches="tight", facecolor="w")
print("wrote", FIGS / "fig_greens_compare.png")
print(f"max |Greens - TD| / max|TD| = "
      f"{np.abs(Eg - E_td).max()/np.abs(E_td).max():.2e}")
