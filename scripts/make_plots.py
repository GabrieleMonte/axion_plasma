"""Main figures from data/run_main.npz: fig_main.png (the E_y(x,t) map and
per-slab peak history) and fig_snapshots.png, plus the printed summary table
of peak field per slab (CLAUDE.md section 5)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent    # the project directory
sys.path.insert(0, str(ROOT / "code"))           # so the solver imports work
DATA, FIGS = ROOT / "data", ROOT / "figures"

from axion_solver import Params

plt.rcParams.update({
    "font.size": 10, "axes.linewidth": 0.8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "cm", "font.family": "serif",
})

p = Params()
d = np.load(DATA / "run_main.npz")
x, t, E, wp = d["x"], d["t"], d["E"] / p.gB0a0, d["wp"]
Lr, Lb = p.L_region, p.L_box
edges = [i * Lr for i in range(6)]
wlab = [1.2, 1.1, 1.0, 0.9, 0.8]

XMAX = 55.0
mx = (x >= -3) & (x <= XMAX)
xp, Ep = x[mx], E[:, mx]

# =====================================================================
fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                      hspace=0.30, wspace=0.24)

# ---- (a) raw E_y -----------------------------------------------------
ax = fig.add_subplot(gs[0, :])
v = np.abs(Ep).max()
im = ax.pcolormesh(xp, t, Ep, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v),
                   shading="auto", rasterized=True)
plt.colorbar(im, ax=ax, pad=0.01, label=r"$E_y\,/\,(g_{a\gamma\gamma}B_0a_0)$")
for e in edges:
    ax.axvline(e, color="0.25", lw=0.7, ls="--")
ax.axvline(Lb, color="k", lw=1.2)
ax.plot(p.x0 + p.v_a * t, t, color="k", lw=1.2, ls=":")
for i, w in enumerate(wlab):
    ax.text((i + 0.5) * Lr, 264, rf"${w}$", ha="center", fontsize=8)
ax.text(Lb + (XMAX - Lb) / 2, 264, r"extension, $\omega_p/m_a=0.8$", ha="center", fontsize=8)
ax.text(-2.5, 264, r"$\omega_p/m_a$:", ha="left", fontsize=8)
ax.set_xlim(-3, XMAX); ax.set_ylim(0, 260)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
ax.set_title(r"(a)  $E_y(x,t)$;  dotted line = packet centre $x_0+v_at$", loc="left", fontsize=10, pad=18)

# ---- (b) per-slice normalised ---------------------------------------
ax = fig.add_subplot(gs[1, 0])
norm = Ep / (np.abs(Ep).max(axis=1, keepdims=True) + 1e-300)
im = ax.pcolormesh(xp, t, np.abs(norm), cmap="magma", vmin=0, vmax=1,
                   shading="auto", rasterized=True)
plt.colorbar(im, ax=ax, pad=0.01, label=r"$|E_y|/\max_x|E_y|$")
for e in edges:
    ax.axvline(e, color="w", lw=0.7, ls="--", alpha=0.6)
ax.plot(p.x0 + p.v_a * t, t, color="c", lw=1.0, ls=":")
ax.set_xlim(-3, XMAX); ax.set_ylim(0, 260)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
ax.set_title(r"(b)  normalised per time slice", loc="left", fontsize=10)

# ---- (c) peak per region vs t ---------------------------------------
ax = fig.add_subplot(gs[1, 1])
cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
for i, w in enumerate(wlab):
    m = (x >= edges[i]) & (x < edges[i + 1])
    ax.plot(t, np.abs(E[:, m]).max(axis=1), color=cols[i], lw=1.4,
            label=rf"$\omega_p/m_a={w}$")
tc = [(edges[i] - p.x0) / p.v_a for i in range(1, 5)]
for i, tt in enumerate(tc):
    ax.axvline(tt, color="0.7", lw=0.7, ls=":")
ax.axvspan(tc[1], tc[2], color="gold", alpha=0.13, lw=0)
ax.text((tc[1] + tc[2]) / 2, 9.4, "packet in\nresonant region", ha="center", fontsize=7.5)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\max_x |E_y| / (g_{a\gamma\gamma}B_0a_0)$")
ax.set_xlim(0, 260); ax.set_ylim(0, 10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title(r"(c)  peak field in each region", loc="left", fontsize=10)

fig.savefig(FIGS / "fig_main.png", bbox_inches="tight", facecolor="w")
plt.close(fig)

# =====================================================================
fig, axs = plt.subplots(3, 2, figsize=(11.5, 8.0), sharex=True)
axs = axs.ravel()
snaps = [40, 80, 100, 120, 160, 250]
for ax, ts in zip(axs, snaps):
    j = np.argmin(np.abs(t - ts))
    ax.plot(xp, Ep[j], color="#1f4e79", lw=0.9)
    ax.axhline(0, color="0.8", lw=0.6)
    for e in edges:
        ax.axvline(e, color="0.75", lw=0.6, ls="--")
    xc = p.x0 + p.v_a * t[j]
    ax.axvspan(xc - 2 * p.sigma_x, xc + 2 * p.sigma_x, color="orange", alpha=0.18, lw=0)
    ax.set_xlim(-3, XMAX)
    ax.set_title(rf"$t={t[j]:.0f}$", loc="left", fontsize=9)
    ax.set_ylabel(r"$E_y/(g B_0 a_0)$", fontsize=8)
for ax in axs[4:]:
    ax.set_xlabel(r"$x$")
fig.suptitle("Snapshots of $E_y(x)$   (shaded: $\\pm2\\sigma_x$ of the axion packet)",
             fontsize=11, y=0.995)
fig.tight_layout()
fig.savefig(FIGS / "fig_snapshots.png", bbox_inches="tight", facecolor="w")
plt.close(fig)

# =====================================================================
# quantitative summary
print("region   wp/m_a   k_gamma        v_g      peak|E|/(gB0a0)   t_peak")
for i, w in enumerate(wlab):
    kk = p.omega_a ** 2 - (w * p.m_a) ** 2
    ks = f"{np.sqrt(kk):.4f}" if kk > 0 else f"i{np.sqrt(-kk):.4f}"
    vg = f"{np.sqrt(kk)/p.omega_a:.4f}" if kk > 0 else "  --  "
    m = (x >= edges[i]) & (x < edges[i + 1])
    pk = np.abs(E[:, m]).max(axis=1)
    print(f"  {i+1}      {w:.1f}    {ks:>9s}   {vg:>7s}      {pk.max():8.3f}      {t[pk.argmax()]:6.1f}")
m = x > Lb
pk = np.abs(E[:, m]).max(axis=1)
print(f"  ext    0.8      0.6084    0.6053      {pk.max():8.3f}      {t[pk.argmax()]:6.1f}")
print()
print("steady-state local estimate  E_y = gB0 wa^2 a /(wp^2+k_a^2-wa^2), in units of gB0a0:")
for w in wlab:
    den = (w * p.m_a) ** 2 + p.k_a ** 2 - p.omega_a ** 2   # = wp^2 - m_a^2 exactly
    est = "     inf  (resonant)" if abs(den) < 1e-12 else f"{p.omega_a**2/den:+8.2f}"
    print(f"   wp={w}:  {est}")
