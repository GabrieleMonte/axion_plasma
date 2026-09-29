"""Animate the two solutions on top of each other: the time-domain solver
(solid) and the frequency-domain Green's function (dashed), both on the
piecewise-constant profile, plus their difference.

Reads data/greens_run.npz, rebuilding it from the solvers if it is absent
(see below), and writes figures/greens_vs_solver.mp4. Needs ffmpeg.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
DATA, FIGS = ROOT / "data", ROOT / "figures"

from axion_solver import Params                    # noqa: E402
from greens_solver import StepParams               # noqa: E402

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm",
                     "font.family": "serif", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})

C_TD, C_GR, C_DIF = "#1f4e79", "#c0392b", "#6d6a63"
p = Params()
sp = StepParams()

# The .npz is a cache. If it is missing, regenerate both fields here -- this
# IS how the committed file was produced, so it is the provenance as well as
# the fallback. The time-domain run uses the step profile at dx=0.005 (~8 min
# for the pair; see CLAUDE.md section 7 for why dx matters at the interfaces).
if not (DATA / "greens_run.npz").exists():
    from axion_solver import solve
    from greens_solver import E_y, omega_grid
    print("greens_run.npz missing; regenerating both solutions (~8 min)...",
          flush=True)
    r = solve(sp, x_lo=-90.0, x_hi=140.0, dx=0.005, t_final=260.0, n_snap=601,
              sponge_width=35.0, sigma_max=2.0, x_sub=4)
    keep = (r["x"] >= -3.0) & (r["x"] <= 55.0)
    xs = r["x"][keep][::4]
    E_td_ = r["E"][:, keep][:, ::4]
    print("  time domain done; building E~ ...", flush=True)
    om_ = omega_grid(p)
    Eg_, _, Et_ = E_y(xs, r["t"], p, om=om_)
    cols = [int(np.argmin(np.abs(xs - xv))) for xv in (7.4, 12.3, 30.0)]
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "greens_run.npz", x=xs, t=r["t"], E=Eg_, E_td=E_td_,
             om=om_, Et_cols=Et_[:, cols], Et_x=xs[cols])

d = np.load(DATA / "greens_run.npz")
x, t = d["x"], d["t"]
Eg = d["E"] / p.gB0a0            # Green's function
Et = d["E_td"] / p.gB0a0         # time-domain solver
dif = Eg - Et

L, Lb = p.L_region, p.L_box
edges = [i * L for i in range(6)]
wlab = [1.2, 1.1, 1.0, 0.9, 0.8]
XMIN, XMAX = x[0], x[-1]
ymax = 1.08 * max(np.abs(Eg).max(), np.abs(Et).max())
dmax = 1.15 * np.abs(dif).max()

fig, (axd, ax, axr) = plt.subplots(
    3, 1, figsize=(10.5, 6.8), sharex=True,
    gridspec_kw=dict(height_ratios=[0.75, 3.0, 1.25], hspace=0.14))

# ---- density profile (piecewise constant) ----------------------------
xf = np.linspace(XMIN, XMAX, 4000)
axd.plot(xf, sp.omega_p(xf) / p.m_a, color="0.25", lw=1.4)
axd.axhline(1.0, color=C_GR, lw=1.0, ls="--")
axd.text(XMAX - 0.5, 1.02, r"$\omega_p=m_a$ (resonance)", color=C_GR,
         fontsize=8, ha="right", va="bottom")
axd.set_ylabel(r"$\omega_p/m_a$", fontsize=9)
axd.set_ylim(0.70, 1.42)
axd.axvspan(Lb, XMAX, color="0.93", lw=0)
for i, w in enumerate(wlab):                      # label the steps here, not
    axd.text((i + 0.5) * L, 1.31, rf"${w}$",      # in the field panel, where
             ha="center", fontsize=8, color="0.35")   # the legend lives
axd.text((Lb + XMAX) / 2, 1.31, "extension, 0.8", ha="center",
         fontsize=8, color="0.45")

# ---- main panel ------------------------------------------------------
for a in (axd, ax, axr):
    for e in edges:
        a.axvline(e, color="0.8", lw=0.7, ls="--")
    a.axvspan(Lb, XMAX, color="0.93", lw=0)
ax.axvspan(edges[2], edges[3], color=C_GR, alpha=0.06, lw=0)
ax.axhline(0, color="0.85", lw=0.7)

(l_td,) = ax.plot([], [], color=C_TD, lw=1.6, label="time-domain solver")
(l_gr,) = ax.plot([], [], color=C_GR, lw=1.3, ls=(0, (5, 3)),
                  label="Green's function")
env_fill = ax.fill_between(x, -ymax, ymax, color="#D98A1F", alpha=0.0, lw=0)
(pk,) = ax.plot([], [], color="#D98A1F", lw=1.0, ls=":", alpha=0.9)
ttxt = ax.text(0.985, 0.94, "", transform=ax.transAxes, ha="right", va="top",
               fontsize=11, family="monospace")
ax.set_ylim(-ymax, ymax)
ax.set_xlim(XMIN, XMAX)
ax.set_ylabel(r"$E_y\,/\,(g_{a\gamma\gamma}B_0 a_0)$")
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

# ---- residual --------------------------------------------------------
(l_d,) = axr.plot([], [], color=C_DIF, lw=1.0)
axr.axhline(0, color="0.85", lw=0.7)
axr.set_ylim(-dmax, dmax)
axr.set_xlim(XMIN, XMAX)
axr.set_xlabel(r"$x$")
axr.set_ylabel("difference", fontsize=9)
rtxt = axr.text(0.985, 0.88, "", transform=axr.transAxes, ha="right",
                fontsize=8.5, color="0.35", family="monospace")
axr.text(0.015, 0.88, "Green's $-$ solver, same units as above",
         transform=axr.transAxes, fontsize=8, color="0.45")

fig.suptitle("Axion wavepacket in five plasma slabs: two independent solutions",
             fontsize=11, y=0.985)


def update(j):
    global env_fill
    l_td.set_data(x, Et[j])
    l_gr.set_data(x, Eg[j])
    l_d.set_data(x, dif[j])
    xc = p.x0 + p.v_a * t[j]
    env = np.exp(-0.5 * ((x - xc) / p.sigma_x) ** 2)
    env_fill.remove()
    env_fill = ax.fill_between(x, -ymax * env, ymax * env, color="#D98A1F",
                               alpha=0.16, lw=0, zorder=0)
    pk.set_data([xc, xc], [-ymax, ymax])
    ttxt.set_text(f"t = {t[j]:6.1f}")
    rtxt.set_text(f"max |diff| = {np.abs(dif[j]).max():.1e}")
    return l_td, l_gr, l_d, pk, ttxt, rtxt


ani = FuncAnimation(fig, update, frames=len(t), blit=False)
w = FFMpegWriter(fps=25, bitrate=3600,
                 metadata=dict(title="Green's function vs time-domain solver"))
out = FIGS / "greens_vs_solver.mp4"
ani.save(str(out), writer=w, dpi=130, savefig_kwargs=dict(facecolor="w"))
print(f"wrote {out}  ({len(t)} frames)")
print(f"max |Greens - solver| over all (x,t) = {np.abs(dif).max():.3e}"
      f"   = {np.abs(dif).max()/np.abs(Et).max():.2e} of peak")
