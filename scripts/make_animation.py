"""Animate data/run_main.npz into figures/axion_wavepacket.mp4 (needs ffmpeg
installed): density profile on top, E_y(x) with the packet envelope in the
middle, |E_y| and its running maximum below."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent    # the project directory
sys.path.insert(0, str(ROOT / "code"))           # so the solver imports work
DATA, FIGS = ROOT / "data", ROOT / "figures"

from axion_solver import Params

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

p = Params()

# The .npz is a cache. If it is missing (a fresh clone, or it was cleared),
# regenerate it here from code/axion_solver.py -- this IS the production
# configuration, so it is the file's provenance as well as its fallback.
# dx=0.01 is 7e-5 relative; ~60 s.
if not (DATA / "run_main.npz").exists():
    from axion_solver import solve
    print("run_main.npz missing; running the production solve (~60 s)...",
          flush=True)
    r = solve(p, x_lo=-90.0, x_hi=140.0, dx=0.01, t_final=260.0, n_snap=601,
              sponge_width=35.0, sigma_max=2.0, x_sub=4)
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "run_main.npz", x=r["x"], t=r["t"], E=r["E"], wp=r["wp"])

d = np.load(DATA / "run_main.npz")
x, t, E = d["x"], d["t"], d["E"] / p.gB0a0

XMIN, XMAX = -3.0, 55.0
m = (x >= XMIN) & (x <= XMAX)
xp, Ep = x[m], E[:, m]
Lr, Lb = p.L_region, p.L_box
edges = [i * Lr for i in range(6)]
wlab = [1.2, 1.1, 1.0, 0.9, 0.8]
wp_prof = p.omega_p(xp) / p.m_a

ymax = 1.08 * np.abs(Ep).max()

fig, (axd, ax, axt) = plt.subplots(
    3, 1, figsize=(10.5, 6.4), sharex=True,
    gridspec_kw=dict(height_ratios=[0.9, 3.0, 1.5], hspace=0.12))

# --- density profile -------------------------------------------------
axd.plot(xp, wp_prof, color="0.25", lw=1.4)
axd.axhline(1.0, color="#c0392b", lw=1.0, ls="--")
axd.text(XMAX - 0.5, 1.02, r"$\omega_p=m_a$  (resonance)", color="#c0392b",
         fontsize=8, ha="right", va="bottom")
axd.set_ylabel(r"$\omega_p/m_a$", fontsize=9)
axd.set_ylim(0.72, 1.28)
for e in edges:
    axd.axvline(e, color="0.8", lw=0.7, ls="--")
axd.axvspan(Lb, XMAX, color="0.93", lw=0)
axd.text((Lb + XMAX) / 2, 1.18, "extension (absorbing)", fontsize=8,
         ha="center", color="0.45")

# --- main field panel ------------------------------------------------
for e in edges:
    ax.axvline(e, color="0.8", lw=0.7, ls="--")
ax.axvspan(edges[2], edges[3], color="#c0392b", alpha=0.06, lw=0)
ax.axvspan(Lb, XMAX, color="0.93", lw=0)
ax.axhline(0, color="0.85", lw=0.7)
for i, w in enumerate(wlab):
    ax.text((i + 0.5) * Lr, ymax * 0.90, rf"${w}$", ha="center", fontsize=8, color="0.4")
ax.text(XMIN + 0.3, ymax * 0.90, r"$\omega_p/m_a$:", fontsize=8, color="0.4")

(line,) = ax.plot([], [], color="#1f4e79", lw=1.0)
env_fill = ax.fill_between(xp, -ymax, ymax, color="orange", alpha=0.0, lw=0)
(pkline,) = ax.plot([], [], color="orange", lw=1.2, alpha=0.9)
ttxt = ax.text(0.985, 0.94, "", transform=ax.transAxes, ha="right", va="top",
               fontsize=11, family="monospace")
ax.set_ylim(-ymax, ymax)
ax.set_xlim(XMIN, XMAX)
ax.set_ylabel(r"$E_y\,/\,(g_{a\gamma\gamma}B_0 a_0)$")

# --- running peak-per-region tracker ---------------------------------
cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
axt.set_xlim(XMIN, XMAX)
axt.set_ylim(0, 1.05 * np.abs(Ep).max())
axt.set_xlabel(r"$x$")
axt.set_ylabel(r"$|E_y|$ env.", fontsize=9)
(envline,) = axt.plot([], [], color="0.3", lw=1.0)
(runmax,) = axt.plot([], [], color="#c0392b", lw=1.2, ls="--")
for e in edges:
    axt.axvline(e, color="0.8", lw=0.7, ls="--")
axt.axvspan(Lb, XMAX, color="0.93", lw=0)
axt.text(0.985, 0.88, "solid: $|E_y|$ now    dashed: running max over $t$",
         transform=axt.transAxes, ha="right", fontsize=8, color="0.35")

running = np.zeros_like(xp)


def init():
    line.set_data([], [])
    return line,


def update(j):
    global running, env_fill
    line.set_data(xp, Ep[j])
    xc = p.x0 + p.v_a * t[j]
    env = np.exp(-0.5 * ((xp - xc) / p.sigma_x) ** 2)
    env_fill.remove()
    env_fill = ax.fill_between(xp, -ymax * env, ymax * env, color="orange",
                               alpha=0.16, lw=0, zorder=0)
    pkline.set_data([xc, xc], [-ymax, ymax])
    running = np.maximum(running, np.abs(Ep[j]))
    envline.set_data(xp, np.abs(Ep[j]))
    runmax.set_data(xp, running)
    ttxt.set_text(f"t = {t[j]:6.1f}")
    return line, pkline, envline, runmax, ttxt


ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False)
w = FFMpegWriter(fps=25, bitrate=3000, metadata=dict(title="axion wavepacket"))
ani.save(str(FIGS / "axion_wavepacket.mp4"), writer=w, dpi=130,
         savefig_kwargs=dict(facecolor="w"))
print("frames:", len(t))
