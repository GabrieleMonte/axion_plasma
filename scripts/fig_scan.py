"""Figure for the single-slab resonance scan: reads data/scan.json (made by
scan.py) and writes figures/fig_resonance_scan.png. Panel (a): radiated energy
vs the scanned slab density. Panel (b): the same divided by the no-slab
(wp = 0.8) value, i.e. how much the slab stands out above the background."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent    # the project directory
DATA, FIGS = ROOT / "data", ROOT / "figures"

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "cm",
                     "font.family": "serif"})

d = json.load(open(DATA / "scan.json"))
ws = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
styles = [("1.5", "#c0392b", "o"), ("4.0", "#1f4e79", "s")]   # sigma_x, colour, marker

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))

ax = axs[0]
for s, c, mk in styles:
    f = np.abs(np.array(d[s]))
    ax.plot(ws, f / f.max(), mk + "-", color=c, ms=4, lw=1.4, label=rf"$\sigma_x={s}$")
ax.axvline(1.0, color="0.6", ls="--", lw=0.9)
ax.text(1.005, 0.06, r"$\omega_p=m_a$", fontsize=8, color="0.4")
ax.set_xlabel(r"$\omega_p/m_a$ of the scanned slab")
ax.set_ylabel("radiated energy (normalised)")
ax.set_title("(a)  resonance curve, single slab in a $0.8$ background", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 1.08)

ax = axs[1]
for s, c, mk in styles:
    f = np.abs(np.array(d[s]))
    ax.plot(ws, f / f[0], mk + "-", color=c, ms=4, lw=1.4, label=rf"$\sigma_x={s}$")
ax.axvline(1.0, color="0.6", ls="--", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel(r"$\omega_p/m_a$ of the scanned slab")
ax.set_ylabel("contrast vs. no slab")
ax.set_title("(b)  same, relative to the background (log)", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(FIGS / "fig_resonance_scan.png", bbox_inches="tight", facecolor="w")
print("contrast (peak/background):  sigma_x=1.5 -> %.2f   sigma_x=4.0 -> %.2f" % (
    np.abs(d["1.5"]).max() / abs(d["1.5"][0]), np.abs(d["4.0"]).max() / abs(d["4.0"][0])))
