"""A gallery of single-frequency modes: what the stack does to one omega at a
time. Writes two animations from the same five modes.

  figures/modes_vs_phase.mp4   x-axis of the clock is the mode's own phase,
                               so all five rows stay locked together and you
                               compare shapes
  figures/modes_vs_time.mp4    real time, so each row runs at its own omega:
                               they start aligned at t = 0 and drift apart,
                               which is the 35% spread in omega made visible

A single mode is Re[E~(x,omega) e^{-i phi}] with phi = omega t -- a fixed
envelope with the phase running through it. Where the crests march the mode
radiates; where they beat in place it does not.

Both animations use ONE shared absolute scale across all five panels, so the
panel heights answer "which frequency does the plasma actually respond to":
omega = m_a wins by a factor 14 over omega = 0.85.

Computes everything itself in a few seconds -- no data file. Needs ffmpeg.
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
FIGS = ROOT / "figures"

from axion_solver import Params                      # noqa: E402
from greens_solver import StepParams, E_tilde        # noqa: E402

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm",
                     "font.family": "serif", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})

p, sp = Params(), StepParams()
L, Lb, ma = p.L_region, p.L_box, p.m_a
EPS = 0.015

# diverging about the resonance: blue below m_a, dark at m_a, red above.
# one omega per propagation regime -- each step up opens one more slab.
MODES = [(0.85, "#7fb0d4"), (0.95, "#2e6da4"), (1.00, "#1a1a1a"),
         (1.05, "#c0392b"), (1.15, "#e0a090")]
oms = np.array([w for w, _ in MODES])

x = np.linspace(-3.0, 55.0, 2000)
Et = E_tilde(x, oms + 1j*EPS, p)/p.gB0a0
amp = np.abs(Et).max(axis=1)
YLIM = 1.06*np.abs(Et).max()                 # one scale for every row
edges = [i*L for i in range(6)]


def open_slabs(w):
    """Slabs the mode can propagate in: omega above the local cutoff."""
    return [j + 1 for j, wj in enumerate(p.w) if w > wj*ma]


def build(subtitle):
    """The static figure; returns it plus the five lines and the readout."""
    fig, axs = plt.subplots(
        len(MODES) + 1, 1, figsize=(10.5, 10.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.5] + [1.0]*len(MODES), hspace=0.40))
    axd, rows = axs[0], axs[1:]

    xf = np.linspace(x[0], x[-1], 4000)
    axd.plot(xf, sp.omega_p(xf)/ma, color="0.3", lw=1.6, zorder=4)
    # labels staggered in x so five closely spaced lines stay readable
    for (w, c), xlab in zip(MODES, (52.0, 45.5, 39.0, 32.5, 26.0)):
        axd.axhline(w, color=c, lw=1.1, zorder=2)
        axd.text(xlab, w, rf"$\omega={w:.2f}$", color=c, fontsize=8.2,
                 ha="center", va="center", zorder=5,
                 bbox=dict(fc="w", ec="none", pad=1.2))
    axd.axvspan(Lb, x[-1], color="0.94", lw=0)
    axd.axvspan(edges[2], edges[3], color="0.82", alpha=0.6, lw=0, zorder=0)
    axd.set_ylim(0.74, 1.30)
    axd.set_ylabel(r"$\omega_p/m_a$", fontsize=9)
    axd.set_title("A mode propagates only where its line sits above the "
                  "staircase; the resonant slab is shaded", loc="left",
                  fontsize=9.5)

    lines = []
    for i, ((w, c), ax) in enumerate(zip(MODES, rows)):
        ax.fill_between(x, -np.abs(Et[i]), np.abs(Et[i]), color=c, alpha=0.16,
                        lw=0, zorder=0)
        (ln,) = ax.plot([], [], color=c, lw=1.5)
        lines.append(ln)
        ax.axhline(0, color="0.9", lw=0.6)
        for e in edges:
            ax.axvline(e, color="0.87", lw=0.7, ls="--")
        ax.axvspan(edges[2], edges[3], color="0.82", alpha=0.35, lw=0, zorder=0)
        ax.axvspan(Lb, x[-1], color="0.94", lw=0, zorder=0)
        ax.set_ylim(-YLIM, YLIM)
        ax.set_yticks([-40, 0, 40])
        ax.tick_params(labelsize=7.5)
        tag = r"the resonance, $\kappa_3=0$ in the shaded slab" \
            if abs(w - ma) < 1e-9 else \
            ("below the resonance" if w < ma else "above the resonance")
        ax.set_title(rf"$\omega={w:.2f}$ — {tag}", loc="left", fontsize=9,
                     color=c, pad=3)
        marg = [j + 1 for j, wj in enumerate(p.w) if abs(w - wj*ma) < 1e-9]
        extra = rf", marginal in slab {marg[0]}" if marg else ""
        ax.set_title(rf"max$|\tilde{{E}}|={amp[i]:.1f}$   ·   propagates in "
                     rf"slabs {','.join(map(str, open_slabs(w)))}{extra}",
                     loc="right", fontsize=7.8, color="0.45", pad=3)
    rows[-1].set_xlabel(r"$x$")
    rows[0].set_xlim(x[0], x[-1])
    fig.text(0.052, 0.40, r"$\mathrm{Re}\,[\,\tilde{E}_y(x,\omega)\,"
             r"e^{-i\phi}\,]\,/\,(g_{a\gamma\gamma}B_0a_0)$"
             "   —   every panel on the same scale",
             rotation=90, va="center", ha="center", fontsize=9.5)
    readout = axd.text(0.986, 0.90, "", transform=axd.transAxes, ha="right",
                       va="top", fontsize=10, family="monospace")
    fig.suptitle("One frequency at a time, all panels on one scale: "
                 rf"{subtitle}", fontsize=11.5, y=0.985)
    fig.subplots_adjust(top=0.935, left=0.105, right=0.985, bottom=0.052)
    return fig, lines, readout


def render(kind):
    """kind='phase': one clock for all rows. kind='time': each row at its own
    omega, so they drift apart."""
    if kind == "phase":
        clock = np.linspace(0.0, 4*np.pi, 240, endpoint=False)   # two cycles
        sub = r"the rows share a clock, so shapes line up"
        lab = lambda v: f"phase = {v % (2*np.pi):4.2f}"
        ph_of = lambda v: np.full(len(MODES), v)
    else:
        clock = np.linspace(0.0, 30.0, 300, endpoint=False)
        sub = r"real time, so each row runs at its own $\omega$"
        lab = lambda v: f"t = {v:5.2f}"
        ph_of = lambda v: oms*v

    fig, lines, readout = build(sub)

    def update(k):
        ph = ph_of(clock[k])
        for i, ln in enumerate(lines):
            ln.set_data(x, (Et[i]*np.exp(-1j*ph[i])).real)
        readout.set_text(lab(clock[k]))
        return (*lines, readout)

    ani = FuncAnimation(fig, update, frames=len(clock), blit=False)
    out = FIGS / f"modes_vs_{kind}.mp4"
    ani.save(str(out), writer=FFMpegWriter(
        fps=24, bitrate=3600,
        metadata=dict(title=f"single-frequency modes vs {kind}")),
        dpi=130, savefig_kwargs=dict(facecolor="w"))
    plt.close(fig)
    print(f"wrote {out}  ({len(clock)} frames)")


for k in ("phase", "time"):
    render(k)
print(f"shared scale: +/- {YLIM:.1f}")
for (w, _), a in zip(MODES, amp):
    print(f"  omega={w:.2f}  max|E~|={a:6.2f}  open in slabs {open_slabs(w)}")
