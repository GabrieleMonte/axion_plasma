# BUILD: frequency-domain Green's function solver

Read `greens_derivation.tex` first. It is the specification. Every equation
number below refers to it.

Goal: compute `E_y(x,t)` for the axion-wavepacket problem by the exact
frequency-domain construction, with **no approximation except the numerical
evaluation of the final ω integral, Eq. (43)**. Then compare against the existing
time-domain solver.

Do not substitute a different method. If something looks intractable, say so and
stop rather than approximating around it.

---

## 1. Physical setup (fixed, do not change)

Units `c = 1`, Heaviside–Lorentz, `m_a = 1`. Fields in units of `g_agg*B0*a0`.

```
v_a      = 0.1
gamma_a  = 1/sqrt(1-v_a^2)  = 1.0050378152592121
omega_a  = gamma_a*m_a      = 1.0050378152592121
k_a      = gamma_a*m_a*v_a  = 0.10050378152592121
sigma_x  = 1.5
x0       = 1.5
g*B0*a0  = 3e-4          (linear system; only the product matters)
L        = 24.576/5 = 4.9152
```

Profile is **piecewise constant** (the `tanh` in the PIC runs was a numerical
convenience, not the model). Five regions, four interfaces at `X_i = i*L`:

```
R1 = (-inf, X1]   wp = 1.2 m_a
R2 = [X1, X2]     wp = 1.1
R3 = [X2, X3]     wp = 1.0     <- resonant, wp = m_a
R4 = [X3, X4]     wp = 0.9
R5 = [X4, +inf)   wp = 0.8
```

Boundaries: **outgoing at both infinities**. Not the finite PIC box — that is a
simulation limitation, not physics.

---

## 2. What already exists

| file | status |
|---|---|
| `axion_solver.py` | Time-domain Yee/leapfrog solver for the layered problem. **Validated**: reproduces the closed-form uniform-plasma answer to `1.2e-4`, converges at second order (ratio 4.00), domain-independent to `2.3e-6`. This is the reference. |
| `run_main.npz` | Production output: `x`, `t`, `E` (units of `g*B0*a0`), `wp`. Domain `[-90,140]`, `dx=0.01`, to `t=260`. |
| `tmatrix.py` | **Steps 3 of the derivation, done and verified.** `psi_states(p, omega)` returns the state vectors `(psi, psi')` of `psi_-` and `psi_+` at every interface, plus the Wronskian. |
| `analytic.py` | Exact solution for a *uniform* slab on the infinite line (`E_slab`), its driven part (`E_slab_ss`), and an erfc closed form valid for `wp > m_a` (`E_steady`). Independent cross-check. |
| `greens.py` | Numerical ODE integration of `psi_pm`. Superseded by `tmatrix.py` but useful as an independent check. |

`tmatrix.py` passes two tests already:
- uniform medium: reproduces `W = -2i*k*exp(-3i*k*L)` to `5e-16`
- vs ODE integration of a resolved `tanh` profile: converges linearly in `ell`
  (`3.6e-3, 1.0e-3, 2.8e-4` for `ell = 0.2, 0.05, 0.0125`)

---

## 3. What to build

Write `greens_solver.py`. Four pieces, in this order, each gated by its test.

### Piece A — the source, Eqs. (10)–(16)

```python
def S_tilde(x, omega, p):     # Eq. (16): g*B0*[ omega^2 * a_tilde - i*omega*a(x,0) ]
def a_tilde(x, omega, p):     # Eq. (22) for evaluation, NOT Eq. (21)
```

**Critical**: evaluate `a_tilde` via the Faddeeva form, Eq. (22), using
`scipy.special.wofz`. The algebraically equivalent Eq. (21) underflows and
overflows simultaneously for `u > 0`. See `greens_derivation.tex` §8 note 1.

*Test A*: numerically transform `a(x,t)` in time by direct quadrature at a few
`(x, omega)` and compare. Target `1e-10`.

### Piece B — the spatial primitives, Eqs. (39)–(40)

```python
def J1(u, lam, alpha, beta):  # int e^{lam u} erfc(alpha u + beta) du
def J2(u, lam, p):            # int e^{-u^2/2 sigma^2} e^{lam u} du
```

`J1` is already verified against quadrature to `1e-16` for real and complex `lam`
— the formula in Eq. (39) is correct as written. But it needs rearranging for
numerical stability, same reason as Piece A: form `e^{lam u} erfc(alpha u + beta)`
as one scaled call rather than a product.

*Test B*: compare both against `scipy.integrate.quad` on finite intervals with
complex `lam`. Target `1e-12`.

### Piece C — assemble `E_tilde(x, omega)`, Eq. (34)

For each region, get `(P_j, Q_j)` from the state vectors via Eq. (35), then sum
the `J1` and `J2` contributions with
`lam = i*(s*k_a + q_s ± kappa_j)` and `lam = i*(s*k_a ± kappa_j)` respectively.

Use the truncated domain `x_L = -90`, `x_R = 140` rather than taking limits at
infinity. This is exact by the causality argument in §4 of the derivation, not an
approximation.

*Test C1*: set all five `w` equal. `E_tilde` must match the temporal transform of
the exact uniform-slab solution `analytic.E_slab`.
*Test C2*: verify `E_tilde` satisfies Eq. (8) by finite-differencing in `x`.
Target `1e-8`.

### Piece D — the ω inversion, Eq. (43)

```python
def E_y(x, t_array, p, omega_grid)
```

Build `E_tilde` once on the ω grid, then every output time is a sum against
`exp(-i*omega*t)`.

Grid: `d_omega <= 2*pi/260 ≈ 0.024` to resolve `t <= 260`. `omega_max` is set by
the switch-on tail — **measure the decay rate of `|E_tilde|` at large ω, do not
assume it.** Eq. (44) of the derivation says the leading `O(omega)` pieces of the
source cancel, but the residual power is not derived.

*Test D*: against `run_main.npz`. Target `~1e-4`, the time-domain solver's own
accuracy.

---

## 4. Hazards

1. **`M(kappa,d)` is singular as `kappa -> 0`** through `sin(kappa*d)/kappa`. The
   limit is `d`. This occurs at `omega^2 = V_j`, i.e. `omega = 1.0` exactly, which
   is in the middle of the band of interest. Use the series for
   `|kappa*d| < 1e-6`. **This will bite if not handled.**

2. **Overflow/underflow in the erfc expressions**, Pieces A and B. The two places
   are flagged above. Symptom: `nan` or `inf` for `x > x0`.

3. **Poles of `1/W` on the real axis.** Outgoing conditions *should* put all zeros
   of `W` in the lower half plane, but this is not proved in the derivation.
   **Before building Piece D, scan `|W(omega)|` along the real axis over
   `omega in [0, 10]` and check it has no near-zeros.** If it does, stop and
   report — the contour needs rethinking, not nudging with an `epsilon`.

4. **Branch of `kappa` for `omega < 0`.** Don't compute it. Use
   `E_tilde(-omega) = conj(E_tilde(omega))` and integrate over `omega > 0` only,
   Eq. (42). Verify this numerically at a few points first.

---

## 5. Do not do these

- Do not replace outgoing boundaries with a finite box.
- Do not narrowband the source to the Gaussian near `±omega_a`. That discards the
  switch-on transient, which carries roughly half the field amplitude at `t=60` in
  the uniform tests. It is real physics.
- Do not smooth the profile back to `tanh`.
- Do not fall back to time-stepping if the ω integral is awkward. That solver
  already exists; duplicating it answers nothing.

---

## 6. Known results to reproduce

From the time-domain run, peak `|E_y|` per region in units of `g*B0*a0`:

| region | `wp/m_a` | peak | `t_peak` |
|---|---|---|---|
| 1 | 1.2 | 2.40 | 13 |
| 2 | 1.1 | 3.71 | 92 |
| 3 | 1.0 | 6.80 | 134 |
| 4 | 0.9 | 8.80 | 147 |
| 5 | 0.8 | 6.85 | 155 |

The peak sits in region 4, not the resonant region 3. Cause is mode conversion:
the driven response flips sign across the crossing (`+2.30, +4.81, inf, -5.32,
-2.81`), cannot do so instantaneously, and sheds the old field as a free wave that
appears downstream. A correct Green's function solution must show this. If it
doesn't, something is wrong with it — or with the time-domain run, which is also
worth considering.

Note those numbers come from a `tanh` profile with `ell = 0.2`. Sharp steps will
shift them slightly; the measured sensitivity was `4.8e-3` between `ell = 0.05`
and `0.2`, so expect sub-percent differences, not qualitative ones.

---

## 7. Why this is being built

The physics question is how efficiently an axion packet converts to photons
crossing `wp = m_a`. The time-domain solver answers it numerically but gives no
handle on the structure. The frequency-domain construction gives `E_tilde(x,omega)`
in closed form, so the conversion amplitude is readable off the residues and the
parameter dependence is explicit rather than scanned.

A separate structural result already in hand, from the uniform-slab analysis in
`analytic.py`: the resonance denominator is

```
Omega_k^2 - Omega_s^2 = wp^2 - m_a^2 + (k - s*k_a)^2 / gamma_a^2
```

so slabs with `wp < m_a` carry genuine resonances at `delta = ± gamma_a*sqrt(m_a^2 - wp^2)`,
weighted `exp(-sigma_x^2 delta^2/2)`. This predicted the measured resonance
contrast at `sigma_x=1.5` as 1.51 against 1.48 observed. The Green's function
solution should reproduce this structure independently.
