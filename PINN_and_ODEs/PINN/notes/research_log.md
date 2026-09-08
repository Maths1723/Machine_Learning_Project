# PINN research log

Follow-up to the Exercise 7 report (Fusar Bassini & Galimberti). Goal: understand
*why* the PINN got stuck on qualitatively wrong solutions while the hybrid
(data-driven) model did not, and whether the PINN can be made competitive.

Setting throughout: second-order IVP families

    y'' = f(t, y, y'),   y(c) = g0,  y'(c) = g1,   c in [c_min, c_max],

learned by a single network u(t, c) on t in [t_min, t_max].

---

## 2026-03-03 – 03-10 · Bugs found in the original PINN code

1. **Collocation on [0, 1] only.** `NoDataLoss` sampled the residual on
   `linspace(0, 1, 101)` while every config evaluates on [0, 10] or [0, 15].
   The physics was imposed on at most 10% of the domain.
2. **alpha / beta ignored.** The values typed at the prompt were overwritten
   by `alpha = beta = 1`.
3. **SciPy "ground truth" wrong for t < c.** The forward-only `solve_ivp`
   dense output was evaluated outside its interval (extrapolation).
4. `np.sin` in the pendulum configs crashes on autograd tensors.

(2) and (4) are cosmetic; (1) and (3) change the *conclusions* of the report:
some of the "PINN is qualitatively wrong" plots compared against an
extrapolated reference, and the PINN never saw the ODE on most of the plot.

## 2026-03-19 · Legacy vs fixed, Van der Pol (mu = 1, y(c) = 0.1, y'(c) = 0)

`pinn_core.py`, 4x64 tanh, Adam 1500 steps, 9 test values of c,
relative L2 error vs DOP853 on [0, 10]:

| variant | final loss | median rel. L2 | max rel. L2 |
|---|---|---|---|
| legacy (raw inputs, residual on [0,1], alpha=beta=1) | 1.6e-6 | 1.13 | 1.25 |
| fixed (normalised inputs, residual on [0,10], alpha=beta=10) | 2.0e-3 | 1.00 | 1.00 |

Two different failures:

* legacy: tiny loss, wrong solution — the loss simply does not see t > 1.
  This is exactly the "low error functional, wrong solution" in the report.
* fixed: rel. error = 1.00 means u ≈ 0. **y ≡ 0 is an exact solution of
  Van der Pol** (unstable equilibrium). With a small IC amplitude 0.1, the
  zero function pays only alpha·0.01 at t = c and zero residual elsewhere,
  so it is a deep local minimum of the soft-constrained loss.

Hypothesis: the stuck PINN is a *trivial-solution trap*, not a capacity issue.
Next: make the ICs exact (hard constraint) so u ≡ 0 is no longer admissible,
and make training respect causality away from t = c.

## 2026-03-24 · Observation: the c-family is one trajectory

All our configs (pendulum, Van der Pol, Rayleigh) are **autonomous**:
f = f(y, y') does not depend on t. If Y solves Y'' = f(Y, Y'), Y(0) = g0,
Y'(0) = g1, then y(t; c) := Y(t - c) satisfies the same ODE and
y(c) = g0, y'(c) = g1. By uniqueness (f is C^1, so locally Lipschitz)

    y(t; c) = Y(t - c)   for all t, c.

Consequences:

* The "parameter-to-solution map" c -> y(.; c) learned in the report is just a
  translation operator applied to a single function Y on
  s in [t_min - c_max, t_max - c_min] (= [-2, 10] for VdP).
* This explains why the hybrid model generalised so well with "even minimal
  data augmentation": shifting one trajectory generates the whole dataset.
  It also means that the c-family is a weak test of *parametric* learning:
  the intrinsic dimension of the solution manifold is 1.
* For a PINN we can hard-code it: u(t, c) = N(t - c) (`ShiftPINN`). The
  residual of the shifted net is then the residual of a single-IVP PINN on
  the s-interval, evaluated at points sampled from every c at once.

Two follow-ups:
1. A genuinely parametric benchmark needs the parameter inside f (e.g. the
   VdP mu, the pendulum damping). Add such configs.
2. With the symmetry built in, any remaining failure is a property of the
   single IVP, not of the parametrisation: a cleaner experiment.

Also added `HardIC`: u = g0 + g1 (t - c) + ((t - c)/T)^2 N(t, c), which
satisfies both ICs exactly for any N (no alpha/beta to tune).

## 2026-04-14 · Benchmark: what actually fixes the PINN

`experiments/run_benchmark.py`, 4x64 tanh, 6000 Adam steps (cosine LR from
1e-3) + 1000 L-BFGS iterations, 3 seeds, 21 test parameters, relative L2
error against DOP853. Raw data in `results/benchmark.csv`, per-parameter
curves in `results/curves/`, table in `results/summary.md`, figure
`results/error_vs_param.png`. Median over seeds (worst seed's max in brackets):

**c-families (parameter = IC point)**

| | legacy | soft ICs | hard ICs | shift + hard ICs |
|---|---|---|---|---|
| VdP, + curriculum | – | 1.00 (1.01) | 1.6e-2 (0.62) | **3.5e-3** (5.6e-3) |
| VdP, no curriculum | 1.18 (1.36) | 1.00 (1.00) | 0.21 (0.91) | **1.4e-3** (1.4e-2) |
| pendulum, + curriculum | 2.58 (3.88) | 2.5e-2 (0.11) | – | **5.3e-4** (1.0e-3) |
| Rayleigh, + curriculum | 1.49 (1.55) | 7.2e-4 (5.5e-3) | – | **8.7e-5** (2.5e-4) |

**Genuinely parametric families (parameter inside f, ICs at t = 0)**

| | soft ICs | hard ICs |
|---|---|---|
| VdP, mu in [0.5, 2] | **4.9e-3** (2.5e-2) | 1.1e-2 (3.3e-2) |
| pendulum, damping in [0.05, 0.6] | 5.9e-2 (0.33) | **3.5e-3** (1.4e-2) |

Reading:

1. **The original PINN fails everywhere** (rel. error > 1), on all three ODEs.
   This is mostly the [0, 1] collocation bug, not an intrinsic PINN weakness.
2. **Van der Pol with soft ICs is stuck at u ≈ 0 for every seed and every
   training variant** (error 1.00, 6/6 runs), confirming the trivial-solution
   trap. Pendulum and Rayleigh do not have it (larger IC amplitude / no
   repelling equilibrium near the IC), and soft ICs already work there.
3. **Hard ICs alone are not enough**: they escape the trap but are
   seed-dependent (VdP: 4e-3 to 0.29 with curriculum, worse without).
4. **Shift structure + hard ICs works on every run**: 12/12 runs below
   1.4e-2, typically 1e-4 – 1e-3: 8x (Rayleigh) to 50x (pendulum) better than
   soft ICs, and the only variant that reliably solves VdP. With the
   symmetry built in the curriculum is no longer needed.
5. Shift-hard error curves are flat in c (see figure), as they must be: every
   c sees the same learned function, so the error is that of a single IVP.
6. On the parametric families hard ICs help a lot for the damped pendulum
   (17x) but are slightly worse for VdP-mu. Not yet understood; candidate
   explanation: the ((t - t0)/T)^2 factor makes the network output large late
   in the interval where VdP-mu has sharp relaxation fronts for mu = 2.

Caveats before claiming anything: one architecture, one budget, 3 seeds,
CPU float32. Needs: wall-clock-matched comparison, a data-driven (hybrid)
baseline from the report on the same error metric, and larger mu (stiff VdP).

## 2026-06-16 · Data-driven baseline on the same metric

`surrogate.py` = the report's hybrid model (ReLU 3x128, p -> solution on a
400-point grid), trained 5000 Adam steps on N DOP853 solutions at equispaced
parameters; tested on the same 21 parameters and rel. L2 as the PINNs (test
parameters mostly fall *between* training ones). `results/benchmark_hybrid.csv`.
Median over 3 seeds:

| family | hybrid-8 | hybrid-32 | best PINN (April table) |
|---|---|---|---|
| VdP, c-family | 6.5e-3 | 2.7e-3 | shift-hard 1.4e-3 – 3.5e-3 |
| pendulum, c-family | 3.9e-3 | 1.3e-3 | shift-hard **5.3e-4** |
| Rayleigh, c-family | 3.5e-3 | 2.0e-3 | shift-hard **8.7e-5** |
| VdP, mu | 3.2e-3 | **2.9e-3** | soft 4.9e-3 |
| pendulum, damping | 3.0e-3 | **2.2e-3** | hard 3.5e-3 |

Cost: ~40 s per surrogate (data generation < 1 s of that) vs ~10-20 min per
PINN on the same CPU.

Honest reading:
* With a cheap, accurate ODE solver available, **the data-driven surrogate is
  the better tool for the parametric families**: better or equal accuracy at
  ~20x less compute. The PINN's "no data needed" advantage is weak when data
  costs one second.
* On c-families the symmetric PINN is more accurate (2.5x pendulum, 22x
  Rayleigh, comparable on VdP) — but the surrogate does not exploit the
  symmetry either. Fair follow-up: a surrogate that predicts Y(s) once and
  shifts it ("shift-hybrid"); it would need ONE solve. Expect it to win.
* So a paper cannot be "PINNs beat surrogates". Possible angles: (a) settings
  where the solver is expensive or data unavailable; (b) the methodological
  point that symmetry, not the learning paradigm, drives accuracy on these
  families — for both PINNs and surrogates.

## 2026-06-30 · The c-family benchmark is (too) easy for any method that knows the symmetry

Follow-up to the "shift-hybrid" idea: under y(t; c) = Y(t - c) no network is
needed at all. One DOP853 solve of Y on s in [t_min - c_max, t_max - c_min]
plus linear interpolation on a 4001-point s-grid gives the whole family with
rel. L2 error ~1e-6 (VdP 1.2e-6, pendulum 1.6e-6, Rayleigh 7.9e-7; limited by
the interpolation, not the solver) in < 0.05 s. So on c-families:

* any learned surrogate that does not use the symmetry is solving an
  artificially hard problem (that is what the 2025 report did);
* any method that does use it reduces to a single IVP, and the comparison
  becomes "PINN vs classical solver on one IVP", which the solver wins.

Consequence for the paper: c-families are useful as a *didactic* example of
how symmetry turns a failing PINN into a working one, not as evidence that
PINNs are competitive. Real tests need families without the symmetry:
parameters inside f (done: VdP-mu, pendulum damping) and non-autonomous
forcing (next).

## 2026-07-14 · Bounded IC profile fixes the VdP-mu regression

April's hard ICs were worse than soft on VdP-mu. Hypothesis: the factor
(s/T)^2 multiplies the network output by up to 1 at the end of the interval
and makes late-time corrections stiff. New `HardIC(profile="saturating")`:
u = g0 + g1 s + (1 - exp(-(s/tau)^2)) N, tau = 1. Same budget as April, 3
seeds (`results/benchmark_summer.csv`):

| family | soft (Apr) | hard, quadratic (Apr) | **hard, saturating** | hybrid-32 (Jun) |
|---|---|---|---|---|
| VdP, mu | 4.9e-3 | 1.1e-2 | **1.1e-3** (worst 8.2e-3) | 2.9e-3 |
| pendulum, damping | 5.9e-2 | 3.5e-3 | **2.4e-3** (worst 2.1e-2) | 2.2e-3 |

The quadratic profile was the problem: with the bounded one, hard ICs are
4x better than soft on VdP-mu and **2.6x better than the data-driven
surrogate** — the first parametric family where the PINN wins on accuracy
(it still costs ~20x more compute). On the pendulum it ties the surrogate.
tau = 1 was not tuned; the worst-case (max over p) errors are still 7-9x
the medians, concentrated at the ends of the parameter range.

## 2026-07-28 · Coordinates or symmetry? (RelativePINN ablation)

`rel-hard`: hard ICs + network on (t - c, c). It can represent y = Y(t - c)
but is not forced to. Same budget, 3 seeds, c-families:

| family | soft | hard (t, c) | **rel-hard (t - c, c)** | shift-hard (t - c only) |
|---|---|---|---|---|
| VdP | 1.00 | 1.6e-2 (worst 0.62) | 1.2e-2 (worst 0.80) | **3.5e-3** (worst 5.6e-3) |
| pendulum | 2.5e-2 | – | 3.1e-3 (worst 1.2e-2) | **5.3e-4** (worst 1.0e-3) |
| Rayleigh | 7.2e-4 | – | 3.2e-4 (worst 1.4e-3) | **8.7e-5** (worst 2.5e-4) |

* The coordinates alone give a real but partial gain: 2-8x over soft ICs on
  pendulum/Rayleigh, about the same as plain hard ICs on VdP.
* They do **not** give robustness: one of three VdP seeds fails (0.45 median),
  exactly the seed-dependence plain hard ICs showed in April.
* Exact equivariance is what makes it reliable and 3-6x more accurate.
  The network that is *allowed* to ignore c does not learn to ignore it
  within this budget.

## 2026-08-14 · Forced pendulum: what survives when the symmetry breaks

`forced_pendulum.py`: y'' + 0.2 y' + sin y = 0.5 cos(1.2 t), y(c) = 0.5,
y'(c) = 0, c in [0, 2], t in [0, 15]. Non-autonomous, so y(t; c) != Y(t - c).
Same budget, 3 seeds:

| method | median rel. L2 | worst seed, max over c |
|---|---|---|
| soft | 1.0e-2 | 4.0e-2 |
| **hard (t, c)** | **1.1e-3** | 7.2e-3 |
| rel-hard (t - c, c) | 1.4e-3 | 4.6e-3 |
| shift-hard (t - c only) | 0.54 | 0.98 |
| hybrid-8 | 2.6e-3 | 1.8e-2 |
| hybrid-32 | 1.4e-3 | 3.4e-3 |

* shift-hard fails on every seed (0.54, identical across seeds): it can only
  represent translates of one function, which this family is not. Expected,
  and a useful negative control — the April gains were not an artefact.
* Hard ICs are the ingredient that transfers: 9x better than soft, on par
  with the surrogate trained on 32 solutions (and better than 8).
* rel-hard is not better than plain hard here; the coordinates mattered
  only through the symmetry.
* No trap here (y = 0 is not a solution of the forced equation), so soft ICs
  "work" but are an order of magnitude less accurate.

## 2026-08-31 · Stiff Van der Pol: every PINN variant fails

`parametric_vdp_mu_stiff.py`: mu in [2, 5], y(0) = 1, y'(0) = 0, t in [0, 20]
(relaxation oscillations, fronts of width ~1/mu). Same budget, 3 seeds:

| method | median rel. L2 | worst |
|---|---|---|
| soft | 1.03 | 1.06 |
| hard (quadratic) | 1.04 | 1.09 |
| hard (saturating) | 1.21 | 1.41 |
| hybrid-8 | 8.8e-2 | 0.14 |
| hybrid-32 | 1.5e-2 | 3.6e-2 |

* All PINNs fail on all seeds (errors >= 1). For hard ICs u = 0 is excluded
  by construction, so this is not the trivial-solution trap; whether it is a
  wrong period or a smoothed-out solution was not checked yet (plot needed).
* The surrogate degrades too (5x worse than on mu in [0.5, 2]) but remains
  usable with 32 samples.
* Not a surprise: the 6000-step budget, fixed 4x64 tanh network and uniform
  collocation are all poorly suited to sharp fronts. Before calling it a
  limitation of the method we would
  need adaptive collocation (residual-based resampling) and a longer budget —
  out of scope for now. Report it as a limitation of this setup.

## 2026-09-08 · Five seeds on the Van der Pol c-family

Seeds 3 and 4 added (curriculum + L-BFGS, same budget), pooled with April's
seeds 0-2. Median rel. L2 per seed:

| model | seed 0 | 1 | 2 | 3 | 4 | runs < 2e-2 |
|---|---|---|---|---|---|---|
| soft | 1.01 | 1.00 | 1.00 | 1.00 | 1.06 | 0/5 |
| hard | 1.6e-2 | 4.3e-3 | 0.29 | 5.2e-3 | 1.0e-2 | 4/5 |
| shift-hard | 3.5e-3 | 3.9e-4 | 4.9e-3 | 4.6e-3 | 1.3e-3 | **5/5** |

Unchanged conclusions with more seeds: soft ICs never escape the trap (8/8
soft runs over both training settings), hard ICs usually do but not always,
shift-hard always does. Five seeds is still small; for a paper use >= 10
and report success rates with intervals.
