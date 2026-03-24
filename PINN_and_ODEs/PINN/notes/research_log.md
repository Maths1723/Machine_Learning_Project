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
