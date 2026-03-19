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
