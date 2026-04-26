# Paper outline (draft, for discussion with Chiara)

Working title: *Symmetry and exact initial conditions make parametric PINNs
reliable for families of nonlinear oscillators*

## Claim we can currently support

For one-parameter families of autonomous second-order IVPs where the
parameter is the initial-condition point, a PINN that (i) satisfies the ICs
exactly and (ii) is translation-equivariant by construction, u(t, c) = N(t - c),
trains reliably (12/12 runs, 3 ODEs) where a standard parametric PINN either
fails outright (Van der Pol: 6/6 runs stuck at the trivial solution) or is
8-50x less accurate. The failure of the standard PINN on Van der Pol is
explained by a trivial-solution trap: u = 0 solves the ODE exactly.

## What is probably NOT new (check before writing)

* Hard-constrained trial solutions: Lagaris, Likas & Fotiadis (1998).
* Causal training for time-dependent PINNs: Wang, Sankaran & Perdikaris (2022).
* PINN failure modes / convergence to trivial or wrong solutions:
  Krishnapriyan et al. (NeurIPS 2021); there is also work on fixed points of
  the dynamics as attractors of PINN training (Rohrhofer et al., ~2023,
  verify exact reference). Our trap observation must be positioned against it.
* Symmetry-aware / equivariant PINNs: search before claiming novelty.

Possible contribution then = a clean, reproducible case study + the
observation that, for c-families, equivariance reduces the parametric problem
to a single IVP and removes the trap *and* the seed sensitivity; plus the
two-sided (bidirectional) causal/curriculum view when the IC is interior.

## Sections

1. Introduction: parametric PINNs vs data-driven surrogates (our 2025 report).
2. Problem setting: IVP families, c-families vs parameter-in-f families.
3. Symmetry: autonomy => y(t; c) = Y(t - c); consequences for data-driven
   and physics-informed learning (explains the data-augmentation result).
4. Methods: hard ICs, shift-equivariant network, interior-IC curriculum,
   Adam + L-BFGS.
5. Experiments: `results/benchmark.csv` + ablations below.
6. Discussion: trivial-solution trap; where hard ICs hurt (VdP-mu).

## Missing before this is a paper

- [ ] Data-driven (hybrid) baseline from the report on the same rel. L2 metric.
- [ ] Wall-clock-matched comparisons and more seeds (>= 5).
- [ ] Stiffer regime: VdP with mu up to 5-10; longer horizons.
- [ ] Understand why hard ICs are worse on VdP-mu (scaling of the
      ((t - t0)/T)^2 factor? try T -> characteristic time scale).
- [ ] Non-autonomous example (forced pendulum) where the shift symmetry
      breaks: what is the right inductive bias there?
- [ ] Literature search (see above) and choose venue (workshop paper first?).
- [ ] Re-run the report's PINN figures with the bug fixes, so the 2025 report
      and the paper are consistent.
