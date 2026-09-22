# Paper outline (draft, for discussion with Chiara) — revised September 2026

Revised after the related-work pass (`related_work.md`) and the summer
experiments (`research_log.md`, June-September). The April framing
("symmetry + hard ICs make parametric PINNs reliable") no longer works as a
headline: the trap is known (Rohrhofer et al., TMLR 2023; Babic, Rohrhofer &
Geiger 2025), and on c-families a single ODE solve plus a shift beats every
learned model (~1e-6 in < 0.05 s).

Working title (candidate): *Parametric PINNs vs data-driven surrogates for
families of nonlinear oscillators: what the comparison depends on*

Format: short paper / workshop note (ML4PS-style) or a teaching-oriented
article. Not a methods paper.

## What we can support (3-5 seeds, one architecture, CPU)

1. **Bugs matter.** The 2025 report's PINN failures were largely due to an
   implementation bug (residual only on [0, 1]); fixed, the PINN works on two
   of the three c-families with plain soft ICs.
2. **Trap (known; we reproduce it).** VdP with small IC: soft ICs 8/8 runs
   stuck at u = 0.
3. **Exact ICs with a bounded profile** are the ingredient that transfers to
   families without symmetry:
   VdP-mu 1.1e-3 (soft 4.9e-3, surrogate-32 2.9e-3);
   pendulum damping 2.4e-3 (surrogate-32 2.2e-3);
   forced pendulum 1.1e-3 (soft 1.0e-2, surrogate-32 1.4e-3).
   The quadratic profile was worse than soft on VdP-mu — profile matters.
4. **Equivariance vs coordinates.** On c-families exact translation
   equivariance gives 5/5 VdP successes; giving the network t - c as an input
   is not enough (seed-sensitive, 3-6x less accurate). On the forced pendulum
   the shift model fails (0.54), as it must: negative control.
5. **Against the report's surrogate** the best PINN is at best ~2-3x more
   accurate (VdP-mu) and otherwise on par, at ~20x the compute. With a cheap
   solver the surrogate is the pragmatic choice.
6. **Stiff VdP (mu in [2, 5])**: all PINN variants fail in our setup; the
   surrogate degrades to 1.5e-2. Limitation, not explained.

## Possible contribution

A careful, reproducible comparison (code + CSVs) showing that (a) the gap
between PINNs and data-driven surrogates on these families is mostly set by
two design choices — exactness of the ICs and use of the time-translation
symmetry — rather than by the learning paradigm; and (b) the c-family
benchmark used in the 2025 report (and common in teaching) is degenerate.

## Sections

1. Introduction: the 2025 report and what it concluded.
2. Setting: IVP families; c-families vs parameter-in-f vs forced.
3. The symmetry and why c-families are degenerate (one solve suffices).
4. Methods compared: soft / hard (quadratic, saturating) / rel / shift PINNs;
   ReLU surrogate; training protocol.
5. Results: tables from `results/benchmark*.csv`; figures from
   `experiments/make_figures.py`.
6. Discussion: traps (cite, do not claim), compute cost, limitations (stiff).

## Still missing

- [ ] Babic et al. (2025) regulariser as a baseline on VdP — implement from
      the paper.
- [ ] >= 10 seeds for the headline numbers; success rates with intervals.
- [ ] Compute-matched comparison (surrogate with more data / longer training
      vs PINN budget); report wall-clock.
- [ ] tau of the saturating profile: sensitivity study (currently tau = 1).
- [ ] Stiff VdP: residual-based adaptive collocation, longer budget; plot the
      failed solutions to classify the failure.
- [ ] Re-run the report's PINN figures with the bug fixes.
- [ ] Talk to Chiara about scope, authorship and venue before writing.
