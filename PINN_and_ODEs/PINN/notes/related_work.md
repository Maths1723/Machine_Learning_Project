# Related work (checked June 2026)

Literature pass for `paper_outline.md`. Bottom line: the trivial-solution
trap is known, and so is using Lie symmetries in PINNs for ODEs. What is left
for us is narrower (see "Positioning").

## Fixed points as traps for PINN training

* **Rohrhofer et al.** (full author list: check), *On the Role of Fixed Points of
  Dynamical Systems in Training Physics-Informed Neural Networks*, TMLR 2023
  (arXiv:2203.13648). Fixed points (constant solutions of ODEs, steady states
  of PDEs) trivially satisfy the physics loss and create local optima in the
  loss landscape; this explains non-physical PINN predictions.
  -> exactly our "trivial-solution trap" on Van der Pol (u = 0). Cite it, do
  not present the trap as a finding.
* **Babic, Rohrhofer & Geiger**, *Stabilizing PINNs: A regularization scheme
  for PINN training to avoid unstable fixed points of dynamical systems*,
  arXiv:2509.11768 (2025). Adds a decaying regulariser that penalises
  predictions near *unstable* fixed points; tested on four systems including
  Van der Pol, improving the training success rate.
  -> the natural baseline for our VdP results. Our approach is different
  (no extra loss term: exact ICs + equivariance), but we must compare.

## Failure modes and training remedies

* **Krishnapriyan, Gholami, Zhe, Kirby & Mahoney**, *Characterizing possible
  failure modes in physics-informed neural networks*, NeurIPS 2021
  (arXiv:2109.01050). Failures come from the optimisation landscape, not
  expressivity; curriculum regularisation and sequence-to-sequence (time
  marching) training help. Our growing-window curriculum is in this family.
* **Wang, Sankaran & Perdikaris**, *Respecting causality is all you need for training
  physics-informed neural networks* (arXiv 2022). Causal weights; our two-sided
  variant for interior ICs is a small adaptation.
* **Lagaris, Likas & Fotiadis**, IEEE TNN 1998. Trial solutions that satisfy
  the ICs/BCs exactly; our `HardIC` is this construction.

## Symmetry

* **Arora, Bihlo & Valiquette**, *Invariant Physics-Informed Neural Networks
  for Ordinary Differential Equations*, JMLR 25(233), 2024
  (arXiv:2310.17053). Uses Lie point symmetries and equivariant moving frames
  to invariantise the ODE, solves the reduced equation + reconstruction
  equations with a PINN; clearly outperforms non-invariant PINNs.
  -> time translation is a Lie point symmetry of every autonomous ODE, so
  `ShiftPINN` is a (very) special case of this philosophy.

## Positioning

Not new: the trap; hard ICs; causal/curriculum training; symmetry in PINNs.

Possibly new / worth a short paper or workshop note:
1. *Parametric families*: for IVP families indexed by the IC time c, the
   symmetry collapses the whole family to one IVP, which (in our benchmark)
   is what turns a seed-dependent method into a reliable one. The relevant
   comparison is against a **data-driven surrogate of the same family** (our
   2025 report), on the same error metric.
2. A controlled ablation separating the ingredients (exact ICs, coordinates
   t - c, exact equivariance, curriculum, L-BFGS).
3. What to do when the symmetry is broken (non-autonomous forcing).

To do: implement the Babic et al. regulariser as a baseline (from the paper,
not from memory) before any comparison on VdP.
