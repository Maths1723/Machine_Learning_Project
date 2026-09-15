"""Figures for the write-up.

    python experiments/make_figures.py                 # both figures
    python experiments/make_figures.py --only summary  # just the summary chart (no training)

1. results/figures/summary.png: median rel. L2 error (over seeds) per family and
   method, from every results/*.csv.
2. results/figures/trap_vdp.png: illustrative Van der Pol c-family solution at
   c = 1 for the soft PINN (trivial-solution trap), the shift + hard-IC PINN and
   the data-driven surrogate, against DOP853. Trains its own (shorter) models,
   so the errors differ slightly from the benchmark tables.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# methods shown in the summary chart, in this order (others are ignored)
METHODS = ["legacy", "soft", "hard", "hard-sat", "rel-hard", "shift-hard", "hybrid-8", "hybrid-32"]


def summary_figure(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.csv"))):
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    # keep the main training setting for PINNs (curriculum + L-BFGS), all hybrids
    groups = defaultdict(list)
    for r in rows:
        is_pinn = not r["model"].startswith("hybrid")
        if is_pinn and r["model"] != "legacy" and float(r["curriculum"]) == 0:
            continue
        groups[(r["config"], r["model"])].append(float(r["median_rel_l2"]))
    configs = sorted({c for c, _ in groups})
    width = 0.8 / len(METHODS)
    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 2, 4.2))
    for j, m in enumerate(METHODS):
        xs, ys = [], []
        for i, c in enumerate(configs):
            if (c, m) in groups:
                xs.append(i + (j - len(METHODS) / 2 + 0.5) * width)
                ys.append(np.median(groups[(c, m)]))
        if xs:
            ax.bar(xs, ys, width, label=m, color=f"C{j}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace("data_augmentation_", "c-family: ").replace("_", " ") for c in configs],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median relative L2 error")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def trap_figure(out_dir, epochs, lbfgs):
    import torch

    from numerics import reference_solution
    from pinn_core import TrainConfig, load_problem, make_model, predict, train_pinn
    from surrogate import train_surrogate

    torch.set_num_threads(4)
    pb = load_problem(os.path.join(HERE, "data_augmentation_van_der_pol.py"))
    t = np.linspace(pb.t_min, pb.t_max, 400)
    c = 1.0
    y_ref = reference_solution(pb.ode_func, pb.g0, pb.g1, c, t)[0]
    curves = {}
    for kind in ("soft", "shift-hard"):
        model = make_model(kind, pb)
        train_pinn(model, pb, TrainConfig(epochs=epochs, lbfgs_steps=lbfgs, curriculum=0.6, seed=0), verbose=False)
        curves[f"PINN, {kind}"] = predict(model, t, c)
    surrogate, _ = train_surrogate(pb, 32, epochs=5000, seed=0)
    curves["surrogate, 32 solutions"] = predict(surrogate, t, c)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, y_ref, "k-", lw=2.5, alpha=0.35, label="DOP853 reference")
    for (label, u), ls in zip(curves.items(), ["--", "-", ":"]):
        err = np.linalg.norm(u - y_ref) / np.linalg.norm(y_ref)
        ax.plot(t, u, ls, lw=1.4, label=f"{label} (rel. L2 {err:.1e})")
    ax.axvline(c, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("Van der Pol, y(c) = 0.1, y'(c) = 0, c = 1")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path = os.path.join(out_dir, "trap_vdp.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["summary", "trap"])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=500)
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    if args.only in (None, "summary"):
        summary_figure(out_dir)
    if args.only in (None, "trap"):
        trap_figure(out_dir, args.epochs, args.lbfgs)


if __name__ == "__main__":
    main()
