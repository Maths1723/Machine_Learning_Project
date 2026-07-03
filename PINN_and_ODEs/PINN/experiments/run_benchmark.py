"""Benchmark PINN variants on the ODE configs.

Example (from PINN_and_ODEs/PINN):

    python experiments/run_benchmark.py --configs data_augmentation_van_der_pol.py \
        --models soft,hard,shift-hard --epochs 6000 --seeds 0,1,2 --curriculum 0.6

Appends one row per (config, model, seed) to results/<csv-name> and stores
the per-parameter error curves in results/curves/.

Model kinds: the pinn_core kinds (legacy, soft, hard, shift, shift-hard, ...)
and hybrid-N, the report's data-driven surrogate trained on N SciPy solutions.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pinn_core import TrainConfig, evaluate, load_problem, make_model, train_pinn  # noqa: E402
from surrogate import train_surrogate  # noqa: E402

FIELDS = ["config", "model", "seed", "epochs", "lbfgs", "curriculum", "causal_eps", "lr",
          "median_rel_l2", "mean_rel_l2", "max_rel_l2", "final_pde", "seconds", "n_data"]

# The original setup, kept for reference: raw inputs, residual on [0, 1], alpha = beta = 1.
LEGACY = dict(alpha=1.0, beta=1.0, colloc_range=(0.0, 1.0), resample=False, n_t=101)


def run_hybrid(problem, kind, seed, args):
    n_data = int(kind.split("-")[1])
    model, info = train_surrogate(problem, n_data, epochs=args.surrogate_epochs, seed=seed, verbose=args.verbose)
    p_vals, errors = evaluate(model, problem, n_p=args.n_eval)
    row = dict(config=problem.name, model=kind, seed=seed, epochs=args.surrogate_epochs, lbfgs=0,
               curriculum=0.0, causal_eps=0.0, lr=1e-3,
               median_rel_l2=float(np.median(errors)), mean_rel_l2=float(errors.mean()),
               max_rel_l2=float(errors.max()), final_pde=float("nan"),
               seconds=round(info["data_seconds"] + info["train_seconds"], 1), n_data=n_data)
    return row, p_vals, errors


def run_one(config_path, kind, seed, args):
    problem = load_problem(config_path)
    if kind.startswith("hybrid-"):
        return run_hybrid(problem, kind, seed, args)
    torch.manual_seed(seed)
    model = make_model(kind, problem, hidden=args.hidden, layers=args.layers, ic_tau=args.ic_tau)
    overrides = dict(LEGACY) if kind == "legacy" else dict(curriculum=args.curriculum, causal_eps=args.causal_eps,
                                                               lbfgs_steps=args.lbfgs)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=seed, log_every=max(args.epochs // 4, 1), **overrides)
    t0 = time.time()
    history = train_pinn(model, problem, cfg, verbose=args.verbose)
    seconds = time.time() - t0
    p_vals, errors = evaluate(model, problem, n_p=args.n_eval)
    row = dict(config=problem.name, model=kind, seed=seed, epochs=args.epochs,
               lbfgs=0 if kind == "legacy" else args.lbfgs,
               curriculum=0.0 if kind == "legacy" else args.curriculum,
               causal_eps=0.0 if kind == "legacy" else args.causal_eps, lr=args.lr,
               median_rel_l2=float(np.median(errors)), mean_rel_l2=float(errors.mean()),
               max_rel_l2=float(errors.max()), final_pde=history[-1]["pde"], seconds=round(seconds, 1), n_data=0)
    return row, p_vals, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True, help="comma-separated config files")
    ap.add_argument("--models", default="legacy,soft,hard,shift-hard")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lbfgs", type=int, default=0, help="L-BFGS refinement iterations after Adam")
    ap.add_argument("--curriculum", type=float, default=0.0)
    ap.add_argument("--causal-eps", type=float, default=0.0)
    ap.add_argument("--ic-tau", type=float, default=1.0, help="time scale of the saturating IC profile (*-sat kinds)")
    ap.add_argument("--surrogate-epochs", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=21)
    ap.add_argument("--csv-name", default="benchmark.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(os.path.join(args.out, "curves"), exist_ok=True)
    csv_path = os.path.join(args.out, args.csv_name)
    new_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for config in args.configs.split(","):
            path = config if os.path.isabs(config) else os.path.join(HERE, config)
            for kind in args.models.split(","):
                if kind.startswith("shift") and not load_problem(path).ic_at_param:
                    print(f"skipping {kind} on {config}: parameter is not the IC point")
                    continue
                for seed in (int(s) for s in args.seeds.split(",")):
                    row, p_vals, errors = run_one(path, kind, seed, args)
                    writer.writerow(row)
                    fh.flush()
                    tag = f"{row['config']}__{kind}__s{seed}__e{args.epochs}__lb{row['lbfgs']}__cur{row['curriculum']}__eps{row['causal_eps']}"
                    with open(os.path.join(args.out, "curves", tag + ".json"), "w") as cf:
                        json.dump({"p": p_vals.tolist(), "rel_l2": errors.tolist(), **row}, cf)
                    print(f"{row['config']:32s} {kind:11s} seed {seed}  median {row['median_rel_l2']:.3e}  "
                          f"max {row['max_rel_l2']:.3e}  ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
