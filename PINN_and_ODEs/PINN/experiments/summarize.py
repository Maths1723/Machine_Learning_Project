"""Aggregate benchmark CSVs into a markdown table (median over seeds) and a figure.

    python experiments/summarize.py results/benchmark.csv [--plot results/error_vs_param.png]
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def setting(row):
    tricks = []
    if float(row["curriculum"]) > 0:
        tricks.append("curriculum")
    if float(row["causal_eps"]) > 0:
        tricks.append(f"causal eps={float(row['causal_eps']):g}")
    if int(row.get("lbfgs") or 0) > 0:
        tricks.append("L-BFGS")
    return " + ".join(tricks) or "-"


def table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["model"], setting(r))].append(r)
    lines = ["| config | model | training | seeds | median rel. L2 (over seeds) | worst seed, max over p |",
             "|---|---|---|---|---|---|"]
    for (config, model, train), rs in sorted(groups.items()):
        med = np.median([float(r["median_rel_l2"]) for r in rs])
        worst = max(float(r["max_rel_l2"]) for r in rs)
        lines.append(f"| {config} | {model} | {train} | {len(rs)} | {med:.2e} | {worst:.2e} |")
    return "\n".join(lines)


def plot_curves(curve_dir, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_config = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(curve_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        by_config[d["config"]].append(d)
    fig, axes = plt.subplots(1, len(by_config), figsize=(5 * len(by_config), 4), squeeze=False)
    for ax, (config, curves) in zip(axes[0], sorted(by_config.items())):
        styles = {}
        for d in curves:
            label = f"{d['model']} ({setting(d)})"
            color = styles.setdefault(label, f"C{len(styles)}")
            ax.semilogy(d["p"], d["rel_l2"], color=color, alpha=0.7,
                        label=label if label not in ax.get_legend_handles_labels()[1] else None)
        ax.set_title(config)
        ax.set_xlabel("parameter")
        ax.set_ylabel("relative L2 error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--plot", help="output png (reads curves/ next to the first csv)")
    args = ap.parse_args()
    print(table(load_rows(args.csv)))
    if args.plot:
        plot_curves(os.path.join(os.path.dirname(args.csv[0]), "curves"), args.plot)


if __name__ == "__main__":
    main()
