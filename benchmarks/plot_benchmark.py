"""Turns results.csv into plots and a markdown table (mean ± std over seeds)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {"ridge": ("#1f77b4", "o"), "attention (AER)": ("#d62728", "s"),
          "node-trajectory attention": ("#2ca02c", "^")}
TITLES = {"lorenz": "Lorenz-63", "rossler": "Rössler", "mackey_glass": "Mackey-Glass (τ=17)"}


def load(path):
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["system"], r["readout"], int(r["nodes"]))
            data[key]["nrmse"].append(float(r["nrmse"]))
            data[key]["vpt"].append(float(r["vpt_lyapunov"]))
    return data


def main(path, only=None):
    data = load(path)
    out = Path(path).parent
    systems = [s for s in TITLES if any(k[0] == s for k in data) and (not only or s in only)]
    readouts = [r for r in STYLES if any(k[1] == r for k in data)]

    fig, axes = plt.subplots(2, len(systems), figsize=(4.2 * len(systems), 7), squeeze=False)
    for col, system in enumerate(systems):
        for readout in readouts:
            sizes = sorted(k[2] for k in data if k[0] == system and k[1] == readout)
            if not sizes:
                continue
            color, marker = STYLES[readout]
            for row, metric in enumerate(["nrmse", "vpt"]):
                values = [data[(system, readout, n)][metric] for n in sizes]
                mean = np.array([np.mean(v) for v in values])
                std = np.array([np.std(v) for v in values])
                ax = axes[row, col]
                ax.plot(sizes, mean, marker=marker, color=color, label=readout)
                ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
        axes[0, col].set_title(TITLES[system])
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("one-step test NRMSE")
        axes[1, col].set_ylabel("valid prediction time [Lyapunov times]")
        for row in range(2):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].set_xlabel("reservoir nodes")
            axes[row, col].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=150)

    lines = ["| system | nodes | readout | NRMSE ↓ | VPT (Λt) ↑ |", "|---|---:|---|---:|---:|"]
    for system in systems:
        for n in sorted({k[2] for k in data if k[0] == system}):
            for readout in readouts:
                v = data.get((system, readout, n))
                if v:
                    lines.append(f"| {TITLES[system]} | {n} | {readout} | "
                                 f"{np.mean(v['nrmse']):.4f} ± {np.std(v['nrmse']):.4f} | "
                                 f"{np.mean(v['vpt']):.2f} ± {np.std(v['vpt']):.2f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/results.csv"
    main(path, sys.argv[2].split(",") if len(sys.argv) > 2 else None)
