"""
Benchmark: ridge vs. attention (AER) vs. node-trajectory attention readouts.

For each system, reservoir size and seed it records
  - one-step test NRMSE
  - closed-loop valid prediction time (VPT, in Lyapunov times), averaged over several forecast start points
and appends one CSV row per (system, size, seed, readout) so partial runs are kept.

    python benchmarks/run_benchmark.py --out benchmarks/results/results.csv
    python benchmarks/plot_benchmark.py benchmarks/results/results.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from AERes import (AttentionTrainer, DynamicalSystemSimulator, NodeTrajectoryAttentionTrainer, Reservoir,
                   RidgeTrainer, nrmse, valid_prediction_time)

# Approximate largest Lyapunov exponents (per unit time) used to express VPT in Lyapunov times.
LYAPUNOV = {"lorenz": 0.906, "rossler": 0.0714, "mackey_glass": 0.0086}
FIELDS = ["system", "nodes", "seed", "readout", "parameters", "nrmse", "vpt_lyapunov", "train_seconds"]


def fit_ridge(states, targets, alphas=(1e-6, 1e-4, 1e-2, 1.0)):
    """Picks the ridge alpha on the last 20 % of the training data, then refits on everything."""
    split = int(0.8 * len(states))
    best = min(alphas, key=lambda a: _val_error(states, targets, split, a))
    ridge = RidgeTrainer(states, targets, alpha=best)
    ridge.train()
    return ridge


def _val_error(states, targets, split, alpha):
    ridge = RidgeTrainer(states[:split], targets[:split], alpha=alpha)
    ridge.train()
    return ridge.evaluate(states[split:], targets[split:], verbose=False)


def mean_vpt(reservoir, readout, X_test, Y_test, dt, lyapunov, starts, warmup, horizon):
    values = []
    for start in starts:
        reservoir.reset_state()
        forecast = reservoir.forecast(readout, X_test[start:start + warmup + 1], steps=horizon)
        truth = Y_test[start + warmup:start + warmup + horizon]
        values.append(valid_prediction_time(forecast, truth, dt=dt, lyapunov_exponent=lyapunov))
    return float(np.mean(values))


def run_config(system, nodes, seed, args, writer):
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = {"total_time": args.total_time, "num_points": args.num_points}
    X_train, Y_train, X_test, Y_test = DynamicalSystemSimulator(system, config=config, seed=seed).simulate()
    dt = args.total_time / (args.num_points - 1)
    starts = np.linspace(0, len(X_test) - args.warmup - args.horizon - 1, args.forecasts).astype(int)

    reservoir = Reservoir(X_train, number_nodes=nodes, input_dimension=X_train.shape[1], seed=seed,
                          standardize=True)
    train_states = reservoir.states_stand.copy()
    reservoir_params = reservoir.adjacency_weight_matrix.size + reservoir.input_weights.size + nodes

    readouts = {}
    t = time.time()
    readouts["ridge"] = (fit_ridge(train_states, Y_train), time.time() - t)

    t = time.time()
    aer = AttentionTrainer(train_states, Y_train, learning_rate=args.lr)
    aer.train(epochs=args.aer_epochs, verbose=False)
    readouts["attention (AER)"] = (aer, time.time() - t)

    if nodes <= args.node_max_nodes:
        t = time.time()
        ntr = NodeTrajectoryAttentionTrainer(train_states, Y_train, window_size=args.window, d_model=32,
                                             nhead=4, learning_rate=args.lr)
        ntr.train(epochs=args.node_epochs, verbose=False)
        readouts["node-trajectory attention"] = (ntr, time.time() - t)

    reservoir.run_given_inputs(X_test, calc_standardize=False)
    test_states = reservoir.states_stand.copy()
    for name, (readout, seconds) in readouts.items():
        if hasattr(readout, "model"):
            params = sum(p.numel() for p in readout.model.parameters())
        else:
            params = readout.parameter_count()
        prediction = readout.predict(test_states)
        error = nrmse(prediction, Y_test[len(Y_test) - len(prediction):])
        vpt = mean_vpt(reservoir, readout, X_test, Y_test, dt, LYAPUNOV[system], starts, args.warmup,
                       args.horizon)
        row = dict(system=system, nodes=nodes, seed=seed, readout=name, parameters=reservoir_params + params,
                   nrmse=f"{error:.6g}", vpt_lyapunov=f"{vpt:.4g}", train_seconds=f"{seconds:.1f}")
        writer.writerow(row)
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="lorenz,rossler,mackey_glass")
    parser.add_argument("--sizes", default="25,50,100,200")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--total-time", type=float, default=400)
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--aer-epochs", type=int, default=150)
    parser.add_argument("--node-epochs", type=int, default=20)
    parser.add_argument("--node-max-nodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--forecasts", type=int, default=5)
    parser.add_argument("--out", default="benchmarks/results/results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["system"], int(r["nodes"]), int(r["seed"])) for r in csv.DictReader(f)}
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for system in args.systems.split(","):
            for nodes in map(int, args.sizes.split(",")):
                for seed in map(int, args.seeds.split(",")):
                    if (system, nodes, seed) in done:
                        continue
                    run_config(system, nodes, seed, args, writer)
                    f.flush()


if __name__ == "__main__":
    main()
