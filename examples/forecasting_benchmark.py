"""
Compare ridge and attention readouts on every built-in dynamical system.

Reports one-step test NRMSE and the closed-loop valid prediction time (in time units of each system).

    python examples/forecasting_benchmark.py --nodes 200 --epochs 50
"""
import argparse

import numpy as np
import torch

from AERes import (AttentionTrainer, DynamicalSystemSimulator, Reservoir, RidgeTrainer, nrmse,
                   valid_prediction_time)


def run(system, args):
    config = {"total_time": args.total_time, "num_points": args.num_points}
    simulator = DynamicalSystemSimulator(system, config=config, seed=args.seed)
    X_train, Y_train, X_test, Y_test = simulator.simulate()
    dt = args.total_time / (args.num_points - 1)

    reservoir = Reservoir(X_train, number_nodes=args.nodes, input_dimension=X_train.shape[1],
                          leakage=args.leakage, seed=args.seed, standardize=True)
    readouts = {
        "ridge": RidgeTrainer(reservoir.states_stand, Y_train, alpha=args.alpha),
        "attention": AttentionTrainer(reservoir.states_stand, Y_train, use_gpu=not args.cpu),
    }
    readouts["ridge"].train()
    readouts["attention"].train(epochs=args.epochs, verbose=False)

    reservoir.run_given_inputs(X_test, calc_standardize=False)
    rows = []
    for name, readout in readouts.items():
        one_step = nrmse(readout.predict(reservoir.states_stand), Y_test)
        reservoir.reset_state()
        forecast = reservoir.forecast(readout, X_test[:args.warmup + 1], steps=args.horizon)
        vpt = valid_prediction_time(forecast, Y_test[args.warmup:args.warmup + args.horizon], dt=dt)
        rows.append((system, name, one_step, vpt))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default=",".join(DynamicalSystemSimulator.SYSTEMS))
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--leakage", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--total-time", type=float, default=500)
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"{'system':<22}{'readout':<11}{'NRMSE':>10}{'VPT':>10}")
    for system in args.systems.split(","):
        for system_name, readout, error, vpt in run(system.strip(), args):
            print(f"{system_name:<22}{readout:<11}{error:>10.3e}{vpt:>10.2f}")


if __name__ == "__main__":
    main()
