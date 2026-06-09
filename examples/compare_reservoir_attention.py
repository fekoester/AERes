import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

from AERes.attention import AttentionTrainer, NodeTrajectoryAttentionTrainer
from AERes.dynamicalSystems import LorenzSimulator
from AERes.reservoir import Reservoir


def parse_sizes(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mse(prediction, target):
    return float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))


def torch_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def reservoir_parameter_count(reservoir):
    return (
        reservoir.adjacency_weight_matrix.size
        + reservoir.input_weights.size
        + reservoir.biases.size
    )


def ridge_parameter_count(number_nodes, target_dim):
    return (number_nodes + 1) * target_dim


def make_dataset(args):
    config = {
        "transient_time": args.transient_time,
        "total_time": args.total_time,
        "num_points": args.num_points,
        "noise_level": args.noise_level,
    }
    simulator = LorenzSimulator(function_name=args.system, config=config)
    return simulator.simulate_lorenz()


def train_classic_reservoir(reservoir, y_train, x_test, y_test, alpha):
    model = Ridge(alpha=alpha)
    model.fit(reservoir.states_stand, y_train)

    reservoir.run_given_inputs(x_test, calc_standardize=False)
    prediction = model.predict(reservoir.states_stand)
    score = mse(prediction, y_test)
    params = reservoir_parameter_count(reservoir) + ridge_parameter_count(
        reservoir.number_nodes,
        y_train.shape[1],
    )
    return score, params


def train_old_aer(reservoir, y_train, x_test, y_test, args):
    trainer = AttentionTrainer(
        reservoir.states_stand,
        y_train,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_gpu=not args.cpu,
        layer_type=args.old_layer,
    )
    trainer.train(epochs=args.old_epochs)

    reservoir.run_given_inputs(x_test, calc_standardize=False)
    prediction = trainer.predict(reservoir.states_stand)
    score = mse(prediction, y_test)
    params = reservoir_parameter_count(reservoir) + torch_parameter_count(trainer.model)
    return score, params


def train_node_trajectory_attention(reservoir, y_train, x_test, y_test, args):
    trainer = NodeTrajectoryAttentionTrainer(
        reservoir.states_stand,
        y_train,
        window_size=args.window_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_gpu=not args.cpu,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    trainer.train(epochs=args.new_epochs)

    reservoir.run_given_inputs(x_test, calc_standardize=False)
    prediction = trainer.predict(reservoir.states_stand)
    score = mse(prediction, y_test[args.window_size - 1:])
    params = reservoir_parameter_count(reservoir) + torch_parameter_count(trainer.model)
    return score, params


def plot_results(results, output_dir):
    styles = {
        "Classic reservoir": {"marker": "o", "color": "#1f77b4"},
        "Old AER": {"marker": "s", "color": "#d62728"},
        "Node trajectory attention": {"marker": "^", "color": "#2ca02c"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in results.items():
        sizes = [row["reservoir_size"] for row in rows]
        scores = [row["mse"] for row in rows]
        ax.plot(sizes, scores, label=name, **styles[name])
    ax.set_xlabel("Reservoir size")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mse_over_reservoir_size.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in results.items():
        params = [row["parameters"] for row in rows]
        scores = [row["mse"] for row in rows]
        ax.plot(params, scores, label=name, **styles[name])
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Test MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mse_over_parameter_count.png", dpi=180)
    plt.close(fig)


def print_summary(results):
    print("\nResults")
    print("architecture,reservoir_size,parameters,test_mse")
    for architecture, rows in results.items():
        for row in rows:
            print(
                f"{architecture},{row['reservoir_size']},"
                f"{row['parameters']},{row['mse']:.8g}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare ridge, old AER, and node-trajectory attention reservoirs."
    )
    parser.add_argument("--reservoir-sizes", default="16,32,64,128", type=parse_sizes)
    parser.add_argument("--output-dir", default="plots/reservoir_attention_comparison")
    parser.add_argument("--system", default="lorenz", choices=["lorenz", "lorenz_coupled", "lorenz96"])
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--total-time", type=float, default=150)
    parser.add_argument("--transient-time", type=float, default=10)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--old-epochs", type=int, default=25)
    parser.add_argument("--new-epochs", type=int, default=25)
    parser.add_argument("--old-layer", default="linear", choices=["linear", "nonlinear"])
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "flatten", "attention", "cls"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = make_dataset(args)
    results = {
        "Classic reservoir": [],
        "Old AER": [],
        "Node trajectory attention": [],
    }

    for number_nodes in args.reservoir_sizes:
        print(f"\nReservoir size: {number_nodes}")

        reservoir = Reservoir(
            x_train,
            number_nodes=number_nodes,
            input_dimension=x_train.shape[1],
            seed=args.seed,
            standardize=True,
        )
        score, params = train_classic_reservoir(
            reservoir,
            y_train,
            x_test,
            y_test,
            args.ridge_alpha,
        )
        results["Classic reservoir"].append({
            "reservoir_size": number_nodes,
            "parameters": params,
            "mse": score,
        })
        print(f"  Classic reservoir MSE: {score:.6g} ({params} params)")

        reservoir = Reservoir(
            x_train,
            number_nodes=number_nodes,
            input_dimension=x_train.shape[1],
            seed=args.seed,
            standardize=True,
        )
        score, params = train_old_aer(reservoir, y_train, x_test, y_test, args)
        results["Old AER"].append({
            "reservoir_size": number_nodes,
            "parameters": params,
            "mse": score,
        })
        print(f"  Old AER MSE: {score:.6g} ({params} params)")

        reservoir = Reservoir(
            x_train,
            number_nodes=number_nodes,
            input_dimension=x_train.shape[1],
            seed=args.seed,
            standardize=True,
        )
        score, params = train_node_trajectory_attention(
            reservoir,
            y_train,
            x_test,
            y_test,
            args,
        )
        results["Node trajectory attention"].append({
            "reservoir_size": number_nodes,
            "parameters": params,
            "mse": score,
        })
        print(f"  Node trajectory attention MSE: {score:.6g} ({params} params)")

    plot_results(results, output_dir)
    print_summary(results)
    print(f"\nSaved plots to {output_dir}")


if __name__ == "__main__":
    main()
