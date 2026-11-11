"""Minimal end-to-end example: Lorenz data -> reservoir -> ridge vs. attention readout -> closed-loop forecast."""
import matplotlib.pyplot as plt
import torch

from AERes import (AttentionTrainer, DynamicalSystemSimulator, Reservoir, RidgeTrainer,
                   valid_prediction_time)

torch.manual_seed(0)

# Simulate the Lorenz system as a basic example of a chaotic dynamical system
simulator = DynamicalSystemSimulator('lorenz', seed=0)
X_train, Y_train, X_test, Y_test = simulator.simulate()
dt = simulator.config["total_time"] / (simulator.config["num_points"] - 1)

# Create the reservoir and drive it with the training inputs
reservoir = Reservoir(X_train, number_nodes=50, input_dimension=X_train.shape[1], seed=1, standardize=True)

# Classic ridge readout and attention readout share the same interface
ridge = RidgeTrainer(reservoir.states_stand, Y_train, alpha=1.0)
ridge.train()
attention = AttentionTrainer(reservoir.states_stand, Y_train, layer_type="linear")
attention.train(epochs=100)

# Evaluate on the test set (re-using the training standardization)
reservoir.run_given_inputs(X_test, calc_standardize=False)
print(f"Ridge test MSE:     {ridge.evaluate(reservoir.states_stand, Y_test, verbose=False):.3e}")
print(f"Attention test MSE: {attention.evaluate(reservoir.states_stand, Y_test, verbose=False):.3e}")

# Closed-loop forecast: synchronise on 250 test steps, then run autonomously
warmup, horizon = 250, 500
reservoir.reset_state()
prediction = reservoir.forecast(attention, X_test[:warmup + 1], steps=horizon)
truth = Y_test[warmup:warmup + horizon]
vpt = valid_prediction_time(prediction, truth, dt=dt, lyapunov_exponent=0.906)
print(f"Valid prediction time: {vpt:.2f} Lyapunov times")

plt.plot(truth[:, 0], label="truth")
plt.plot(prediction[:, 0], "--", label="attention forecast")
plt.legend()
plt.show()

# Save / load the trained model
# attention.save_model('aeres_attention.pt')
# attention.load_model('aeres_attention.pt')
