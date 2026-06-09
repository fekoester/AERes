from AERes.attention import NodeTrajectoryAttentionTrainer
from AERes.dynamicalSystems import LorenzSimulator
from AERes.reservoir import Reservoir


lorenz_simulator = LorenzSimulator(function_name='lorenz')
X_train, Y_train, X_test, Y_test = lorenz_simulator.simulate_lorenz()

reservoir = Reservoir(
    X_train,
    number_nodes=50,
    input_dimension=X_train.shape[1],
    seed=1,
    standardize=True,
)

window_size = 32
trainer = NodeTrajectoryAttentionTrainer(
    reservoir.states_stand,
    Y_train,
    window_size=window_size,
    d_model=64,
    nhead=4,
    pooling="mean",
)
trainer.train(epochs=20)

reservoir.run_given_inputs(X_test, calc_standardize=False)
trainer.evaluate(reservoir.states_stand, Y_test)

predictions = trainer.predict(reservoir.states_stand)
print(f"Predictions shape: {predictions.shape}")
